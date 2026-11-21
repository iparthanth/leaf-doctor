"""Leaf Doctor server: the phone sends a photo, the server answers in Bangla.

  GET  /health                               liveness + model version
  GET  /api/v1/meta                          crops, classes, threshold, measured metrics
  POST /api/v1/diagnose                      1-3 photos (+ crop, district) -> verdict, top-3, advice, case id
  POST /api/v1/cases/{id}/feedback           farmer says whether the answer was right (the data loop)
  DELETE /api/v1/cases/{id}                  farmer deletes the case and its photo
  GET  /api/v1/cases/{id}                    the case (needs its access key)
  POST /api/v1/cases/{id}/messages           farmer asks an expert about this case
  GET  /api/v1/cases/{id}/messages           the thread
  GET  /api/v1/expert/cases                  open questions            (expert token)
  POST /api/v1/expert/cases/{id}/reply       expert answers            (expert token)

Configuration (environment variables):
  MODEL_DIR     folder with app_model.onnx, labels.txt, app_config.json (notebook 28.6)
  DATA_DIR      where the SQLite file and case thumbnails go
  EXPERT_TOKEN  shared secret for the expert endpoints; expert endpoints are off if unset
  MAX_UPLOAD_MB upload limit, default 6
  KEEP_AWAKE_URL  this server's public address; it opens its own /health every 10 minutes so a free host
                (Render sleeps after 15 idle minutes) never sleeps. Render sets RENDER_EXTERNAL_URL itself.
"""
from __future__ import annotations

import hmac
import io
import os
import re
import threading
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import engine
from .knowledge import Knowledge
from .store import Store

MODEL_DIR = os.environ.get('MODEL_DIR', str(Path(__file__).resolve().parents[1] / 'model'))
DATA_DIR = Path(os.environ.get('DATA_DIR', str(Path(__file__).resolve().parents[1] / 'data')))
EXPERT_TOKEN = os.environ.get('EXPERT_TOKEN', '')
MAX_UPLOAD = int(float(os.environ.get('MAX_UPLOAD_MB', '6')) * 1024 * 1024)
RATE_PER_MIN = int(os.environ.get('RATE_PER_MIN', '30'))

app = FastAPI(title='Leaf Doctor', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET', 'POST'], allow_headers=['*'])

_bundle: engine.ModelBundle | None = None
_store = Store(DATA_DIR / 'leafdoctor.sqlite3')
_kb = Knowledge()
_hits: dict[str, deque] = defaultdict(deque)

# Keep a free host awake: the request goes out to the public address and back in, so the host counts it as a
# visitor. ponytail: cannot wake a server that is already asleep - add an outside monitor if that ever happens.
KEEP_AWAKE_URL = (os.environ.get('KEEP_AWAKE_URL') or os.environ.get('RENDER_EXTERNAL_URL', '')).rstrip('/')


def _keep_awake(url: str, every_s: int = 600):
    while True:
        time.sleep(every_s)
        try:
            urllib.request.urlopen(f'{url}/health', timeout=60).read()
        except Exception:
            pass   # a missed ping is retried next round


if KEEP_AWAKE_URL:
    threading.Thread(target=_keep_awake, args=(KEEP_AWAKE_URL,), name='keep-awake', daemon=True).start()


def bundle() -> engine.ModelBundle:
    global _bundle
    if _bundle is None:
        try:
            _bundle = engine.load_bundle(MODEL_DIR)
        except FileNotFoundError as e:
            raise HTTPException(503, f'model not installed: {e.filename}')
    return _bundle


def rate_limit(request: Request):
    # ponytail: in-memory per-IP window; move to Redis if this ever runs on more than one worker
    ip = request.client.host if request.client else 'unknown'
    q, now = _hits[ip], time.monotonic()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        raise HTTPException(429, 'too many requests, try again in a minute')
    q.append(now)


def expert(authorization: str = Header(default='')):
    if not EXPERT_TOKEN:
        raise HTTPException(404, 'expert endpoints are disabled')
    if not hmac.compare_digest(authorization.removeprefix('Bearer ').strip(), EXPERT_TOKEN):
        raise HTTPException(401, 'bad expert token')


def own_case(case_id: str, key: str) -> dict:
    c = _store.case(case_id)
    if not c or not hmac.compare_digest(c['access_key'], key or ''):
        raise HTTPException(404, 'no such case')
    return c


@app.get('/expert', include_in_schema=False)
def expert_desk():
    # the page itself holds no data; every call it makes needs the expert token
    return FileResponse(Path(__file__).parent / 'static' / 'expert.html')


@app.get('/health')
def health():
    try:
        b = bundle()
        return {'status': 'ok', 'model_version': b.version, 'classes': len(b.labels)}
    except HTTPException as e:
        return {'status': 'no_model', 'detail': e.detail}


@app.get('/api/v1/meta')
def meta(lang: str = 'bn'):
    b = bundle()
    return {'model_version': b.version, 'input_size': b.size, 'reject_below': b.reject_below,
            'crops': [{'id': c, 'name': _kb.crop_name(c, lang)} for c in b.crops],
            'classes': [{'id': l, 'crop': engine.crop_of(l), 'name': _kb.name(l, lang),
                         'healthy': engine.is_healthy(l)} for l in b.labels],
            'metrics': b.metrics, 'helpline': _kb.helpline}


@app.post('/api/v1/diagnose', dependencies=[Depends(rate_limit)])
async def diagnose(image: list[UploadFile] = File(...), crop: str | None = Form(default=None),
                   lang: str = Form(default='bn'), district: str | None = Form(default=None, max_length=40)):
    if not 1 <= len(image) <= 3:
        raise HTTPException(422, 'send 1 to 3 photos of the same plant')
    blobs = []
    for f in image:
        data = await f.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f'photo larger than {MAX_UPLOAD // (1024 * 1024)} MB')
        blobs.append(data)
    b = bundle()
    if crop and crop not in b.crops:
        raise HTTPException(422, f'unknown crop {crop!r}; known: {b.crops}')
    t0 = time.perf_counter()
    try:
        # several photos of one plant: average the probabilities (BRRI's app takes up to 5)
        probs = sum(engine.run(b, d) for d in blobs) / len(blobs)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, 'not an image the server can read')
    result = engine.decide(b.labels, probs, b.reject_below, crop, b.min_crop_mass, reject_by_crop=b.reject_by_crop)
    thumb = _save_thumb(blobs[0])
    case = _store.new_case(crop, result['verdict'], result['top'], b.version, thumb, district,
                           round((time.perf_counter() - t0) * 1000, 1))
    for t in result['top']:
        t['name'] = _kb.name(t['label'], lang)
    top = result['top'][0]['label']
    return {'case_id': case['id'], 'access_key': case['access_key'], 'verdict': result['verdict'],
            'top': result['top'], 'crop_mass': result['crop_mass'],
            'advice': _kb.advice(top, lang) if result['verdict'] == 'disease' else None,
            'message': _kb.verdict_text(result['verdict'], lang, crop),
            'helpline': _kb.helpline, 'model_version': b.version}


def _save_thumb(data: bytes) -> str | None:
    """Keep a small copy for the expert. Re-encoding drops EXIF, including any GPS position."""
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
        im.thumbnail((640, 640))
        d = DATA_DIR / 'cases'
        d.mkdir(parents=True, exist_ok=True)
        p = d / f'{time.time_ns()}.jpg'
        im.save(p, 'JPEG', quality=85)
        return str(p)
    except Exception:
        return None


class Question(BaseModel):
    body: str = Field(min_length=1, max_length=1000)


@app.get('/api/v1/cases/{case_id}')
def get_case(case_id: str, x_case_key: str = Header(default='')):
    c = own_case(case_id, x_case_key)
    c.pop('access_key'); c.pop('image_path', None)
    return c


@app.post('/api/v1/cases/{case_id}/messages', dependencies=[Depends(rate_limit)])
def ask(case_id: str, q: Question, x_case_key: str = Header(default='')):
    own_case(case_id, x_case_key)
    return _store.add_message(case_id, 'farmer', q.body.strip(), new_status='asked')


class Feedback(BaseModel):
    correct: bool
    true_label: str | None = Field(default=None, max_length=80)


@app.post('/api/v1/cases/{case_id}/feedback')
def feedback(case_id: str, f: Feedback, x_case_key: str = Header(default='')):
    own_case(case_id, x_case_key)
    _store.set_feedback(case_id, f.correct, f.true_label)
    return {'ok': True}


@app.delete('/api/v1/cases/{case_id}')
def delete_case(case_id: str, x_case_key: str = Header(default='')):
    c = own_case(case_id, x_case_key)
    if c.get('image_path') and os.path.exists(c['image_path']):
        os.remove(c['image_path'])
    _store.delete(case_id)
    return {'deleted': case_id}


@app.get('/api/v1/cases/{case_id}/messages')
def thread(case_id: str, x_case_key: str = Header(default='')):
    own_case(case_id, x_case_key)
    return _store.messages(case_id)


@app.get('/api/v1/expert/cases', dependencies=[Depends(expert)])
def open_cases():
    return [{**c, 'messages': _store.messages(c['id'])} for c in _store.open_cases()]


@app.post('/api/v1/expert/cases/{case_id}/reply', dependencies=[Depends(expert)])
def reply(case_id: str, q: Question):
    if not _store.case(case_id):
        raise HTTPException(404, 'no such case')
    return _store.add_message(case_id, 'expert', q.body.strip(), new_status='answered')


# ---- farmers' photos labelled by the expert: the data the next model learns from (notebook cell 28.0)
LABEL_RE = re.compile(r'^([a-z][a-z_]*__[a-z0-9_]+|unusable)$')
# Render's free disk is wiped on every deploy: a labelled photo is also copied to this Hugging Face dataset
# (set FARMER_HF_REPO and HF_TOKEN on the host). Without them the desk's export zip is the copy to keep.
FARMER_HF_REPO = os.environ.get('FARMER_HF_REPO', '')


class Label(BaseModel):
    label: str = Field(min_length=3, max_length=80)


def _push_label(case_id: str, label: str, image_path: str | None, old: str | None):
    token = os.environ.get('HF_TOKEN')
    if not (FARMER_HF_REPO and token):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        if old and old not in ('unusable', label):
            try:
                api.delete_file(f'{old}/{case_id}.jpg', FARMER_HF_REPO, repo_type='dataset',
                                commit_message=f'{case_id}: relabelled')
            except Exception:
                pass
        if label != 'unusable' and image_path and os.path.exists(image_path):
            api.upload_file(path_or_fileobj=image_path, path_in_repo=f'{label}/{case_id}.jpg',
                            repo_id=FARMER_HF_REPO, repo_type='dataset', commit_message=f'{case_id}: {label}')
    except Exception as e:      # ponytail: best effort, no retry queue - the export zip still holds the photo
        print(f'farmer photo push failed: {type(e).__name__}')


@app.get('/api/v1/expert/queue', dependencies=[Depends(expert)])
def label_queue(limit: int = 40):
    cases = _store.label_queue(min(max(limit, 1), 200))
    for c in cases:
        c.pop('image_path', None)
        c['photo'] = f"/api/v1/expert/cases/{c['id']}/photo"
    return {'cases': cases, 'labelled': _store.label_counts()}


@app.get('/api/v1/expert/cases/{case_id}/photo', dependencies=[Depends(expert)])
def case_photo(case_id: str):
    c = _store.case(case_id)
    if not c or not c.get('image_path') or not os.path.exists(c['image_path']):
        raise HTTPException(404, 'no photo')
    return FileResponse(c['image_path'], media_type='image/jpeg')


@app.post('/api/v1/expert/cases/{case_id}/label', dependencies=[Depends(expert)])
def label_case(case_id: str, body: Label):
    label = body.label.strip().lower()
    if not LABEL_RE.match(label):
        raise HTTPException(422, "label must be 'crop__disease' in lower case (e.g. rice__blast) or 'unusable'")
    c = _store.case(case_id)
    if not c:
        raise HTTPException(404, 'no such case')
    old = _store.set_label(case_id, label)
    threading.Thread(target=_push_label, args=(case_id, label, c.get('image_path'), old), daemon=True).start()
    return {'id': case_id, 'label': label, 'pushed': bool(FARMER_HF_REPO and os.environ.get('HF_TOKEN'))}


@app.get('/api/v1/expert/export.zip', dependencies=[Depends(expert)])
def export_labelled():
    """Every labelled photo as leaf_doctor_farmer_photos/<crop__disease>/<case>.jpg plus labels.csv: attach the
    zip to Kaggle as a dataset and notebook cell 28.0 trains on it (and 28.5 tests on its held-out part)."""
    import csv
    import zipfile
    from fastapi.responses import Response
    buf, table = io.BytesIO(), io.StringIO()
    w = csv.writer(table)
    w.writerow(['case_id', 'label', 'crop_picked', 'app_answer', 'app_confidence', 'model_version', 'created'])
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as z:        # JPEGs do not compress further
        for c in _store.labelled():     # ponytail: built in memory; stream it once thousands are labelled
            top = c['top'][0] if c['top'] else {}
            w.writerow([c['id'], c['expert_label'], c.get('crop') or '', top.get('label', ''),
                        top.get('confidence', ''), c.get('model_version') or '', round(c['created'])])
            if c['expert_label'] != 'unusable' and c.get('image_path') and os.path.exists(c['image_path']):
                z.write(c['image_path'], f"leaf_doctor_farmer_photos/{c['expert_label']}/{c['id']}.jpg")
        z.writestr('leaf_doctor_farmer_photos/labels.csv', table.getvalue())
    return Response(buf.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="leaf_doctor_farmer_photos.zip"'})


# The web build of the app (flutter build web) and the Android APK, served from this same address,
# so one link gives anyone the app. Mounted last: every API route above wins over a file.
WEB_DIR = Path(os.environ.get('WEB_DIR', str(Path(__file__).resolve().parents[1] / 'web')))
if (WEB_DIR / 'index.html').exists():
    import mimetypes
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    mimetypes.add_type('application/vnd.android.package-archive', '.apk')
    mimetypes.add_type('application/wasm', '.wasm')

    @app.get('/flutter_service_worker.js', include_in_schema=False)
    def retire_service_worker():
        # whatever ?v= an old page asks for: the worker that clears the old app out of the browser
        return FileResponse(Path(__file__).parent / 'static' / 'flutter_service_worker.js',
                            media_type='text/javascript', headers={'Cache-Control': 'no-cache'})

    @app.middleware('http')
    async def revalidate_web_files(request: Request, call_next):
        # the app's file names never change between versions (main.dart.js, ...): make browsers ask every time;
        # an unchanged file costs a 304 thanks to its ETag
        response = await call_next(request)
        if request.method == 'GET' and not request.url.path.startswith('/api/'):
            response.headers.setdefault('Cache-Control', 'no-cache')
        return response

    app.mount('/', StaticFiles(directory=WEB_DIR, html=True), name='web')
