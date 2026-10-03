"""Model loading, preprocessing and the diagnosis rule.

The ONNX file comes from notebook cell 28.6 (app_model.onnx). /255, ImageNet
normalisation and the temperature are baked into it, so the server feeds raw
0-255 RGB pixels, exactly like the phone does with the TFLite twin.
The decision rule here is the single source of truth; the Flutter app mirrors it
in lib/core/verdict.dart for offline use.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HEALTHY_TOKENS = ('healthy', 'health', 'fresh', 'normal', 'disease_free')


def is_healthy(label: str) -> bool:
    return any(t in label.lower() for t in HEALTHY_TOKENS)


def crop_of(label: str) -> str:
    return label.split('__', 1)[0] if '__' in label else 'unknown'


@dataclass
class ModelBundle:
    session: object
    labels: list[str]
    size: int
    reject_below: float
    min_crop_mass: float
    version: str
    nchw: bool
    metrics: dict = field(default_factory=dict)

    @property
    def crops(self) -> list[str]:
        return sorted({crop_of(l) for l in self.labels})


def load_bundle(model_dir: str | Path) -> ModelBundle:
    import onnxruntime as ort
    d = Path(model_dir)
    cfg = json.loads((d / 'app_config.json').read_text(encoding='utf-8'))
    labels = [l.strip() for l in (d / 'labels.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
    sess = ort.InferenceSession(str(d / 'app_model.onnx'), providers=['CPUExecutionProvider'])
    shape = sess.get_inputs()[0].shape            # [1,3,S,S] (NCHW) or [1,S,S,3]
    nchw = shape[1] == 3
    size = int(cfg.get('input', {}).get('shape', [1, 300, 300, 3])[1])
    n_out = sess.get_outputs()[0].shape[-1]
    if isinstance(n_out, int) and n_out != len(labels):
        raise ValueError(f'model has {n_out} outputs but labels.txt has {len(labels)} lines')
    return ModelBundle(session=sess, labels=labels, size=size,
                       reject_below=float(cfg.get('reject_below_confidence', 0.70)),
                       min_crop_mass=float(cfg.get('min_crop_mass', 0.30)),
                       version=str(cfg.get('model_version') or cfg.get('split_id') or cfg.get('model', 'unknown')),
                       nchw=nchw, metrics=cfg.get('metrics', {}))


def preprocess(data: bytes, size: int, nchw: bool) -> np.ndarray:
    """Decode -> EXIF orientation -> RGB -> squash-resize (bilinear, antialiased) -> float32 0..255."""
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert('RGB')
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(im, dtype=np.float32)[None]          # 1,S,S,3
    return x.transpose(0, 3, 1, 2).copy() if nchw else x


def run(bundle: ModelBundle, data: bytes) -> np.ndarray:
    x = preprocess(data, bundle.size, bundle.nchw)
    name = bundle.session.get_inputs()[0].name
    p = bundle.session.run(None, {name: x})[0][0].astype(np.float64)
    s = p.sum()
    return p / s if s > 0 else p


def decide(labels: list[str], probs: np.ndarray, reject_below: float, crop: str | None = None,
           min_crop_mass: float = 0.30, k: int = 3) -> dict:
    """not_sure | wrong_crop | healthy | disease, plus the top-k it is based on.

    If the farmer named the crop, probabilities are renormalised inside that crop:
    the model no longer has to tell rice from cotton, only which rice disease.
    If the model puts less than `min_crop_mass` on that crop at all, the photo
    probably is not that crop and the answer is `wrong_crop`.
    """
    probs = np.asarray(probs, dtype=np.float64)
    idx = np.arange(len(labels))
    crop_mass = None
    if crop:
        keep = np.array([crop_of(l) == crop for l in labels])
        if keep.any():
            crop_mass = float(probs[keep].sum())
            idx = idx[keep]
            probs = np.where(keep, probs, 0.0)
            probs = probs / probs.sum() if probs.sum() > 0 else probs
    order = idx[np.argsort(-probs[idx])][:k]
    top = [{'label': labels[i], 'confidence': round(float(probs[i]), 4), 'healthy': is_healthy(labels[i])}
           for i in order]
    if crop_mass is not None and crop_mass < min_crop_mass:
        verdict = 'wrong_crop'
    elif top[0]['confidence'] < reject_below:
        verdict = 'not_sure'
    elif top[0]['healthy']:
        verdict = 'healthy'
    else:
        verdict = 'disease'
    return {'verdict': verdict, 'top': top, 'crop_mass': crop_mass}
