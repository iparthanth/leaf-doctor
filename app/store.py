"""Cases and the farmer <-> expert message thread, in one SQLite file (stdlib only)."""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY, created REAL NOT NULL, crop TEXT, verdict TEXT NOT NULL,
  top_json TEXT NOT NULL, model_version TEXT, image_path TEXT,
  status TEXT NOT NULL DEFAULT 'auto', access_key TEXT NOT NULL,
  district TEXT, latency_ms REAL, feedback_correct INTEGER, feedback_label TEXT
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL REFERENCES cases(id),
  created REAL NOT NULL, sender TEXT NOT NULL CHECK (sender IN ('farmer','expert')), body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_case ON messages(case_id, created);
CREATE INDEX IF NOT EXISTS cases_status ON cases(status, created);
"""


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()   # ponytail: one writer lock; a thesis demo, not a fleet
        with self._lock:
            self._db.executescript(SCHEMA)

    def new_case(self, crop, verdict, top, model_version, image_path, district=None, latency_ms=None) -> dict:
        cid, key = secrets.token_hex(8), secrets.token_urlsafe(16)
        with self._lock, self._db:
            self._db.execute('INSERT INTO cases(id,created,crop,verdict,top_json,model_version,image_path,access_key,'
                             'district,latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?)',
                             (cid, time.time(), crop, verdict, json.dumps(top), model_version, image_path, key,
                              district, latency_ms))
        return {'id': cid, 'access_key': key}

    def case(self, cid: str) -> dict | None:
        r = self._db.execute('SELECT * FROM cases WHERE id=?', (cid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d['top'] = json.loads(d.pop('top_json'))
        return d

    def add_message(self, cid: str, sender: str, body: str, new_status: str | None = None) -> dict:
        now = time.time()
        with self._lock, self._db:
            cur = self._db.execute('INSERT INTO messages(case_id,created,sender,body) VALUES (?,?,?,?)',
                                   (cid, now, sender, body))
            if new_status:
                self._db.execute('UPDATE cases SET status=? WHERE id=?', (new_status, cid))
        return {'id': cur.lastrowid, 'case_id': cid, 'created': now, 'sender': sender, 'body': body}

    def set_feedback(self, cid: str, correct: bool, label: str | None):
        with self._lock, self._db:
            self._db.execute('UPDATE cases SET feedback_correct=?, feedback_label=? WHERE id=?', (int(correct), label, cid))

    def delete(self, cid: str):
        with self._lock, self._db:
            self._db.execute('DELETE FROM messages WHERE case_id=?', (cid,))
            self._db.execute('DELETE FROM cases WHERE id=?', (cid,))

    def messages(self, cid: str) -> list[dict]:
        return [dict(r) for r in self._db.execute(
            'SELECT id,case_id,created,sender,body FROM messages WHERE case_id=? ORDER BY created,id', (cid,))]

    def open_cases(self, limit: int = 50) -> list[dict]:
        rows = self._db.execute("SELECT * FROM cases WHERE status='asked' ORDER BY created LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d['top'] = json.loads(d.pop('top_json'))
            d.pop('access_key', None)
            out.append(d)
        return out
