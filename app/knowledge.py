"""Farmer-facing text: disease names, what to do first, and the verdict sentences, in bn and en.

knowledge/diseases.json is shared with the Flutter app (assets/knowledge/diseases.json),
so the offline answer and the server answer read the same.
"""
from __future__ import annotations

import json
from pathlib import Path

KB = Path(__file__).resolve().parents[1] / 'knowledge' / 'diseases.json'


class Knowledge:
    def __init__(self, path: Path = KB):
        d = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        self.diseases: dict = d.get('diseases', {})
        self.crops: dict = d.get('crops', {})
        self.verdicts: dict = d.get('verdicts', {})
        self.helpline: dict = d.get('helpline', {})

    @staticmethod
    def _pretty(label: str) -> str:
        return label.split('__', 1)[-1].replace('_', ' ').capitalize()

    def _entry(self, label: str) -> dict:
        # exact class first, then a crop-independent entry ('*__downy_mildew') shared by several crops
        return self.diseases.get(label) or self.diseases.get('*__' + label.split('__')[-1]) or {}

    def name(self, label: str, lang: str) -> str:
        e = self._entry(label)
        if not e and label.endswith('__healthy'):    # healthy has no advice entry, but needs a name
            return 'সুস্থ পাতা' if lang == 'bn' else 'Healthy leaf'
        return e.get(f'name_{lang}') or e.get('name_en') or self._pretty(label)

    def crop_name(self, crop: str, lang: str) -> str:
        c = self.crops.get(crop, {})
        return c.get(lang) or c.get('en') or crop.replace('_', ' ').capitalize()

    def advice(self, label: str, lang: str) -> dict | None:
        e = self._entry(label)
        if not e:
            return None
        return {'what': e.get(f'what_{lang}') or e.get('what_en'),
                'steps': e.get(f'steps_{lang}') or e.get('steps_en') or [],
                'source': e.get('source')}

    def verdict_text(self, verdict: str, lang: str, crop: str | None) -> str:
        t = self.verdicts.get(verdict, {}).get(lang) or self.verdicts.get(verdict, {}).get('en') or verdict
        return t.replace('{crop}', self.crop_name(crop, lang) if crop else '')
