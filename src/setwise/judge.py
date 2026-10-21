"""Transport/cache only. Prompt construction and score parsing belong to RR."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import urllib.request


class JudgeError(RuntimeError):
    pass


class OpenAIJudge:
    """One request per call; retry and parse rules are owned by upstream RR."""
    def __init__(self, config):
        self.config = config
        self.stats = {"judge_calls": 0, "cache_hit": 0}

    def _request(self, prompt_text):
        # Exactly the RR request shape; no new system prompt or decoding options.
        payload = {"model": self.config["model"], "messages": [{"role": "user", "content": prompt_text}]}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        cache_key = hashlib.sha256(self.config["base_url"].encode() + encoded).hexdigest()
        return encoded, cache_key

    def _database(self):
        path = Path(self.config["cache_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=60)
        db.execute("CREATE TABLE IF NOT EXISTS rr_responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        return db

    def complete(self, prompt_text):
        cfg = self.config
        if cfg["model"].startswith("UNSET"):
            raise JudgeError("Set a pinned judge model before API use")
        encoded, cache_key = self._request(prompt_text)
        db = self._database()
        try:
            found = db.execute("SELECT value FROM rr_responses WHERE key=?", (cache_key,)).fetchone()
        finally:
            db.close()
        if found:
            self.stats["cache_hit"] += 1
            return found[0]
        key = os.environ.get(cfg["api_key_env"])
        if not key:
            raise JudgeError(f"Missing credential environment variable: {cfg['api_key_env']}")
        self.stats["judge_calls"] += 1
        request = urllib.request.Request(cfg["base_url"].rstrip("/") + "/chat/completions", data=encoded,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=cfg["timeout_seconds"]) as response:
                result = json.load(response)
            value = result["choices"][0]["message"]["content"]
            if not isinstance(value, str) or not value:
                raise ValueError("Empty judge response")
            return value
        except Exception:
            raise JudgeError("Judge request failed") from None

    def remember(self, prompt_text, response_text):
        """Only cache responses accepted by the unchanged upstream parser."""
        _, key = self._request(prompt_text)
        db = self._database()
        try:
            with db:
                db.execute("INSERT OR REPLACE INTO rr_responses VALUES (?,?)", (key, response_text))
        finally:
            db.close()
