"""Журнал распознанных номеров (SQLite + снимки)."""
import os
import sqlite3
import time
from datetime import datetime

import cv2


class PlateDB:
    def __init__(self, path="data/plates.db", img_dir="data/plates", dedup_sec=60):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        self.img_dir = img_dir
        self.dedup_sec = dedup_sec
        self.last = {}  # номер -> время последней записи
        self.con = sqlite3.connect(path)
        self.con.execute("""CREATE TABLE IF NOT EXISTS plates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT, conf REAL, votes INTEGER, track_id INTEGER,
            ts TEXT, image TEXT)""")
        self.con.commit()

    def save(self, plate, conf, votes, track_id, img):
        now = time.time()
        if now - self.last.get(plate, 0) < self.dedup_sec:
            return False
        self.last[plate] = now
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{track_id}_{plate}.jpg"
        path = os.path.join(self.img_dir, name)
        if img is not None and img.size:
            cv2.imwrite(path, img)
        self.con.execute("INSERT INTO plates(plate, conf, votes, track_id, ts, image) VALUES(?,?,?,?,?,?)",
                         (plate, conf, votes, track_id, ts, path))
        self.con.commit()
        return True

    def recent(self, n=8):
        return self.con.execute("SELECT plate, ts FROM plates ORDER BY id DESC LIMIT ?", (n,)).fetchall()
