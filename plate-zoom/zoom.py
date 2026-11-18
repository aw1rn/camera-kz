"""Цифровой зум: вырезка области + увеличение (суперразрешение или бикубика)."""
import os
import re
import cv2
import numpy as np


def crop_with_margin(img, box, margin=0.1):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    return img[y1:y2, x1:x2].copy(), (x1, y1)


def sharpen(img, amount=0.6):
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def enhance_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


class Upscaler:
    """Если указан файл модели (например FSRCNN_x4.pb, ESPCN_x3.pb, EDSR_x4.pb) —
    использует нейросетевое суперразрешение OpenCV (нужен opencv-contrib-python).
    Иначе — бикубическое увеличение + повышение резкости."""

    def __init__(self, model_path=None):
        self.sr = None
        self.scale = 1
        if model_path:
            m = re.match(r"([A-Za-z]+)_x(\d)", os.path.basename(model_path))
            if not m:
                raise ValueError("Имя файла модели должно быть вида FSRCNN_x4.pb")
            try:
                self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            except AttributeError:
                print("[!] Нет cv2.dnn_superres — установи opencv-contrib-python. Работаю без нейросети.")
                return
            self.sr.readModel(model_path)
            self.scale = int(m.group(2))
            self.sr.setModel(m.group(1).lower(), self.scale)
            print(f"[i] Суперразрешение: {m.group(1)} x{self.scale}")

    def zoom(self, img, min_h):
        """Увеличивает картинку, пока высота меньше min_h."""
        if img.size == 0 or img.shape[0] >= min_h:
            return img
        out = img
        if self.sr is not None:
            for _ in range(2):  # максимум два прохода
                if out.shape[0] >= min_h:
                    break
                out = self.sr.upsample(out)
        if out.shape[0] < min_h:
            f = min_h / out.shape[0]
            out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        return sharpen(out)


def fit(img, w, h, bg=30):
    """Вписывает картинку в прямоугольник w×h с сохранением пропорций."""
    canvas = np.full((h, w, 3), bg, np.uint8)
    if img is None or img.size == 0:
        return canvas
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas
