"""Распознавание текста номера + нормализация под казахстанский формат."""
import re

ALLOWED = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Частые ошибки OCR: буква вместо цифры и наоборот
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G",
             "7": "T", "4": "A", "3": "E"}

# D — цифра, L — буква.
#   123 ABC 01 — физлица (с 2012 г.)
#   123 AB 01  — юрлица
TEMPLATES = ["DDDLLLDD", "DDDLLDD"]


def _coerce(s, tpl):
    """Подгоняет строку под шаблон. Возвращает (результат, число исправлений) или None."""
    out, fixes = [], 0
    for ch, t in zip(s, tpl):
        if t == "D" and not ch.isdigit():
            ch, fixes = TO_DIGIT.get(ch, ch), fixes + 1
            if not ch.isdigit():
                return None
        elif t == "L" and not ch.isalpha():
            ch, fixes = TO_LETTER.get(ch, ch), fixes + 1
            if not ch.isalpha():
                return None
        out.append(ch)
    return "".join(out), fixes


def _valid_region(code):
    return code.isdigit() and 1 <= int(code) <= 20


def normalize(raw):
    """'KZ 123 A8C 01' -> '123ABC01'. Возвращает None, если не похоже на номер."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if s.startswith("KZ") and len(s) > 8:
        s = s[2:]  # надпись KZ под флагом
    best = None  # (исправлений, -длина, позиция, номер)
    for tpl in TEMPLATES:
        n = len(tpl)
        for i in range(len(s) - n + 1):
            r = _coerce(s[i:i + n], tpl)
            if r and _valid_region(r[0][-2:]):
                cand = (r[1], -n, i, r[0])
                if best is None or cand < best:
                    best = cand
    return best[3] if best else None


def pretty(plate):
    """'123ABC01' -> '123 ABC 01'."""
    if not plate:
        return ""
    return f"{plate[:3]} {plate[3:-2]} {plate[-2:]}"


class PlateOCR:
    def __init__(self, gpu=False):
        import easyocr  # импорт здесь, чтобы normalize() можно было тестировать без easyocr
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, img):
        """Возвращает (номер, уверенность) или (None, 0)."""
        res = self.reader.readtext(img, allowlist=ALLOWED, detail=1, paragraph=False)
        if not res:
            return None, 0.0
        res.sort(key=lambda r: min(p[0] for p in r[0]))  # слева направо
        raw = "".join(r[1] for r in res)
        conf = sum(r[2] for r in res) / len(res)
        plate = normalize(raw)
        return (plate, float(conf)) if plate else (None, 0.0)
