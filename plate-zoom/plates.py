"""Поиск номерной пластины внутри (увеличенного) кадра машины."""
import cv2


class PlateFinder:
    def __init__(self, weights=None, device=None):
        self.model = None
        if weights:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.device = device
            print(f"[i] Детектор номеров: {weights}")
        else:
            print("[i] Детектор номеров: контурный (без нейросети). Для точности укажи --plate-weights")

    def find(self, car):
        """Возвращает (x1, y1, x2, y2) номера в координатах car или None."""
        if self.model is not None:
            r = self.model(car, conf=0.25, verbose=False, device=self.device)[0]
            if len(r.boxes) == 0:
                return None
            i = int(r.boxes.conf.argmax())
            return tuple(int(v) for v in r.boxes.xyxy[i].tolist())
        return self._contour(car)

    @staticmethod
    def _contour(img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch == 0:
                continue
            ar = cw / ch
            # номер: вытянутый прямоугольник, в нижних 2/3 машины
            if 2.0 <= ar <= 6.0 and 0.12 * w < cw < 0.7 * w and y > 0.3 * h:
                if cw * ch > best_area:
                    best, best_area = (x, y, x + cw, y + ch), cw * ch
        return best
