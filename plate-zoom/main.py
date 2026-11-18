"""Распознавание номеров с цифровым зумом на готовом видеопотоке.

Запуск:
    python main.py --source "https://.../stream.m3u8"
    python main.py --source video.mp4 --plate-weights plate.pt --sr-model FSRCNN_x4.pb

Управление в окне:
    ЛКМ по машине  — «камера слежения» (крупный план в правой панели)
    ПКМ            — снять выбор
    + / -          — сильнее / слабее приближение
    Q / Esc        — выход
"""
import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from db import PlateDB
from ocr import PlateOCR, pretty
from plates import PlateFinder
from zoom import Upscaler, crop_with_margin, enhance_for_ocr, fit

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
PANEL_W = 380
VIEW_H = 720


class Track:
    def __init__(self):
        self.box = None
        self.last_seen = 0
        self.votes = defaultdict(float)  # номер -> сумма уверенностей
        self.hits = defaultdict(int)     # номер -> сколько раз прочитан
        self.best_img = {}               # номер -> лучший снимок пластины
        self.best_conf = {}
        self.saved = False

    def leader(self):
        if not self.votes:
            return None
        return max(self.votes, key=self.votes.get)


def open_source(src):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="URL потока (m3u8/rtsp), файл или номер вебкамеры")
    ap.add_argument("--vehicle-model", default="yolov8n.pt")
    ap.add_argument("--plate-weights", default=None, help="YOLO-веса детектора номеров (.pt)")
    ap.add_argument("--sr-model", default=None, help="Модель суперразрешения, напр. FSRCNN_x4.pb")
    ap.add_argument("--device", default=None, help="cpu / 0 (GPU)")
    ap.add_argument("--ocr-every", type=int, default=3, help="OCR каждые N кадров на машину")
    ap.add_argument("--min-car-h", type=int, default=60, help="Меньшие машины пропускаются")
    ap.add_argument("--lock-hits", type=int, default=4, help="Сколько одинаковых чтений = уверенность")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    vehicles = YOLO(args.vehicle_model)
    finder = PlateFinder(args.plate_weights, args.device)
    up = Upscaler(args.sr_model)
    ocr = PlateOCR(gpu=args.device not in (None, "cpu"))
    db = PlateDB()

    tracks = {}
    state = {"selected": None, "follow_zoom": 1.0, "scale": 1.0, "boxes": []}
    last_plate_img, last_plate_text = None, ""

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            fx, fy = x / state["scale"], y / state["scale"]
            for tid, (x1, y1, x2, y2) in state["boxes"]:
                if x1 <= fx <= x2 and y1 <= fy <= y2:
                    state["selected"] = tid
                    break
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["selected"] = None

    if not args.no_show:
        cv2.namedWindow("Plate Zoom", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Plate Zoom", on_mouse)

    cap = open_source(args.source)
    frame_idx, t0, fps = 0, time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[!] Поток прервался, переподключаюсь...")
            cap.release()
            time.sleep(2)
            cap = open_source(args.source)
            continue
        frame_idx += 1

        res = vehicles.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=0.35,
                             tracker="bytetrack.yaml", verbose=False, device=args.device)[0]
        state["boxes"] = []

        if res.boxes.id is not None:
            for xyxy, tid in zip(res.boxes.xyxy.cpu().numpy().astype(int), res.boxes.id.int().cpu().tolist()):
                tr = tracks.setdefault(tid, Track())
                tr.box, tr.last_seen = tuple(xyxy), frame_idx
                state["boxes"].append((tid, tr.box))

                if tr.saved or (frame_idx + tid) % args.ocr_every:
                    continue
                if xyxy[3] - xyxy[1] < args.min_car_h:
                    continue

                # 1) цифровой зум машины
                car, _ = crop_with_margin(frame, xyxy, 0.05)
                car = up.zoom(car, min_h=480)
                # 2) поиск номера
                pb = finder.find(car)
                if pb is None:
                    continue
                plate, _ = crop_with_margin(car, pb, 0.08)
                # 3) зум пластины до читаемого размера
                plate = up.zoom(plate, min_h=120)
                # 4) OCR
                text, conf = ocr.read(enhance_for_ocr(plate))
                if not text:
                    continue
                tr.votes[text] += conf
                tr.hits[text] += 1
                if conf > tr.best_conf.get(text, 0):
                    tr.best_conf[text], tr.best_img[text] = conf, plate
                last_plate_img, last_plate_text = plate, text

                if tr.hits[text] >= args.lock_hits:
                    if db.save(text, tr.best_conf[text], tr.hits[text], tid, tr.best_img[text]):
                        print(f"[+] {pretty(text)}  (ID {tid}, conf {tr.best_conf[text]:.2f})")
                    tr.saved = True

        # Машина уехала: сохраняем лучший вариант
        for tid in [t for t, tr in tracks.items() if frame_idx - tr.last_seen > 30]:
            tr = tracks.pop(tid)
            lead = tr.leader()
            if lead and not tr.saved and tr.hits[lead] >= 2:
                if db.save(lead, tr.best_conf[lead], tr.hits[lead], tid, tr.best_img[lead]):
                    print(f"[+] {pretty(lead)}  (ID {tid}, уехал)")
            if state["selected"] == tid:
                state["selected"] = None

        if args.no_show:
            continue

        # ---------- Отрисовка ----------
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - t0, 1e-6))
        t0 = now

        vis = frame.copy()
        for tid, (x1, y1, x2, y2) in state["boxes"]:
            tr = tracks[tid]
            lead = tr.leader()
            color = (0, 200, 0) if tr.saved else (0, 180, 255)
            if tid == state["selected"]:
                color = (255, 80, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid}" + (f"  {pretty(lead)}" if lead else "")
            cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis, f"FPS {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        state["scale"] = VIEW_H / vis.shape[0]
        vis = cv2.resize(vis, (int(vis.shape[1] * state["scale"]), VIEW_H))

        # Правая панель: камера слежения + лупа номера + журнал
        panel = np.full((VIEW_H, PANEL_W, 3), 25, np.uint8)
        sel = state["selected"]
        cv2.putText(panel, "Camera follow" + (f" ID {sel}" if sel else " (click a car)"),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        if sel in tracks and tracks[sel].box:
            margin = max(0.05, 0.6 / state["follow_zoom"] - 0.1)
            follow, _ = crop_with_margin(frame, tracks[sel].box, margin)
            panel[35:315, 10:PANEL_W - 10] = fit(up.zoom(follow, 280), PANEL_W - 20, 280)

        cv2.putText(panel, "Plate zoom", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        panel[350:470, 10:PANEL_W - 10] = fit(last_plate_img, PANEL_W - 20, 120)
        cv2.putText(panel, pretty(last_plate_text), (10, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.putText(panel, "Log", (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        for i, (p, ts) in enumerate(db.recent(7)):
            cv2.putText(panel, f"{ts[11:]}  {pretty(p)}", (10, 565 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

        cv2.imshow("Plate Zoom", np.hstack([vis, panel]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            state["follow_zoom"] = min(6.0, state["follow_zoom"] * 1.25)
        if key == ord("-"):
            state["follow_zoom"] = max(1.0, state["follow_zoom"] / 1.25)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
