# yolo_webcam_server.py
# webcam -> YOLO -> annotated frame + counts -> WebSocket -> browser TTS
# pip install ultralytics websockets opencv-python

import asyncio, json, os, time, cv2, websockets
from collections import Counter
from ultralytics import YOLO

CAM_INDEX = 0               # change if you have multiple webcams
IMG_SIZE  = 800
DEFAULT_CONF = 0.25
IOU = 0.60
MAX_DET = 300
FPS_LIMIT = 5               # inference fps cap (to save CPU)
OUT_DIR = "out_frames"      # served by http.server
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "latest.jpg")

# Cooldown between announcements of the same class (seconds)
ANNOUNCE_COOLDOWN = 5.0

# Load model once
model = YOLO("best.pt")
CLASS_NAMES = model.names  # {id: name}

def pluralize(label, n):
    # naive pluralizer for labels like sleeping / eating / misbehaving
    return f"{n} {label}" if n == 1 else f"{n} {label}s" if not label.endswith("s") else f"{n} {label}"

def draw_boxes(img, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
        tag = f'{d["class"]} {d["conf"]:.2f}'
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0,255,255), -1)
        cv2.putText(img, tag, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

async def stream_webcam(ws):
    print("📷 Webcam client connected")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        await ws.send(json.dumps({"type":"error","msg":"Cannot open webcam"}))
        return

    try:
        last_announce = {}   # per-class last spoken time
        last_sent_ts  = 0.0
        min_interval  = 1.0 / max(1, FPS_LIMIT)

        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                await ws.send(json.dumps({"type":"error","msg":"Webcam frame read failed"}))
                break

            results = model(frame, imgsz=IMG_SIZE, conf=DEFAULT_CONF, iou=IOU, max_det=MAX_DET)

            # collect detections
            dets = []
            for r in results:
                if not r.boxes: continue
                for b in r.boxes:
                    cid  = int(b.cls)
                    conf = float(b.conf)
                    cname = CLASS_NAMES.get(cid, f"class_{cid}")
                    dets.append({
                        "class": cname.replace("_"," "),
                        "conf": round(conf, 3),
                        "xyxy": b.xyxy[0].cpu().numpy().tolist()
                    })

            # counts + spoken sentence with cooldown
            cnt = Counter(d["class"] for d in dets)
            counts_list = [{"class": k, "count": v} for k, v in cnt.items()]
            if counts_list:
                # announce classes that are not on cooldown
                now = time.time()
                speak_parts = []
                for k, v in cnt.items():
                    if now - last_announce.get(k, 0) >= ANNOUNCE_COOLDOWN:
                        speak_parts.append(pluralize(k, v))
                        last_announce[k] = now
                spoken = f"For webcam: {', '.join(speak_parts)}" if speak_parts else ""
            else:
                spoken = ""  # no announcement if nothing detected

            # draw and save latest annotated frame
            annotated = frame.copy()
            draw_boxes(annotated, dets)
            cv2.imwrite(OUT_PATH, annotated)

            # throttle sends to ~FPS_LIMIT
            if time.time() - last_sent_ts >= min_interval:
                await ws.send(json.dumps({
                    "type": "frame",
                    "annot": f"{OUT_DIR}/latest.jpg",  # relative path for browser
                    "counts": counts_list,
                    "speak": spoken
                }))
                last_sent_ts = time.time()

            # maintain FPS limit (best effort)
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, min_interval - elapsed))
    except websockets.ConnectionClosed:
        pass
    finally:
        cap.release()
        print("🛑 Webcam stream ended")

async def main():
    print("✅ Webcam server at ws://127.0.0.1:8000 — open tts_cam.html")
    async with websockets.serve(stream_webcam, "127.0.0.1", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
