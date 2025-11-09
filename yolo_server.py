# yolo_server.py
import asyncio, json, glob, os, cv2, websockets
from ultralytics import YOLO

IMAGES_DIR = "demo_images"
OUT_DIR = "out_images"  # will be served by http.server
os.makedirs(OUT_DIR, exist_ok=True)

# Load model once
model = YOLO("best.pt")
CLASS_NAMES = model.names
DEFAULT_CONF = 0.25
PER_CLASS_CONF = {
    "sleeping": 0.25,
    "phone_usage": 0.25,
    "no_id_card": 0.30,
    "misbehaving": 0.25,
}

def draw_boxes(img, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
        label = f'{d["class"]} {d["conf"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0,255,255), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

async def handle_client(ws):  # one-arg version you’re using
    print("👤 Client connected")
    try:
        files = []
        for pat in ("*.jpg","*.jpeg","*.png"):
            files += glob.glob(os.path.join(IMAGES_DIR, pat))
        files.sort()

        if not files:
            await ws.send(json.dumps({"type":"info","msg":"No images found in demo_images/"}))
            return

        for idx, p in enumerate(files, start=1):   # << add index here
            img = cv2.imread(p)
            if img is None:
                await ws.send(json.dumps({"type":"info","msg":f"Skip unreadable: {os.path.basename(p)}"}))
                continue

            results = model(img, imgsz=800, conf=DEFAULT_CONF, iou=0.6, max_det=300)

            dets = []
            for r in results:
                if not r.boxes: 
                    continue
                for b in r.boxes:
                    cid = int(b.cls); conf = float(b.conf)
                    cname = CLASS_NAMES.get(cid, f"class_{cid}")
                    thr = PER_CLASS_CONF.get(cname, DEFAULT_CONF)
                    if conf >= thr:
                        dets.append({
                            "class": cname.replace("_"," "),
                            "conf": round(conf,3),
                            "xyxy": b.xyxy[0].cpu().numpy().tolist()
                        })

            annotated = img.copy()
            draw_boxes(annotated, dets)
            base = os.path.basename(p)
            out_path = os.path.join(OUT_DIR, f"annot_{base}")
            cv2.imwrite(out_path, annotated)

            await ws.send(json.dumps({
                "type": "detections",
                "index": idx,                      # << send index
                "image": base,
                "annot": f"{OUT_DIR}/annot_{base}",
                "events": ([{"say": f'{d["class"]} detected', "conf": d["conf"]} for d in dets]
                           or [{"say":"no anomaly detected","conf":1.0}])
            }))

        await ws.send(json.dumps({"type":"done"}))
        print("✅ Finished sending all images")
    except Exception as e:
        print("❌ Server error:", e)

async def main():
    print("✅ Image server at ws://127.0.0.1:8000 — put images in demo_images/")
    async with websockets.serve(handle_client, "127.0.0.1", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
