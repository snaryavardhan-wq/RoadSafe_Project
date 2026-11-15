"""
RoadSafe v3 - AI-Based Driver Behaviour Analysis and Road Safety Alert System
Flask + YOLOv8 + ByteTrack + OpenCV + SQLite

New in v3
  * danger zone: draw the junction area on the first frame; only events inside count
  * vehicle counting per video (unique track IDs, approximate)
  * junction risk report: counts, rates, peak interval, risk map, printable
All measurements are image-space estimates, not calibrated real-world values.
"""
import csv
import io
import json
import math
import os
import sqlite3
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, send_from_directory, url_for)
from ultralytics import YOLO

# ------------------------------------------------------------------ config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DB_PATH = os.path.join(BASE_DIR, "roadsafe.db")
ALLOWED = {".mp4", ".avi", ".mov", ".mkv"}

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF = 0.30
PROCESS_WIDTH = 960

HISTORY_LEN = 30
MIN_HIST = 6
SPEED_WINDOW = 10
MOVING_PCT = 1.5

LATERAL_WINDOW = 5
LATERAL_RATIO = 0.055
LATERAL_COOLDOWN = 45

DUP_IOU = 0.70
CONTACT_K = 0.35
MIN_CLOSING_PCT = 2.0
TTC_HIGH = 1.5
TTC_MEDIUM = 3.0
PAIR_COOLDOWN = 60
BOARD_SECONDS = 10

# Risk index = (3*High + Medium) per 100 vehicles. Thresholds are indicative only.
RISK_MODERATE = 10
RISK_HIGH = 30

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "roadsafe-dev-key"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

_model = None
PROGRESS = {}


def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


# ---------------------------------------------------------------- database
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                track_id INTEGER,
                vehicle_type TEXT,
                event_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                details TEXT
            )""")
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(alerts)")}
        for col in ("ttc", "video_time", "x", "y"):      # migrations
            if col not in cols:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {col} REAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_stats (
                video_name TEXT PRIMARY KEY,
                duration REAL,
                vehicles INTEGER,
                by_type TEXT,
                zone_used INTEGER,
                analyzed_at TEXT
            )""")


def save_alert(video, frame_no, video_time, track_id, vtype, event, risk,
               details, ttc=None, x=None, y=None):
    with db() as conn:
        conn.execute(
            "INSERT INTO alerts (video_name, timestamp, frame_number, track_id,"
            " vehicle_type, event_type, risk_level, details, ttc, video_time, x, y)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (video, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_no,
             track_id, vtype, event, risk, details, ttc, video_time, x, y))


# ------------------------------------------------------------ geometry helpers
def centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def diag(box):
    return math.hypot(box[2] - box[0], box[3] - box[1])


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def velocity(hist, fps):
    if len(hist) < MIN_HIST:
        return None
    k = min(SPEED_WINDOW, len(hist) - 1)
    dt = k / fps
    return ((hist[-1][0] - hist[-1 - k][0]) / dt,
            (hist[-1][1] - hist[-1 - k][1]) / dt)


def time_to_contact(ca, va, cb, vb, contact):
    """Constant-velocity model: smallest t>=0 with |r + v t| = contact."""
    rx, ry = cb[0] - ca[0], cb[1] - ca[1]
    vx, vy = vb[0] - va[0], vb[1] - va[1]
    d = math.hypot(rx, ry)
    if d < 1e-6:
        return None
    closing = -(rx * vx + ry * vy) / d
    if closing <= 0:
        return None
    c = d * d - contact * contact
    if c <= 0:
        return 0.0, closing
    a = vx * vx + vy * vy
    b = 2 * (rx * vx + ry * vy)
    disc = b * b - 4 * a * c
    if a < 1e-9 or disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2 * a)
    return (t, closing) if t >= 0 else None


def draw_box(frame, box, label, colour):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


RED, ORANGE, GREEN, GREY, YELLOW = ((0, 0, 255), (0, 165, 255), (0, 200, 0),
                                    (150, 150, 150), (0, 196, 255))


# ------------------------------------------------------------------- zones
def zone_path(name):
    return os.path.join(UPLOAD_DIR, name + ".zone.json")


def load_zone_points(name):
    """Normalised polygon [[x,y],...] or None."""
    try:
        with open(zone_path(name), encoding="utf-8") as f:
            pts = json.load(f).get("points")
        return pts if pts and len(pts) >= 3 else None
    except (OSError, ValueError):
        return None


def save_first_frame(name):
    cap = cv2.VideoCapture(os.path.join(UPLOAD_DIR, name))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(10, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    h, w = frame.shape[:2]
    if w > PROCESS_WIDTH:
        frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * PROCESS_WIDTH / w)))
    cv2.imwrite(os.path.join(UPLOAD_DIR, name + ".jpg"), frame)
    return True


# ------------------------------------------------------------ video analysis
def analyze_video(name):
    path = os.path.join(UPLOAD_DIR, name)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        PROGRESS[name] = {"frame": 0, "total": 0, "done": True}
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, PROCESS_WIDTH / src_w)
    W, H = int(src_w * scale), int(src_h * scale)

    # danger zone (normalised -> pixels of the processed frame)
    zpts = load_zone_points(name)
    zone = None
    if zpts:
        zone = np.array([[int(x * W), int(y * H)] for x, y in zpts], dtype=np.int32)

    def in_zone(px, py):
        return zone is None or cv2.pointPolygonTest(zone, (float(px), float(py)), False) >= 0

    # a re-analysis replaces earlier results for this video
    with db() as conn:
        conn.execute("DELETE FROM alerts WHERE video_name=?", (name,))
        conn.execute("DELETE FROM video_stats WHERE video_name=?", (name,))

    out_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + "_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    model = get_model()
    history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    last_lateral, last_pair = {}, {}
    seen = defaultdict(set)          # vehicle type -> track IDs seen in zone
    frame_no = 0
    PROGRESS[name] = {"frame": 0, "total": total, "done": False}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            vtime = frame_no / fps
            if scale < 1.0:
                frame = cv2.resize(frame, (W, H))

            results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  classes=list(VEHICLE_CLASSES), conf=CONF,
                                  verbose=False)
            vehicles = []
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                for b, tid, cls in zip(boxes.xyxy.tolist(),
                                       boxes.id.int().tolist(),
                                       boxes.cls.int().tolist()):
                    vehicles.append((tid, VEHICLE_CLASSES.get(cls, "vehicle"), b))

            vel, speed, moving = {}, {}, {}
            for tid, vtype, box in vehicles:
                c = centre(box)
                history[tid].append(c)
                if in_zone(*c):
                    seen[vtype].add(tid)
                v = velocity(history[tid], fps)
                vel[tid] = v
                speed[tid] = (math.hypot(*v) / W * 100) if v else None
                moving[tid] = speed[tid] is not None and speed[tid] >= MOVING_PCT

            flagged = {}

            def mark(tid, colour):
                if flagged.get(tid) != RED:
                    flagged[tid] = colour

            # A. sudden lateral movement (moving vehicles inside the zone)
            for tid, vtype, box in vehicles:
                h = history[tid]
                if not moving[tid] or len(h) <= LATERAL_WINDOW or not in_zone(*h[-1]):
                    continue
                dx = h[-1][0] - h[-1 - LATERAL_WINDOW][0]
                if abs(dx) >= LATERAL_RATIO * W:
                    mark(tid, ORANGE)
                    if frame_no - last_lateral.get(tid, -10**9) >= LATERAL_COOLDOWN:
                        last_lateral[tid] = frame_no
                        side = "right" if dx > 0 else "left"
                        save_alert(name, frame_no, vtime, tid, vtype,
                                   "Sudden lateral movement", "Medium",
                                   f"Moved {abs(dx) / W * 100:.1f}% of frame width "
                                   f"{side} in {LATERAL_WINDOW} frames "
                                   f"(speed {speed[tid]:.0f}% width/s)",
                                   x=h[-1][0] / W, y=h[-1][1] / H)

            # B. predicted time-to-contact (pair midpoint inside the zone)
            lines = []
            for i in range(len(vehicles)):
                for j in range(i + 1, len(vehicles)):
                    ida, ta, ba = vehicles[i]
                    idb, tb, bb = vehicles[j]
                    if vel[ida] is None or vel[idb] is None:
                        continue
                    if not (moving[ida] or moving[idb]):
                        continue
                    if ta == tb and iou(ba, bb) >= DUP_IOU:
                        continue
                    ca, cb = centre(ba), centre(bb)
                    mid = ((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2)
                    if not in_zone(*mid):
                        continue
                    contact = CONTACT_K * (diag(ba) + diag(bb))
                    res = time_to_contact(ca, vel[ida], cb, vel[idb], contact)
                    if res is None:
                        continue
                    ttc, closing = res
                    if closing < MIN_CLOSING_PCT * W / 100:
                        continue
                    if ttc <= TTC_HIGH:
                        risk, colour = "High", RED
                    elif ttc <= TTC_MEDIUM:
                        risk, colour = "Medium", ORANGE
                    else:
                        continue
                    mark(ida, colour)
                    mark(idb, colour)
                    lines.append((ca, cb, ttc, colour))
                    key = (min(ida, idb), max(ida, idb))
                    if frame_no - last_pair.get(key, -10**9) >= PAIR_COOLDOWN:
                        last_pair[key] = frame_no
                        save_alert(name, frame_no, vtime, ida, ta,
                                   "Collision risk (TTC)", risk,
                                   f"{ta} #{ida} and {tb} #{idb}: predicted contact "
                                   f"in {ttc:.1f}s, closing at "
                                   f"{closing / W * 100:.0f}% width/s",
                                   ttc=round(ttc, 2), x=mid[0] / W, y=mid[1] / H)

            # draw zone, vehicles, TTC lines, status text
            if zone is not None:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [zone], YELLOW)
                frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
                cv2.polylines(frame, [zone], True, YELLOW, 2)
            for tid, vtype, box in vehicles:
                colour = flagged.get(tid, GREEN if moving[tid] else GREY)
                tag = f"{vtype} #{tid}"
                if speed[tid] is not None:
                    tag += f" {speed[tid]:.0f}"
                draw_box(frame, box, tag, colour)
            for (pa, pb, ttc, colour) in lines:
                p1, p2 = (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1]))
                cv2.line(frame, p1, p2, colour, 2)
                cv2.putText(frame, f"TTC {ttc:.1f}s",
                            ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
            if any(c == RED for c in flagged.values()):
                cv2.putText(frame, "DANGER", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, RED, 3, cv2.LINE_AA)
            elif flagged:
                cv2.putText(frame, "CAUTION", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, ORANGE, 3, cv2.LINE_AA)

            writer.write(frame)
            PROGRESS[name]["frame"] = frame_no
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpg.tobytes() + b"\r\n")
    finally:
        cap.release()
        writer.release()
        if frame_no > 0:
            counts = {t: len(s) for t, s in seen.items()}
            with db() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO video_stats VALUES (?,?,?,?,?,?)",
                    (name, frame_no / fps, sum(counts.values()), json.dumps(counts),
                     1 if zone is not None else 0,
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        PROGRESS[name]["done"] = True


# ------------------------------------------------------- warning board logic
def board_state(video=None):
    cutoff = (datetime.now() - timedelta(seconds=BOARD_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")
    q, args = "SELECT * FROM alerts WHERE timestamp >= ?", [cutoff]
    if video:
        q += " AND video_name = ?"
        args.append(video)
    q += " ORDER BY id DESC"
    with db() as conn:
        rows = conn.execute(q, args).fetchall()

    def message(r):
        if r["event_type"] == "Sudden lateral movement":
            return "Vehicle swerving ahead"
        return "Vehicles on a collision course ahead"

    high = next((r for r in rows if r["risk_level"] == "High"), None)
    med = next((r for r in rows if r["risk_level"] == "Medium"), None)
    if high:
        return {"level": "danger", "title": "DANGER: SLOW DOWN", "message": message(high)}
    if med:
        return {"level": "caution", "title": "CAUTION", "message": message(med)}
    return {"level": "clear", "title": "ROAD CLEAR", "message": "No risky movement detected"}


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    with db() as conn:
        videos = conn.execute(
            "SELECT s.video_name, s.analyzed_at, s.vehicles, "
            "(SELECT COUNT(*) FROM alerts a WHERE a.video_name=s.video_name) AS n "
            "FROM video_stats s ORDER BY s.analyzed_at DESC LIMIT 8").fetchall()
        stats = conn.execute(
            "SELECT COUNT(*) AS total, SUM(risk_level='High') AS high, "
            "SUM(risk_level='Medium') AS medium FROM alerts").fetchone()
    return render_template("index.html", videos=videos, stats=stats)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        flash("Choose a video file first.")
        return redirect(url_for("index"))
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        flash("Unsupported file type. Use .mp4, .avi, .mov or .mkv.")
        return redirect(url_for("index"))
    name = f"{uuid.uuid4().hex[:8]}{ext}"
    f.save(os.path.join(UPLOAD_DIR, name))
    if not save_first_frame(name):
        flash("This video could not be read. Try another file.")
        return redirect(url_for("index"))
    return redirect(url_for("zone", name=name))


@app.route("/frame/<name>")
def frame_image(name):
    return send_from_directory(UPLOAD_DIR, os.path.basename(name) + ".jpg")


@app.route("/zone/<name>")
def zone(name):
    name = os.path.basename(name)
    if not os.path.exists(os.path.join(UPLOAD_DIR, name + ".jpg")):
        flash("That video was not found. Upload it again.")
        return redirect(url_for("index"))
    return render_template("zone.html", name=name, existing=load_zone_points(name))


@app.route("/api/zone/<name>", methods=["POST"])
def api_zone(name):
    name = os.path.basename(name)
    pts = (request.get_json(silent=True) or {}).get("points")
    ok = (isinstance(pts, list) and 3 <= len(pts) <= 50 and
          all(isinstance(p, list) and len(p) == 2 and
              all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in p) for p in pts))
    if not ok:
        return jsonify(error="Zone needs 3 or more points inside the frame."), 400
    with open(zone_path(name), "w", encoding="utf-8") as f:
        json.dump({"points": pts}, f)
    return jsonify(ok=True)


@app.route("/api/zone/<name>", methods=["DELETE"])
def api_zone_clear(name):
    try:
        os.remove(zone_path(os.path.basename(name)))
    except OSError:
        pass
    return jsonify(ok=True)


@app.route("/analyze/<name>")
def analyze(name):
    name = os.path.basename(name)
    if not os.path.exists(os.path.join(UPLOAD_DIR, name)):
        flash("That video was not found. Upload it again.")
        return redirect(url_for("index"))
    return render_template("analyze.html", name=name)


@app.route("/stream/<name>")
def stream(name):
    return Response(analyze_video(os.path.basename(name)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/progress/<name>")
def api_progress(name):
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE video_name=? ORDER BY id DESC LIMIT 50",
            (name,)).fetchall()
    return jsonify(progress=PROGRESS.get(name, {"frame": 0, "total": 0, "done": False}),
                   alerts=[dict(r) for r in rows], board=board_state(name))


@app.route("/board")
def board():
    return render_template("board.html")


@app.route("/api/board")
def api_board():
    return jsonify(board_state())


@app.route("/alerts")
def alerts():
    risk = request.args.get("risk", "")
    query, args = "SELECT * FROM alerts", []
    if risk in ("High", "Medium"):
        query += " WHERE risk_level=?"
        args.append(risk)
    query += " ORDER BY id DESC LIMIT 500"
    with db() as conn:
        rows = conn.execute(query, args).fetchall()
    return render_template("alerts.html", rows=rows, risk=risk)


@app.route("/analytics")
def analytics():
    video = request.args.get("video", "")
    with db() as conn:
        videos = [r["video_name"] for r in conn.execute(
            "SELECT DISTINCT video_name FROM alerts ORDER BY video_name")]
        where, args = ("WHERE video_name=?", [video]) if video else ("", [])
        extra = "AND" if where else "WHERE"

        def group(col):
            return [(r[0] or "unknown", r[1]) for r in conn.execute(
                f"SELECT {col}, COUNT(*) FROM alerts {where} "
                f"GROUP BY {col} ORDER BY COUNT(*) DESC", args)]

        total = conn.execute(f"SELECT COUNT(*) FROM alerts {where}", args).fetchone()[0]
        by_risk, by_event, by_vehicle = group("risk_level"), group("event_type"), group("vehicle_type")
        times = [r[0] for r in conn.execute(
            f"SELECT video_time FROM alerts {where} {extra} video_time IS NOT NULL", args)]
        min_ttc = conn.execute(
            f"SELECT MIN(ttc) FROM alerts {where} {extra} ttc IS NOT NULL", args).fetchone()[0]

    counts = defaultdict(int)
    for t in times:
        counts[int(t // 5)] += 1
    timeline = [(f"{b * 5}-{(b + 1) * 5}s", counts.get(b, 0))
                for b in range(max(counts) + 1)] if counts else []
    return render_template("analytics.html", videos=videos, video=video, total=total,
                           by_risk=by_risk, by_event=by_event, by_vehicle=by_vehicle,
                           timeline=timeline, min_ttc=min_ttc)


@app.route("/report/<name>")
def report(name):
    name = os.path.basename(name)
    with db() as conn:
        st = conn.execute("SELECT * FROM video_stats WHERE video_name=?", (name,)).fetchone()
        rows = conn.execute(
            "SELECT * FROM alerts WHERE video_name=? ORDER BY video_time, id", (name,)).fetchall()
    if not st:
        flash("No report for that video yet. Analyze it first.")
        return redirect(url_for("index"))

    high = sum(1 for r in rows if r["risk_level"] == "High")
    med = sum(1 for r in rows if r["risk_level"] == "Medium")
    weighted = 3 * high + med
    duration = st["duration"] or 0
    vehicles = st["vehicles"] or 0
    index_val = (weighted / vehicles * 100) if vehicles else None
    per_hour = (weighted / (duration / 3600)) if duration else None
    if index_val is None:
        rating = "Not enough data"
    elif index_val >= RISK_HIGH:
        rating = "High"
    elif index_val >= RISK_MODERATE:
        rating = "Moderate"
    else:
        rating = "Low"

    buckets = defaultdict(int)
    for r in rows:
        if r["video_time"] is not None:
            buckets[int(r["video_time"] // 5)] += 1
    peak = None
    if buckets:
        b = max(buckets, key=buckets.get)
        peak = (f"{b * 5}-{(b + 1) * 5}s", buckets[b])

    by_event = defaultdict(int)
    for r in rows:
        by_event[r["event_type"]] += 1
    ttcs = [r["ttc"] for r in rows if r["ttc"] is not None]
    zone_pts = load_zone_points(name)
    by_type = json.loads(st["by_type"] or "{}")

    r = dict(name=name, duration=duration, vehicles=vehicles, by_type=by_type,
             zone_used=bool(st["zone_used"]), analyzed_at=st["analyzed_at"],
             total=len(rows), high=high, med=med, index=index_val, per_hour=per_hour,
             rating=rating, peak=peak, by_event=sorted(by_event.items(), key=lambda kv: -kv[1]),
             min_ttc=min(ttcs) if ttcs else None,
             dots=[dict(x=a["x"] * 100, y=a["y"] * 100, risk=a["risk_level"].lower())
                   for a in rows if a["x"] is not None and a["y"] is not None],
             zone_svg=" ".join(f"{x * 100:.1f},{y * 100:.1f}" for x, y in zone_pts) if zone_pts else "",
             has_frame=os.path.exists(os.path.join(UPLOAD_DIR, name + ".jpg")))
    return render_template("report.html", r=r, alerts=rows,
                           moderate=RISK_MODERATE, high_t=RISK_HIGH)


@app.route("/export.csv")
def export_csv():
    video = request.args.get("video", "")
    with db() as conn:
        if video:
            rows = conn.execute("SELECT * FROM alerts WHERE video_name=? ORDER BY id", (video,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM alerts ORDER BY id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(rows[0].keys() if rows else
               ["id", "video_name", "timestamp", "frame_number", "track_id", "vehicle_type",
                "event_type", "risk_level", "details", "ttc", "video_time", "x", "y"])
    for r in rows:
        w.writerow(list(r))
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=roadsafe_alerts.csv"})


@app.route("/clear", methods=["POST"])
def clear():
    with db() as conn:
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM video_stats")
    flash("Alert history cleared.")
    return redirect(url_for("alerts"))


@app.route("/download/<name>")
def download(name):
    out = os.path.splitext(os.path.basename(name))[0] + "_annotated.mp4"
    return send_from_directory(OUTPUT_DIR, out, as_attachment=True)


init_db()

if __name__ == "__main__":
    app.run(debug=False, threaded=True)
