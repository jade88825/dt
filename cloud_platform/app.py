"""
智慧交通云平台 - YOLO检测结果展示 (升级版)
基于Flask的Web应用，使用最优2类模型，提供检测结果可视化、训练指标分析、
实时检测、交通分析、跟踪回放等功能

运行:
    python cloud_platform/app.py
    然后浏览器访问 http://127.0.0.1:5000
"""

import csv
import time
import base64
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, Response, url_for
)
from ultralytics import YOLO

# ============================================================
# 配置 - 路径自适应 (本地Windows / Render Linux)
# ============================================================
import platform
import os

IS_RENDER = os.environ.get("RENDER", False)

if IS_RENDER:
    # Render Linux 部署
    BASE_DIR = Path(__file__).resolve().parent
    RUNS_DIR = BASE_DIR / "runs"
    DETECT_DIR = BASE_DIR / "runs" / "detect"
    KITTI_IMAGES_DIR = BASE_DIR / "kitti" / "images" / "val"
else:
    # 本地 Windows
    BASE_DIR = Path(__file__).resolve().parent.parent
    RUNS_DIR = BASE_DIR / "runs"
    DETECT_DIR = RUNS_DIR / "detect"
    KITTI_IMAGES_DIR = BASE_DIR / "kitti" / "images" / "val"

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# 最优模型
BEST_RUN = "baseline_2cls-5"
BEST_WEIGHTS = DETECT_DIR / BEST_RUN / "weights" / "best.pt"

# 跟踪视频
TRACK_VIDEO = RUNS_DIR / "track" / "track" / "val" / "vl_detected.mp4"
ORIGINAL_VIDEO = RUNS_DIR / "track" / "track" / "val" / "vl_original.mp4"

# 2类模型类别定义
CLASS_NAMES = {0: "车辆", 1: "行人"}
CLASS_NAMES_EN = {0: "vehicle", 1: "pedestrian"}

# 类别分组
VEHICLE_CLASSES = {0}
PEDESTRIAN_CLASSES = {1}

# 拥堵等级阈值
CONGESTION_LEVELS = [
    (0, 5, "畅通", "green"),
    (5, 10, "轻度拥堵", "yellow"),
    (10, 20, "中度拥堵", "orange"),
    (20, 999, "严重拥堵", "red"),
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["JSON_AS_ASCII"] = False

# ============================================================
# YOLO模型懒加载
# ============================================================
_yolo_model = None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        print(f"加载YOLO模型: {BEST_WEIGHTS}")
        _yolo_model = YOLO(str(BEST_WEIGHTS))
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _yolo_model.predict(source=dummy, verbose=False)
        print("模型加载并预热完成")
    return _yolo_model


# ============================================================
# 辅助函数
# ============================================================
def load_results_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                key = k.strip() if k else k
                cleaned[key] = v
            rows.append(cleaned)
    return rows


def get_final_metrics(rows):
    if not rows:
        return {}
    last = rows[-1]
    return {
        "epoch": int(last.get("epoch", 0)),
        "mAP50": float(last.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(last.get("metrics/mAP50-95(B)", 0)),
        "precision": float(last.get("metrics/precision(B)", 0)),
        "recall": float(last.get("metrics/recall(B)", 0)),
    }


def get_all_experiments():
    """收集三组对比实验结果"""
    experiments = {
        "8类Baseline(50ep)": DETECT_DIR / "train-4" / "results.csv",
        "2类Baseline(100ep)": DETECT_DIR / "baseline_2cls-5" / "results.csv",
        "2类+CBAM(100ep)": DETECT_DIR / "cbam_2cls-2" / "results.csv",
    }
    results = {}
    for name, csv_path in experiments.items():
        rows = load_results_csv(csv_path)
        if rows:
            results[name] = rows
    return results


def get_congestion_level(vehicle_count):
    for low, high, label, color in CONGESTION_LEVELS:
        if low <= vehicle_count < high:
            return {"label": label, "color": color}
    return {"label": "畅通", "color": "green"}


def run_detection(image_data, conf=0.25, iou=0.45):
    model = get_yolo_model()
    t0 = time.time()
    results = model.predict(
        source=image_data, conf=conf, iou=iou,
        imgsz=640, verbose=False,
    )
    inference_time = (time.time() - t0) * 1000

    result = results[0]
    annotated = result.plot()

    detections = []
    class_counts = {0: 0, 1: 0}
    vehicle_count = 0
    pedestrian_count = 0

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_data = result.boxes.data.cpu().numpy()
        for box in boxes_data:
            x1, y1, x2, y2, cf, cls = box
            cls_int = int(cls)
            detections.append({
                "class": CLASS_NAMES.get(cls_int, "unknown"),
                "class_en": CLASS_NAMES_EN.get(cls_int, "unknown"),
                "confidence": round(float(cf), 3),
                "bbox": [round(float(x1)), round(float(y1)),
                         round(float(x2)), round(float(y2))],
            })
            if cls_int in class_counts:
                class_counts[cls_int] += 1
            if cls_int in VEHICLE_CLASSES:
                vehicle_count += 1
            elif cls_int in PEDESTRIAN_CLASSES:
                pedestrian_count += 1

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "image": f"data:image/jpeg;base64,{result_b64}",
        "detections": detections,
        "total": len(detections),
        "class_counts": {CLASS_NAMES[k]: v for k, v in class_counts.items() if v > 0},
        "category_counts": {"车辆": vehicle_count, "行人": pedestrian_count},
        "inference_time": round(inference_time, 1),
        "vehicle_count": vehicle_count,
        "pedestrian_count": pedestrian_count,
    }


# ============================================================
# 实时摄像头视频流
# ============================================================
CAMERA_ACTIVE = False


def generate_camera_frames():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = True
    model = get_yolo_model()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while CAMERA_ACTIVE:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame, conf=0.25, iou=0.45,
                imgsz=640, device=model.device, verbose=False
            )

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        color = (0, 140, 255) if cls == 0 else (0, 255, 140)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{CLASS_NAMES_EN.get(cls, 'obj')} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            box_count = len(results[0].boxes) if len(results) > 0 else 0
            cv2.putText(frame, f"Objects: {box_count}  Model: YOLOv8s 2-class",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        CAMERA_ACTIVE = False


# ============================================================
# 路由: 页面
# ============================================================
@app.route("/")
def dashboard():
    csv_path = DETECT_DIR / BEST_RUN / "results.csv"
    rows = load_results_csv(csv_path)
    metrics = get_final_metrics(rows)
    all_exp = get_all_experiments()

    exp_chart = []
    for name, exp_rows in all_exp.items():
        m = get_final_metrics(exp_rows)
        exp_chart.append({
            "name": name,
            "mAP50": round(m.get("mAP50", 0) * 100, 1),
            "mAP50_95": round(m.get("mAP50_95", 0) * 100, 1),
            "precision": round(m.get("precision", 0) * 100, 1),
            "recall": round(m.get("recall", 0) * 100, 1),
        })

    track_exists = TRACK_VIDEO.exists()

    stats = {
        "model": "YOLOv8s (2类: 车辆+行人)",
        "dataset": "KITTI",
        "classes": 2,
        "class_names": list(CLASS_NAMES.values()),
        "epochs": metrics.get("epoch", 0),
        "mAP50": metrics.get("mAP50", 0),
        "mAP50_95": metrics.get("mAP50_95", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "experiments": len(all_exp),
        "train_images": 5985,
        "val_images": 1496,
        "exp_chart": exp_chart,
        "track_video_exists": track_exists,
        "track_stats": {
            "frames": 1496,
            "detections": 7805,
            "track_ids": 311,
            "trajectories": 780,
            "fps": 16.2,
            "reid_fps": 74.5,
        },
    }
    return render_template("dashboard.html", stats=stats)


@app.route("/detection")
def detection():
    run_dir = DETECT_DIR / BEST_RUN
    pred_images = []
    label_images = []
    train_images = []

    if run_dir.exists():
        for f in sorted(run_dir.glob("val_batch*_pred.jpg")):
            pred_images.append(f.name)
        for f in sorted(run_dir.glob("val_batch*_labels.jpg")):
            label_images.append(f.name)
        for f in sorted(run_dir.glob("train_batch*.jpg")):
            train_images.append(f.name)

    return render_template("detection.html",
                           pred_images=pred_images,
                           label_images=label_images,
                           train_images=train_images,
                           run_name=BEST_RUN)


@app.route("/metrics")
def metrics():
    csv_path = DETECT_DIR / BEST_RUN / "results.csv"
    rows = load_results_csv(csv_path)

    chart_data = {
        "epochs": [], "mAP50": [], "mAP50_95": [],
        "precision": [], "recall": [],
        "box_loss": [], "cls_loss": [],
    }
    for row in rows:
        chart_data["epochs"].append(int(row.get("epoch", 0)))
        chart_data["mAP50"].append(float(row.get("metrics/mAP50(B)", 0)))
        chart_data["mAP50_95"].append(float(row.get("metrics/mAP50-95(B)", 0)))
        chart_data["precision"].append(float(row.get("metrics/precision(B)", 0)))
        chart_data["recall"].append(float(row.get("metrics/recall(B)", 0)))
        chart_data["box_loss"].append(float(row.get("train/box_loss", 0)))
        chart_data["cls_loss"].append(float(row.get("train/cls_loss", 0)))

    all_exp = get_all_experiments()
    comparison = []
    for name, exp_rows in all_exp.items():
        m = get_final_metrics(exp_rows)
        comparison.append({
            "name": name,
            "epoch": m.get("epoch", 0),
            "mAP50": m.get("mAP50", 0),
            "mAP50_95": m.get("mAP50_95", 0),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
        })

    run_dir = DETECT_DIR / BEST_RUN
    curve_images = []
    if run_dir.exists():
        for f in sorted(run_dir.glob("*.png")):
            curve_images.append(f.name)

    return render_template("metrics.html",
                           chart_data=chart_data,
                           comparison=comparison,
                           curve_images=curve_images,
                           run_name=BEST_RUN,
                           final_metrics=get_final_metrics(rows))


@app.route("/detect")
def detect():
    sample_images = []
    if KITTI_IMAGES_DIR.exists():
        for f in sorted(KITTI_IMAGES_DIR.glob("*.png"))[:12]:
            sample_images.append(f.name)
    return render_template("detect.html", sample_images=sample_images)


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/tracking")
def tracking():
    track_exists = TRACK_VIDEO.exists()
    original_exists = ORIGINAL_VIDEO.exists()
    return render_template("tracking.html", 
                           track_exists=track_exists,
                           original_exists=original_exists)


# ============================================================
# 路由: API
# ============================================================
@app.route("/api/detect", methods=["POST"])
def api_detect():
    if request.content_type and "multipart/form-data" in request.content_type:
        conf = float(request.form.get("conf", 0.25))
        iou = float(request.form.get("iou", 0.45))
    else:
        conf = float(request.json.get("conf", 0.25)) if request.json else 0.25
        iou = float(request.json.get("iou", 0.45)) if request.json else 0.45

    image_data = None
    source_name = "upload"

    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image_data = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            source_name = file.filename
    elif request.json and "image" in request.json:
        b64 = request.json["image"].split(",")[-1]
        file_bytes = np.frombuffer(base64.b64decode(b64), np.uint8)
        image_data = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif request.json and "sample" in request.json:
        sample_path = KITTI_IMAGES_DIR / request.json["sample"]
        if sample_path.exists():
            image_data = cv2.imread(str(sample_path))
            source_name = request.json["sample"]

    if image_data is None:
        return jsonify({"error": "未收到有效图片"}), 400

    try:
        result = run_detection(image_data, conf=conf, iou=iou)
        result["source"] = source_name
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"检测失败: {str(e)}"}), 500


@app.route("/api/batch_detect", methods=["POST"])
def api_batch_detect():
    data = request.json or {}
    count = min(int(data.get("count", 10)), 30)
    conf = float(data.get("conf", 0.25))
    iou = float(data.get("iou", 0.45))

    all_images = sorted(KITTI_IMAGES_DIR.glob("*.png"))
    if not all_images:
        return jsonify({"error": "未找到KITTI验证集图片"}), 404

    import random
    random.seed(42)
    selected = random.sample(all_images, min(count, len(all_images)))

    results_list = []
    total_vehicles = 0
    total_pedestrians = 0
    congestion_counts = {"畅通": 0, "轻度拥堵": 0, "中度拥堵": 0, "严重拥堵": 0}

    for img_path in selected:
        image_data = cv2.imread(str(img_path))
        if image_data is None:
            continue
        try:
            result = run_detection(image_data, conf=conf, iou=iou)
            cong = get_congestion_level(result["vehicle_count"])
            congestion_counts[cong["label"]] += 1
            total_vehicles += result["vehicle_count"]
            total_pedestrians += result["pedestrian_count"]
            results_list.append({
                "image": img_path.name,
                "vehicles": result["vehicle_count"],
                "pedestrians": result["pedestrian_count"],
                "total": result["total"],
                "congestion": cong["label"],
                "congestion_color": cong["color"],
                "inference_time": result["inference_time"],
            })
        except Exception:
            continue

    avg_vehicles = total_vehicles / len(results_list) if results_list else 0
    avg_pedestrians = total_pedestrians / len(results_list) if results_list else 0

    return jsonify({
        "results": results_list,
        "summary": {
            "total_images": len(results_list),
            "total_vehicles": total_vehicles,
            "total_pedestrians": total_pedestrians,
            "avg_vehicles": round(avg_vehicles, 1),
            "avg_pedestrians": round(avg_pedestrians, 1),
            "congestion_distribution": congestion_counts,
        },
    })


@app.route("/api/metrics")
def api_metrics():
    csv_path = DETECT_DIR / BEST_RUN / "results.csv"
    rows = load_results_csv(csv_path)
    return jsonify({"data": rows, "final": get_final_metrics(rows)})


@app.route("/api/camera/start")
def api_camera_start():
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/camera/stop")
def api_camera_stop():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = False
    return jsonify({"status": "stopped"})


# ============================================================
# 路由: 静态资源
# ============================================================
@app.route("/runs_img/<path:filename>")
def runs_image(filename):
    return send_from_directory(str(DETECT_DIR / BEST_RUN), filename)


@app.route("/kitti_img/<path:filename>")
def kitti_image(filename):
    return send_from_directory(str(KITTI_IMAGES_DIR), filename)


@app.route("/upload_img/<path:filename>")
def upload_image(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/track_video")
def track_video():
    if TRACK_VIDEO.exists():
        return video_stream(TRACK_VIDEO)
    return "Video not found", 404


@app.route("/original_video")
def original_video():
    if ORIGINAL_VIDEO.exists():
        return video_stream(ORIGINAL_VIDEO)
    return "Video not found", 404


def video_stream(video_path: Path):
    """支持Range请求的视频流式传输, 浏览器兼容"""
    from flask import Response, request
    
    path = Path(video_path)
    file_size = path.stat().st_size
    range_header = request.headers.get("Range", None)
    
    if range_header:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = min(int(byte_range[1]), file_size - 1) if len(byte_range) > 1 and byte_range[1] else file_size - 1
        length = end - start + 1
        
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)
        
        return Response(data, status=206, mimetype="video/mp4",
                        headers={
                            "Content-Range": f"bytes {start}-{end}/{file_size}",
                            "Accept-Ranges": "bytes",
                            "Content-Length": str(length),
                        })
    else:
        with open(path, "rb") as f:
            data = f.read()
        return Response(data, mimetype="video/mp4",
                        headers={
                            "Content-Length": str(file_size),
                            "Accept-Ranges": "bytes",
                        })


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  智慧交通云平台")
    print(f"  访问: http://0.0.0.0:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
