import eventlet

eventlet.monkey_patch()

import os
import sys
import json
import asyncio
import threading
import time
from pathlib import Path
from datetime import datetime
import uuid

import numpy as np
import torch
from torchvision.ops import box_convert
from PIL import Image
import cv2
import glob

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    redirect,
    url_for,
    send_from_directory,
)
from flask_socketio import SocketIO, emit

from sam2.build_sam import build_sam2_video_predictor
from GroundingDINO.groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate,
)
from drawers import *
from pathcfg import *
from videowriter import *
from sam2mot.TMS import TMS
from sam2mot.CI import CI


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = ""
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


class SAM2MOTTracker:
    """Object-oriented SAM2 Multi-Object Tracker for Web Interface"""

    def __init__(
        self,
        detector_checkpoint=None,
        detector_cfg=None,
        sam2_checkpoint=None,
        sam2_model_cfg=None,
    ):
        # Model paths
        self.detector_checkpoint = (
            detector_checkpoint or "./GroundingDINO/weights/groundingdino_swint_ogc.pth"
        )
        self.detector_cfg = (
            detector_cfg
            or "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
        self.sam2_checkpoint = sam2_checkpoint or "./checkpoints/sam2.1_hiera_large.pt"
        self.sam2_model_cfg = sam2_model_cfg or "./configs/sam2.1/sam2.1_hiera_l.yaml"

        # Model instances
        self.detector = None
        self.predictor = None
        self.inference_state = None
        self.tms = None
        self.ci = None

        # Tracking parameters
        self.text_prompt = "person"
        self.box_threshold = 0.35
        self.text_threshold = 0.25

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (
            self.device.type == "cuda"
            and torch.cuda.get_device_properties(0).major >= 8
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Status tracking
        self.is_tracking = False
        self.current_session = None

        self.dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )

    @property
    def is_initialized(self):
        return self.detector is not None and self.predictor is not None

    def initialize_models(self, session_id):
        """Initialize all models"""
        try:
            socketio.emit(
                "progress_update",
                {"message": "Loading GroundingDINO...", "progress": 10},
                room=session_id,
            )

            self.detector = load_model(
                model_config_path=self.detector_cfg,
                model_checkpoint_path=self.detector_checkpoint,
                device=self.device,
            )

            socketio.emit(
                "progress_update",
                {"message": "Loading SAM2...", "progress": 50},
                room=session_id,
            )

            self.predictor = build_sam2_video_predictor(
                self.sam2_model_cfg,
                self.sam2_checkpoint,
                device=self.device,
                vos_optimized=False,
            )

            socketio.emit(
                "progress_update",
                {"message": "Initializing tracking modules...", "progress": 80},
                room=session_id,
            )

            self.tms = TMS()
            self.ci = CI()

            socketio.emit(
                "progress_update",
                {"message": "Models loaded successfully!", "progress": 100},
                room=session_id,
            )

            socketio.emit("models_initialized", {"success": True}, room=session_id)
            return True

        except Exception as e:
            error_msg = f"Error loading models: {str(e)}"
            socketio.emit(
                "progress_update",
                {"message": error_msg, "progress": 0},
                room=session_id,
            )
            socketio.emit(
                "models_initialized",
                {"success": False, "error": error_msg},
                room=session_id,
            )
            return False

    def set_tracking_parameters(
        self, text_prompt="person", box_threshold=0.35, text_threshold=0.25
    ):
        """Set tracking parameters"""
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def track_video(self, video_dir, output_path, output_frames_dir, session_id, fps):
        os.makedirs(output_frames_dir, exist_ok=True)
        """Main tracking function"""
        try:
            self.is_tracking = True
            self.current_session = session_id

            # Scan video frames
            frame_names = [
                p
                for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

            total_frames = len(frame_names)
            if not total_frames:
                raise ValueError("No valid image files found in the video directory")

            # [LOG ADDED]
            socketio.emit(
                "progress_update",
                {
                    "message": f"[PREP] Found {total_frames} frames for processing.",
                    "progress": 0,
                },
                room=session_id,
            )
            eventlet.sleep(0)

            start_time = time.time()
            results = []
            detector_results = {}

            # Phase 1: Detection
            socketio.emit(
                "progress_update",
                {
                    "message": "[DETECTION] Running detection on all frames...",
                    "progress": 3,
                },
                room=session_id,
            )
            eventlet.sleep(0)

            for frame_id, frame_name in enumerate(frame_names):
                if not self.is_tracking:
                    socketio.emit(
                        "progress_update",
                        {
                            "message": "[STOPPED] Tracking stopped by user during detection phase.",
                            "progress": 0,
                        },
                        room=session_id,
                    )
                    return False, "Tracking stopped by user"

                progress = 3 + ((frame_id + 1) / total_frames) * 47
                socketio.emit(
                    "progress_update",
                    {
                        "message": f"[DETECTION] Frame {frame_id+1}/{total_frames}",
                        "progress": progress,
                    },
                    room=session_id,
                )

                torch.cuda.empty_cache()
                image_path = os.path.join(video_dir, frame_name)
                image_source, image = load_image(image_path)

                with torch.no_grad():
                    boxes, logits, _ = predict(
                        model=self.detector,
                        image=image,
                        caption=self.text_prompt,
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                    )
                    boxes = box_convert(
                        boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
                    ).numpy()
                    detector_results[frame_id] = [boxes, logits]

                eventlet.sleep(0)

            # Phase 2: Tracking
            socketio.emit(
                "progress_update",
                {"message": "[TRACKING] Running tracking...", "progress": 50},
                room=session_id,
            )
            eventlet.sleep(0)

            # Initialize inference state
            socketio.emit(
                "progress_update",
                {"message": "[INIT] Initializing inference state...", "progress": 51},
                room=session_id,
            )
            eventlet.sleep(0)

            self.inference_state = self.predictor.init_state(
                video_path=video_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )

            socketio.emit(
                "progress_update",
                {"message": "[INIT] Inference state initialized.", "progress": 52},
                room=session_id,
            )
            eventlet.sleep(0)
            for frame_id, frame_name in enumerate(frame_names):
                if not self.is_tracking:
                    socketio.emit(
                        "progress_update",
                        {
                            "message": "[STOPPED] Tracking stopped by user during tracking phase.",
                            "progress": 0,
                        },
                        room=session_id,
                    )
                    self.predictor.reset_state(self.inference_state)
                    return False, "Tracking stopped by user"

                progress = 52 + ((frame_id + 1) / total_frames) * 38

                torch.cuda.empty_cache()
                det_bboxes = detector_results[frame_id][0]
                det_logits = detector_results[frame_id][1]

                # Propagate tracking for non-first frames
                if frame_id > 0:
                    has_objects = len(self.inference_state["obj_ids"]) > 0
                    if has_objects:
                        with torch.autocast("cuda", dtype=self.dtype):
                            for _, _, _ in self.predictor.propagate_in_video(
                                self.inference_state,
                                start_frame_idx=frame_id,
                                max_frame_num_to_track=0,
                            ):
                                pass

                # Trajectory management
                new_prompts, obj_ids, removed = self.tms.TrajectroyManage(
                    frame_id=frame_id,
                    inference_state=self.inference_state,
                    det_bboxes=det_bboxes,
                    det_logits=det_logits,
                    device=self.device,
                )

                socketio.emit(
                    "progress_update",
                    {
                        "message": f"[TRACKING] Frame {frame_id+1}/{total_frames}",
                        "progress": progress,
                    },
                    room=session_id,
                )

                # SAM2 tracking
                with torch.autocast("cuda", dtype=self.dtype):
                    if removed:
                        for obj_id in removed:
                            self.predictor.remove_object(
                                inference_state=self.inference_state, obj_id=obj_id
                            )
                    if len(new_prompts) > 0:
                        for i, obj_id in enumerate(obj_ids):
                            _, out_obj_ids, out_mask_logits = (
                                self.predictor.add_new_points_or_box(
                                    inference_state=self.inference_state,
                                    frame_idx=frame_id,
                                    obj_id=obj_id,
                                    box=np.array(new_prompts[i]),
                                )
                            )
                        for _, _, _ in self.predictor.propagate_in_video(
                            self.inference_state,
                            start_frame_idx=frame_id,
                            max_frame_num_to_track=0,
                        ):
                            pass

                # Occlusion handling
                occluded = self.ci.remove_occlusion(
                    inference_state=self.inference_state,
                    frame_id=frame_id,
                    device=self.device,
                )
                if occluded:
                    for obj_id in occluded:
                        if obj_id not in self.inference_state["obj_id_to_idx"]:
                            continue
                        obj_idx = self.inference_state["obj_id_to_idx"][obj_id]
                        out_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                        if frame_id in out_dict["non_cond_frame_outputs"].keys():
                            out_dict["non_cond_frame_outputs"][frame_id][
                                "object_score_logits"
                            ] = torch.tensor(
                                [[-1.0]], device=self.device, dtype=self.dtype
                            )

                # Save results and visualizations
                tck_ids, _, tck_bboxes, _, tck_masks = self.tms.get_trackinfo(
                    frame_idx=frame_id,
                    inference_state=self.inference_state,
                    device=self.device,
                )

                frame_path = os.path.join(video_dir, frame_names[frame_id])
                image = np.array(Image.open(frame_path).convert("RGB"))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                for i in range(len(tck_ids)):
                    obj_id = tck_ids[i]
                    bbox = tck_bboxes[i]
                    mask = tck_masks[i].cpu().numpy().astype(np.uint8)

                    bb_left = bbox[0]
                    bb_top = bbox[3]
                    bb_width = bbox[2] - bbox[0]
                    bb_height = bbox[3] - bbox[1]

                    res = [
                        frame_id + 1,
                        obj_id,
                        bb_left,
                        bb_top,
                        bb_width,
                        bb_height,
                        1,
                        -1,
                        -1,
                        -1,
                    ]
                    results.append(res)

                    # Append to file immediately
                    with open(output_path, "a") as f:
                        line = ",".join(map(str, res)) + "\n"
                        f.write(line)

                    image = show_mask(image, mask, obj_id=obj_id)
                    image = show_box(image=image, box=bbox, obj_id=obj_id)

                save_path = os.path.join(output_frames_dir, frame_names[frame_id])
                cv2.imwrite(save_path, image)
                eventlet.sleep(0)
                socketio.emit(
                    "frame_tracked",
                    {
                        "frame_id": frame_id + 1,
                        "image_url": f"/tracked_frame?path={save_path}&t={int(time.time()*1000)}",
                    },
                    room=session_id,
                )

                eventlet.sleep(0)

            end_time = time.time()
            fps_process = total_frames / (end_time - start_time)

            socketio.emit(
                "progress_update",
                {"message": "[FINALIZE] Creating output video...", "progress": 95},
                room=session_id,
            )
            eventlet.sleep(0)

            # Create output video
            output_mp4_path = os.path.join(
                os.path.dirname(output_frames_dir), "output_video.mp4"
            )
            create_video_from_images(
                image_folder=output_frames_dir,
                output_video_path=output_mp4_path,
                frame_rate=fps,
            )
            for jpg in glob.glob(os.path.join(output_frames_dir, "*.jpg")):
                os.remove(jpg)

            success_msg = f"Tracking completed successfully!\nProcessed {total_frames} frames in {end_time - start_time:.2f}s\nFPS: {fps_process:.2f}"

            socketio.emit(
                "progress_update",
                {"message": "[DONE] " + success_msg, "progress": 100},
                room=session_id,
            )
            eventlet.sleep(0)

            self.is_tracking = False
            return True, success_msg

        except Exception as e:
            error_msg = f"Error during tracking: {str(e)}"
            socketio.emit(
                "progress_update",
                {"message": error_msg, "progress": 0},
                room=session_id,
            )
            self.is_tracking = False
            return False, error_msg
        finally:
            self.cleanup()

    def stop_tracking(self):
        """Stop current tracking"""
        self.is_tracking = False

    def cleanup(self):
        """Clean up resources"""
        torch.cuda.empty_cache()

    def unload_models(self):
        self.detector = None
        self.predictor = None
        self.tms = None
        self.ci = None
        torch.cuda.empty_cache()


# Global tracker instance
tracker = SAM2MOTTracker()

import shutil
import subprocess
import uuid

from werkzeug.utils import secure_filename

UPLOAD_ROOT = "/tmp/sam2mot_videos"
os.makedirs(UPLOAD_ROOT, exist_ok=True)


def extract_frames_ffmpeg(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    # Use 8-digit padded filenames for sorting
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-q:v",
        "2",  # high quality
        os.path.join(frames_dir, "%08d.jpg"),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def extract_frames_cv2(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    idx = 1
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        out_path = os.path.join(frames_dir, f"{idx:08d}.jpg")
        cv2.imwrite(out_path, frame)
        idx += 1
    vid.release()


def extract_frames(video_path, frames_dir):
    # Try ffmpeg first
    try:
        extract_frames_ffmpeg(video_path, frames_dir)
    except Exception as ffmpeg_exc:
        print("[WARN] ffmpeg failed, falling back to OpenCV frame extraction...")
        try:
            extract_frames_cv2(video_path, frames_dir)
        except Exception as cv2_exc:
            raise RuntimeError(
                f"Both ffmpeg and OpenCV failed:\nffmpeg: {ffmpeg_exc}\ncv2: {cv2_exc}"
            )


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not video_file.filename.lower().endswith(".mp4"):
        return jsonify({"error": "File must be .mp4"}), 400

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_ROOT, session_id)
    os.makedirs(session_dir, exist_ok=True)
    video_save_path = os.path.join(session_dir, secure_filename(video_file.filename))
    frames_dir = os.path.join(session_dir, "frames")
    video_file.save(video_save_path)
    video_url = f"/uploads/{session_id}/{secure_filename(video_file.filename)}"

    try:
        extract_frames(video_save_path, frames_dir)
    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        return jsonify({"error": f"Frame extraction failed: {str(e)}"}), 500

    # Count frames
    num_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    fps = get_video_fps(video_save_path)
    return jsonify(
        {
            "session_id": session_id,
            "frames_dir": frames_dir,
            "video_filename": video_file.filename,
            "num_frames": num_frames,
            "video_url": video_url,
            "fps": int(fps),
        }
    )


@app.route("/uploads/<session_id>/<filename>")
def serve_uploaded_video(session_id, filename):
    directory = os.path.join(UPLOAD_ROOT, session_id)
    return send_from_directory(directory, filename)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    return jsonify(
        {
            "initialized": tracker.is_initialized,
            "tracking": tracker.is_tracking,
            "device": str(tracker.device),
        }
    )


@app.route("/api/initialize", methods=["POST"])
def initialize_models():
    if tracker.is_tracking:
        return jsonify({"error": "Cannot initialize while tracking"}), 400

    data = request.json

    # Update model paths if provided
    if "detector_cfg" in data:
        tracker.detector_cfg = data["detector_cfg"]
    if "detector_checkpoint" in data:
        tracker.detector_checkpoint = data["detector_checkpoint"]
    if "sam2_cfg" in data:
        tracker.sam2_model_cfg = data["sam2_cfg"]
    if "sam2_checkpoint" in data:
        tracker.sam2_checkpoint = data["sam2_checkpoint"]

    session_id = data.get("session_id", "default")

    # Run initialization in background thread
    def init_worker():
        tracker.initialize_models(session_id)

    eventlet.spawn_n(init_worker)

    return jsonify({"message": "Initialization started"})


@app.route("/api/unload", methods=["POST"])
def unload_models():
    tracker.unload_models()
    return jsonify({"message": "Models unloaded"})


@app.route("/api/track", methods=["POST"])
def start_tracking():
    if not tracker.is_initialized:
        return jsonify({"error": "Models not initialized"}), 400

    if tracker.is_tracking:
        return jsonify({"error": "Already tracking"}), 400

    data = request.json

    # Validate required fields
    required_fields = ["video_dir", "session_id", "fps"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # Set tracking parameters
    tracker.set_tracking_parameters(
        text_prompt=data.get("text_prompt", "person"),
        box_threshold=data.get("box_threshold", 0.35),
        text_threshold=data.get("text_threshold", 0.25),
    )

    video_dir = data["video_dir"]
    session_id = data["session_id"]
    fps = data["fps"]
    session_dir = os.path.join(UPLOAD_ROOT, session_id)
    os.makedirs(session_dir, exist_ok=True)

    output_path = os.path.join(session_dir, "tracker.txt")
    output_frames_dir = os.path.join(session_dir, "output_frames")

    os.makedirs(output_frames_dir, exist_ok=True)

    # Run tracking in background thread
    def track_worker():
        success, message = tracker.track_video(
            video_dir, output_path, output_frames_dir, session_id, fps
        )
        socketio.emit(
            "tracking_complete",
            {
                "success": success,
                "message": message,
                "output_video_url": f"/output_video?session={session_id}",
                "output_path": output_path,
                "video_path": os.path.join(session_dir, "output_video.mp4"),
            },
            room=session_id,
        )

    threading.Thread(target=track_worker, daemon=True).start()

    return jsonify({"message": "Tracking started"})


@app.route("/api/stop", methods=["POST"])
def stop_tracking():
    tracker.stop_tracking()
    return jsonify({"message": "Stopping tracking..."})


@app.route("/tracked_frame")
def tracked_frame():
    path = request.args.get("path")
    if not path or ".." in path:
        return "Invalid path", 400
    return send_file(path)


@app.route("/output_video")
def output_video():
    session = request.args.get("session")
    path = f"/tmp/sam2mot_videos/{session}/output_video.mp4"
    return send_file(path)


@app.route("/api/directories")
def list_directories():
    """List available directories"""
    path = request.args.get("path", "/")
    try:
        items = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                items.append({"name": item, "path": item_path, "type": "directory"})
        items.sort(key=lambda x: x["name"])
        return jsonify(
            {
                "current_path": path,
                "parent_path": os.path.dirname(path) if path != "/" else None,
                "items": items,
            }
        )
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@socketio.on("connect")
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit("connected", {"session_id": request.sid})


@socketio.on("disconnect")
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAM2MOT - Multi-Object Tracker</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .main-content {
            padding: 30px;
        }
        
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            background: #f9fafb;
        }
        
        .section h3 {
            color: #374151;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #374151;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #d1d5db;
            border-radius: 5px;
            font-size: 14px;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #4f46e5;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .btn {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            background: #9ca3af;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }
        
        .progress-container {
            margin-top: 20px;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .status {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            font-weight: 500;
        }
        
        .status.info {
            background: #dbeafe;
            color: #1e40af;
        }
        
        .status.success {
            background: #dcfce7;
            color: #166534;
        }
        
        .status.error {
            background: #fee2e2;
            color: #dc2626;
        }
        
        .log-container {
            height: 200px;
            overflow-y: auto;
            background: #1f2937;
            color: #e5e7eb;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        
        .device-info {
            background: #f0f9ff;
            border: 2px solid #0ea5e9;
            color: #0c4a6e;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SAM2MOT Tracker</h1>
            <p>Multi-Object Tracking with SAM2 and GroundingDINO</p>
        </div>
        
        <div class="main-content">
            <div id="deviceInfo" class="device-info">
                <strong>Device:</strong> <span id="deviceName">Loading...</span>
            </div>

            <div class="section" style="background: #e0f7fa;">
                <h3>🎦 Video Input</h3>
                <form id="videoUploadForm" enctype="multipart/form-data" style="margin-bottom: 10px;">
                    <label>
                    Select MP4 Video:
                    <input type="file" id="videoFile" name="video" accept="video/mp4" required>
                    </label>
                    <button type="submit" class="btn">Upload Video</button>
                </form>
                <div id="videoInfo"></div>
                <div id="videoPreview" style="margin:15px 0;max-width: 400px;"></div>
            </div>

            <div class="section">
                <h3>🔧 Model Configuration</h3>
                <div class="form-row">
                    <div class="form-group">
                        <label for="detectorCfg">GroundingDINO Config:</label>
                        <input type="text" id="detectorCfg" value="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py">
                    </div>
                    <div class="form-group">
                        <label for="detectorCheckpoint">GroundingDINO Checkpoint:</label>
                        <input type="text" id="detectorCheckpoint" value="./GroundingDINO/weights/groundingdino_swint_ogc.pth">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="sam2Cfg">SAM2 Config:</label>
                        <select id="sam2Cfg">
                            <option value="./configs/sam2.1/sam2.1_hiera_t.yaml">sam2.1_hiera_t.yaml</option>
                            <option value="./configs/sam2.1/sam2.1_hiera_s.yaml">sam2.1_hiera_s.yaml</option>
                            <option value="./configs/sam2.1/sam2.1_hiera_b+.yaml">sam2.1_hiera_b+.yaml</option>
                            <option value="./configs/sam2.1/sam2.1_hiera_l.yaml" selected>sam2.1_hiera_l.yaml</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="sam2Checkpoint">SAM2 Checkpoint:</label>
                        <select id="sam2Checkpoint">
                            <option value="./checkpoints/sam2.1_hiera_tiny.pt">sam2.1_hiera_tiny.pt</option>
                            <option value="./checkpoints/sam2.1_hiera_small.pt">sam2.1_hiera_small.pt</option>
                            <option value="./checkpoints/sam2.1_hiera_base_plus.pt">sam2.1_hiera_base_plus.pt</option>
                            <option value="./checkpoints/sam2.1_hiera_large.pt" selected>sam2.1_hiera_large.pt</option>
                        </select>
                    </div>
                </div>
                <button id="initBtn" class="btn">Initialize Models</button>
            </div>
            
            <div class="section">
                <h3>⚙️ Tracking Parameters</h3>
                <div class="form-row">
                    <div class="form-group">
                        <label for="textPrompt">Text Prompt:</label>
                        <input type="text" id="textPrompt" value="person">
                    </div>
                    <div class="form-group">
                        <label for="boxThreshold">Box Threshold: <span id="boxThresholdValue">0.35</span></label>
                        <input type="range" id="boxThreshold" min="0.1" max="1.0" step="0.05" value="0.35">
                    </div>
                </div>
                <div class="form-group">
                    <label for="textThreshold">Text Threshold: <span id="textThresholdValue">0.25</span></label>
                    <input type="range" id="textThreshold" min="0.1" max="1.0" step="0.05" value="0.25">
                </div>
            </div>
            
            <div class="section">
                <h3>🎬 Control Panel</h3>
                <button id="trackBtn" class="btn" disabled>Start Tracking</button>
                <button id="stopBtn" class="btn btn-danger" disabled>Stop Tracking</button>
                
                <div class="progress-container">
                    <div class="progress-bar">
                        <div id="progressFill" class="progress-fill"></div>
                    </div>
                    <div id="statusMessage" class="status info">Ready to initialize models...</div>
                </div>

                <div id="downloadLinkContainer" style="margin-top:12px;"></div>


            </div>
            
            <div class="section">
                <h3>📝 Log</h3>
                <div id="logContainer" class="log-container"></div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO
        const socket = io();
        let sessionId = null;
        let modelsInitialized = false;
        
        // DOM elements
        const initBtn = document.getElementById('initBtn');
        const trackBtn = document.getElementById('trackBtn');
        const stopBtn = document.getElementById('stopBtn');
        const progressFill = document.getElementById('progressFill');
        const statusMessage = document.getElementById('statusMessage');
        const logContainer = document.getElementById('logContainer');
        const deviceName = document.getElementById('deviceName');
        
        // Threshold sliders
        const boxThreshold = document.getElementById('boxThreshold');
        const textThreshold = document.getElementById('textThreshold');
        const boxThresholdValue = document.getElementById('boxThresholdValue');
        const textThresholdValue = document.getElementById('textThresholdValue');
        
        // Update threshold displays
        boxThreshold.addEventListener('input', () => {
            boxThresholdValue.textContent = boxThreshold.value;
        });
        
        textThreshold.addEventListener('input', () => {
            textThresholdValue.textContent = textThreshold.value;
        });
        
        // Utility functions
        function logMessage(message) {
            const timestamp = new Date().toLocaleTimeString();
            logContainer.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        function updateProgress(progress) {
            progressFill.style.width = progress + '%';
        }
        
        function updateStatus(message, type = 'info') {
            statusMessage.textContent = message;
            statusMessage.className = `status ${type}`;
        }
        
        function setButtonStates(init, track, stop) {
            initBtn.disabled = !init;
            trackBtn.disabled = !track;
            stopBtn.disabled = !stop;
        }
        
        // Socket event handlers
        // const trackedImg = document.getElementById('trackedFrameImg');
        // const frameLabel = document.getElementById('frameLabel');

        // socket.on('frame_tracked', (data) => {
        //     frameLabel.textContent = `Tracking Frame ${data.frame_id}`;
        //     trackedImg.src = data.image_url;
        //     trackedImg.style.display = '';
        // });


        socket.on('connected', (data) => {
            sessionId = data.session_id;
            logMessage('Connected to server');
            loadStatus();
        });
        
        socket.on('progress_update', (data) => {
            updateStatus(data.message);
            if (data.progress !== undefined) {
                updateProgress(data.progress);
            }
            logMessage(data.message);
        });
        
        socket.on('models_initialized', (data) => {
            if (data.success) {
                modelsInitialized = true;
                updateStatus('Models initialized successfully!', 'success');
                setButtonStates(true, true, false);
                logMessage('✅ Models ready for tracking');
            } else {
                updateStatus('Model initialization failed', 'error');
                setButtonStates(true, false, false);
                logMessage('❌ Model initialization failed: ' + (data.error || 'Unknown error'));
            }
        });
        
        socket.on('tracking_complete', (data) => {
            if (data.success) {
                updateStatus('Tracking completed successfully!', 'success');
                logMessage('✅ Tracking completed');
                logMessage('📄 Results saved to: ' + data.output_path);
                logMessage('🎥 Video saved to: ' + data.video_path);
                // document.getElementById('liveTrackedFrame').innerHTML =
                // //      `<video controls style="max-width:520px;border-radius:12px;box-shadow:0 2px 14px #0002;" src="${data.output_video_url}"></video>`;

                // Show download button
                document.getElementById('downloadLinkContainer').innerHTML =
                    `<a href="${data.output_video_url}" download="output_video.mp4" class="btn" style="display:inline-block;margin-top:10px;">
                        ⬇️ Download Output Video
                    </a>`;
            } else {
                updateStatus('Tracking failed', 'error');
                logMessage('❌ Tracking failed: ' + data.message);
            }
            setButtonStates(true, true, false);
            updateProgress(100);
        });

        // Button event handlers
        initBtn.addEventListener('click', async () => {
            setButtonStates(false, false, false);
            updateStatus('Initializing models...', 'info');
            updateProgress(0);
            
            const response = await fetch('/api/initialize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: sessionId,
                    detector_cfg: document.getElementById('detectorCfg').value,
                    detector_checkpoint: document.getElementById('detectorCheckpoint').value,
                    sam2_cfg: document.getElementById('sam2Cfg').value,
                    sam2_checkpoint: document.getElementById('sam2Checkpoint').value
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                updateStatus('Failed to start initialization', 'error');
                logMessage('❌ ' + error.error);
                setButtonStates(true, false, false);
            }
        });
        
        trackBtn.addEventListener('click', async () => {
            
            setButtonStates(false, false, true);
            updateStatus('Starting tracking...', 'info');
            updateProgress(0);
            
            const response = await fetch('/api/track', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: sessionId,
                    video_dir: sessionFramesDir,
                    text_prompt: document.getElementById('textPrompt').value,
                    box_threshold: parseFloat(document.getElementById('boxThreshold').value),
                    text_threshold: parseFloat(document.getElementById('textThreshold').value),
                    fps: videoFps
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                updateStatus('Failed to start tracking', 'error');
                logMessage('❌ ' + error.error);
                setButtonStates(true, true, false);
            }
        });
        
        stopBtn.addEventListener('click', async () => {
            const response = await fetch('/api/stop', {
                method: 'POST'
            });
            
            if (response.ok) {
                updateStatus('Stopping tracking...', 'info');
                logMessage('🛑 Stop signal sent');
            }
        });
        
        // Load initial status
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                deviceName.textContent = status.device;
                modelsInitialized = status.initialized;
                
                if (status.initialized) {
                    updateStatus('Models already initialized', 'success');
                    setButtonStates(true, true, false);
                } else {
                    updateStatus('Ready to initialize models...', 'info');
                    setButtonStates(true, false, false);
                }
                
                if (status.tracking) {
                    updateStatus('Tracking in progress...', 'info');
                    setButtonStates(false, false, true);
                }
                
                logMessage('Status loaded - Device: ' + status.device + ', Initialized: ' + status.initialized);
            } catch (error) {
                logMessage('❌ Failed to load status: ' + error.message);
            }
        }
    </script>
    <script>
            let sessionFramesDir = null;
            let uploadedVideoName = null;
            let numFrames = 0;
            let videoFps = 20;

            // Handle video upload
            document.getElementById('videoUploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const videoInput = document.getElementById('videoFile');
            if (!videoInput.files[0]) return;

            const formData = new FormData();
            formData.append('video', videoInput.files[0]);

            updateStatus('Uploading video and extracting frames...', 'info');
            logMessage('Uploading video...');
            const res = await fetch('/api/upload_video', {method: 'POST', body: formData});
            const data = await res.json();

            if (!res.ok) {
                logMessage('❌ ' + (data.error || 'Video upload failed.'));
                updateStatus('Upload failed: ' + (data.error || 'Unknown error'), 'error');
                return;
            }

            sessionFramesDir = data.frames_dir;
            uploadedVideoName = data.video_filename;
            numFrames = data.num_frames;
            videoFps = data.fps;
            document.getElementById('videoInfo').innerHTML = 
                `<div style="color:#059669"><b>Uploaded:</b> ${uploadedVideoName} (${numFrames} frames extracted)</div>`;
            document.getElementById('videoPreview').innerHTML =
                `<video controls width="500" style="max-width:100%;">
                    <source src="${data.video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>`;
            logMessage(`✅ Video uploaded: ${uploadedVideoName}, ${numFrames} frames ready.`);
            checkEnableTracking();
            });

            // Only enable tracking when both are set
            function checkEnableTracking() {
            trackBtn.disabled = !(sessionFramesDir && modelsInitialized);
            }
        </script>
</body>
</html>"""


# Create templates directory and save HTML template
def create_template():
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    with open(templates_dir / "index.html", "w") as f:
        f.write(HTML_TEMPLATE)


if __name__ == "__main__":
    create_template()
    print("🚀 Starting SAM2MOT Web Application...")
    print("📱 Access the application at: http://localhost:6011")
    print("🌐 Or from remote: http://0.0.0.0:6011")
    print("💡 Make sure to open port 6011 in your firewall!")

    # Run with eventlet for better WebSocket support
    socketio.run(app, host="0.0.0.0", port=6011, debug=False)
