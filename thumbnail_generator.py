import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.cluster import KMeans
from transformers import AutoProcessor, AutoModelForImageClassification
import torch

# --- CONFIG ---
video_folder = "videos"
output_folder = "thumbnails"
scene_model_name = "facebook/deit-base-distilled-patch16-224"
device = "cuda" if torch.cuda.is_available() else "cpu"
top_k = 5
frame_interval = 30

# --- MODELS ---
yolo_model = YOLO("yolov8n.pt")
scene_processor = AutoProcessor.from_pretrained(scene_model_name, use_fast=True)
scene_model = AutoModelForImageClassification.from_pretrained(scene_model_name).to(device)

# --- FUNCTIONS ---
def extract_frames(video_path, interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append((count, frame))
        count += 1
    cap.release()
    return frames

def get_focus_score(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def detect_objects(image):
    results = yolo_model.predict(source=image, save=False, verbose=False)
    return results[0].boxes.xyxy.cpu().numpy()

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def classify_scene(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = scene_processor(images=img, return_tensors="pt").to(device)
    outputs = scene_model(**inputs)
    label = outputs.logits.argmax(-1).item()
    return scene_model.config.id2label[label]

def extract_palette(image, k=5):
    img = cv2.resize(image, (100, 100))
    data = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10).fit(data)
    return kmeans.cluster_centers_.astype(int)

def apply_overlay(image, label, palette):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    # Text overlay
    draw.text((10, 10), label, fill=(255, 255, 255, 255), font=font)

    # Palette bar
    for i, color in enumerate(palette):
        box = [10 + i * 30, pil_img.height - 30, 30 + i * 30, pil_img.height - 10]
        draw.rectangle(box, fill=tuple(map(int, color)) + (255,))
    return pil_img

def save_thumbnail(image, path):
    image.convert("RGB").save(path, quality=95)

# --- MAIN ---
if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(video_folder):
        if not video_file.endswith(".mp4"):
            continue
        name = os.path.splitext(video_file)[0]
        print(f"\n🎞️ Processing: {name}")
        video_path = os.path.join(video_folder, video_file)
        out_dir = os.path.join(output_folder, name)
        os.makedirs(out_dir, exist_ok=True)

        frames = extract_frames(video_path, frame_interval)
        scored = []

        for frame_no, frame in frames:
            objects = detect_objects(frame)
            score = get_focus_score(frame)
            scored.append((frame_no, frame, len(objects), score))

        # Rank by (object count + score)
        ranked = sorted(scored, key=lambda x: (x[2], x[3]), reverse=True)[:top_k]

        for i, (frame_no, frame, obj_count, score) in enumerate(ranked):
            scene = classify_scene(frame)
            print(f"[A] Frame {frame_no} → Scene: {scene}")
            print(f"[B] Frame {frame_no} → Label: Top Moment")

            # --- A: Face Crop Thumbnail ---
            faces = detect_faces(frame)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_crop = frame[y:y+h, x:x+w]
            else:
                face_crop = frame
            thumbA = apply_overlay(face_crop, f"{scene}", extract_palette(frame))
            save_thumbnail(thumbA, os.path.join(out_dir, f"thumb_{frame_no}_a.jpg"))

            # --- B: Full Frame + Effects Thumbnail ---
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            overlayB = apply_overlay(blurred, f"{scene}", extract_palette(frame))
            save_thumbnail(overlayB, os.path.join(out_dir, f"thumb_{frame_no}_b.jpg"))

        print(f"✅ Done: {len(ranked)} thumbnails saved to '{out_dir}'")

