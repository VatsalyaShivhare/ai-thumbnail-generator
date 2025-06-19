# 🎯 Thumbnail Generator – AI-Powered Video Frame Enhancer

## 📦 Project Overview

This application automatically extracts and enhances video frames using AI and computer vision. It generates high-quality thumbnails from videos with:

- ✅ Key frame extraction
- ✅ Object & face detection (YOLO + OpenCV)
- ✅ Frame quality scoring
- ✅ Scene classification (Hugging Face ViT)
- ✅ Color palette extraction
- ✅ A/B thumbnail variants with filters and overlays

---

## 🖥️ How to Run

### ✅ Requirements

- Python 3.8+
- Windows or Linux/macOS
- Video files in `.mp4` format

| Task                 | Model Used                                       |
| -------------------- | ------------------------------------------------ |
| Object Detection     | YOLOv8 (via Ultralytics)                         |
| Face Detection       | OpenCV Haar Cascades                             |
| Scene Classification | ViT (`facebook/deit-base-distilled-patch16-224`) |
| Color Palette        | scikit-learn KMeans                              |
