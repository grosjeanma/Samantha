# 🛡️ Samantha – A Video Anonymization App

**Privacy-first desktop app for anonymizing educational videos**  
Built with **Quasar (Vue)** for the frontend and **Python** for the backend.

---

## 📦 Features

- 🎥 Cut and anonymize selected video segments
- 🧠 Deep learning object detection (RT-DETR & RT-DETR-Face)
- 🕵️ Face & object masking with FastSAM-X
- 🧽 Inpainting with Big-Lama
- 💻 Runs locally on Apple Silicon or CUDA devices
- 🔄 Modular backend via WebSocket communication

---

## 🖥️ Technologies Used

| Layer     | Stack                                   |
|-----------|------------------------------------------|
| Frontend  | Quasar (Vue 3) + Electron                |
| Backend   | Python + PyTorch + OpenCV + ffmpeg       |
| Detection | RT-DETR-X (COCO) + RT-DETR-Face (WIDER)  |
| Masking   | FastSAM-X                                |
| Inpainting| Big-Lama                                 |

---

## 🚀 Getting Started

### 🔧 Prerequisites

- **Node.js (LTS or latest)** for the frontend
- **Python 3.10+** for the backend
- [Git LFS](https://git-lfs.github.com/) for large model files

---

## 🖼️ Frontend Setup

To run the desktop application:

```bash
npm install -g @quasar/cli
npm install
quasar dev -m electron
```

---

## 🧠 Backend Setup

To run the backend server:

```bash
pip install -r requirements src-python/requirements.txt
nodemon src-python/main.py
```



---

## 🔄 Architecture Overview

- The **frontend** handles user interaction and controls.
- The **backend** performs object detection, video decoding, anonymization, and returns results.
- Both communicate via **WebSocket**, enabling either local or remote execution.

---

## ⚠️ Known Limitations

- Detection can be unreliable in complex or fast-moving scenes
- Masks may be imperfect; ghosting possible with inpainting
- Performance depends on hardware — anonymization may be slow
- Manual anonymization tools are still in progress


---

## 📜 License

MIT License – Free to use, adapt, and distribute.

---

> Developed by Marcel Grosjean at HEP Vaud, as part of the CAS Applied Data Science 2024 (University of Bern)