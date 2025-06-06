# 🛡️ Samantha – A Video Anonymization Tool for Education 

**CAS Applied Data Science 2024 – Final Project**  

**University of Bern & HEP Vaud**

**marcel.grosjean@hepl.ch**

![Status](https://img.shields.io/badge/Status-Finished-brightgreen)
![Model](https://img.shields.io/badge/Model-RT--DETR--X-orange)
![Hardware](https://img.shields.io/badge/Compatible-Apple_Silicon_and_CUDA-green)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

**Samantha** is a privacy-preserving video anonymization app designed specifically for educators and researchers. It enables the secure use of classroom video data while complying with strict legal and ethical standards. 

This anonymization software was developed at the University of Teacher Education, Vaud (HEP Vaud), to facilitate the ethical and secure handling of sensitive data used in research and educational projects. It enables the automatic anonymization of video files by removing or pseudonymizing personally identifiable information, while ensuring traceability and reproducibility of the applied processes.

Designed to meet the specific needs of the academic environment, the tool emphasizes ease of use, process transparency, and compliance with current ethical and legal standards, including data protection regulations (GDPR, Swiss FADP).

The development of this software is part of a broader commitment to responsible research and open knowledge sharing, offering a free, adaptable, and sustainable solution to the educational community.

This project was developed as part of the final project in the **Certificate of Advanced Studies in Applied Data Science (CAS-ADS)**.

## 📄 Final Report

The detailed project report — including model training, architecture, anonymization pipeline, evaluation, and references — is available in the [`/report/final_report.pdf`](./report) folder.

---

## 🎥 Demo

This is a demo of the global workflow:

![Anonymization workflow demo](./report/steps_all.gif)


---

## 📁 Repository Structure

---

## 🎯 Project Goals

- Allow teachers and researchers to **anonymize faces and objects** in educational videos.
- Use **deep learning models** (RT-DETR-X, FastSAM-X, Big-Lama) optimized for both **Apple Silicon and CUDA GPUs**.
- Provide a **user-friendly, cross-platform application** for non-technical users.

---

## ⚙️ Technologies Used

| Component        | Tech Stack                                |
|------------------|--------------------------------------------|
| Frontend         | Vue.js (Quasar), Electron, TypeScript      |
| Backend          | Python, PyTorch, OpenCV, ffmpeg            |
| Models           | RT-DETR-X, FastSAM-X, Big-Lama             |
| Hardware Support | Apple Silicon, CUDA                        |

---

## 🧠 Deep Learning Models

- 🟠 **RT-DETR-X** – real-time transformer model for object detection  
- 🟡 **RT-DETR-Face** – fine-tuned on WIDER Face dataset for high-accuracy face detection  
- 🔵 **FastSAM-X** – segment-anything model for object masking  
- 🔴 **Big-Lama** – inpainting model to erase sensitive content

All models are chosen for **speed, accuracy, and Apple Silicon compatibility**.

---

## 🧪 Current Status

- ✅ Detection pipeline works reliably for most educational videos  
- ✅ UI is fully functional and tested with real users  
- ⚠️ Anonymization (especially inpainting) can be slow  
- ⚠️ Tracking and mask precision still have limitations  
- ⚠️ App is **not yet ready for deployment**, still in development  

Final release goals:
- One-click installers for **Windows, macOS, and Linux**
- Better tracking & inpainting accuracy
- Manual annotation tools for full control

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to fork, contribute, or adapt for your educational or research use.

---

> Developed with care at the University of Teacher Education, Vaud (HEP Vaud) & the University of Bern, 2024.