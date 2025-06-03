# Report CAS-ADS 24-25

## 1. Introduction

In recent years, the use of video recordings in educational contexts has become increasingly common. Videos are captured for teacher training, classroom observation, analysis of student behavior, and a wide range of pedagogical and research purposes. These recordings often contain personally identifiable information, such as the faces of students and teachers, voices, or background details that can reveal identity. As a result, their use is heavily regulated and subject to ethical constraints.

Despite the growing availability of video data, teachers and educational researchers face significant challenges in using this material. Many educators record valuable content that they ultimately cannot exploit — not because of a lack of educational relevance, but because of legal and ethical concerns related to privacy. Consent is a fragile foundation: parents or legal guardians can revoke permission to use a video at any time, and when this happens, the project may be jeopardized or entire recordings may have to be discarded. This fragility makes it extremely difficult to build sustainable long-term video-based projects or datasets.

Moreover, manual anonymization is labor-intensive and technically complex, and most educators do not have the time, tools, or expertise to do it reliably. Commercial solutions, when available, are often not adapted to the specific needs of academic environments — they may be costly, closed-source, or too generic. As a result, teachers are left with tons of valuable but unusable video data, constrained by their responsibility to protect personal data and comply with strict data protection laws such as the General Data Protection Regulation (GDPR) and the Swiss Federal Act on Data Protection (FADP).

This report presents a solution to this pressing need: a lightweight, transparent, and freely available video anonymization tool designed specifically for educators. Developed at the University of Teacher Education, Vaud (HEP Vaud), this software empowers teachers and researchers to anonymize video content easily and securely. It ensures that PII such as facial data is removed or pseudonymized, thereby making videos reusable even if consent is later withdrawn, and opening new possibilities for ethical educational research and practice.


## 2. Theoretical Foundations
The anonymization of video data in educational settings involves complex challenges at the intersection of technology, ethics, and law. This chapter provides the theoretical background necessary to understand the importance and complexity of anonymizing personally identifiable information in audiovisual material. It introduces key definitions, discusses the legal and ethical frameworks guiding anonymization practices, and highlights the technical difficulties involved in automating these processes.


### 2.1 What is Video Anonymization?
Video anonymization refers to the process of modifying video content in a way that removes or conceals information that can be used to identify individuals [1]. This typically includes:

-	Faces: The most common biometric identifier in video.
-	Voices: Can also reveal identity and must be anonymized in some contexts.
-	Background elements: Posters, screens, or badges that might show names, school names, or personal details.
- 	Text overlays: Subtitles, annotations, or names displayed on screen.

The goal is to protect the privacy of individuals captured in video footage without significantly degrading the usability of the content for educational or research purposes. We will only focus on faces and common objects in this project. 

A crucial distinction must be made between:
- Anonymization: The irreversible removal of identifying data such that individuals can no longer be identified by any means reasonably likely to be used.
- Pseudonymization: The replacement of identifying data with pseudonyms or placeholders, where the original data is stored separately and can be re-identified under certain conditions.

In this project, face anonymization via blurring or masking is considered a form of pseudonymization, especially if the raw footage is preserved. For complete anonymization, the original data must be deleted or made inaccessible.

### 2.2 Ethical and Legal Considerations
The ethical handling of video data is especially important in educational contexts, where recordings often involve minors, teachers, or vulnerable populations. Even when legal requirements are met, ethical issues may persist regarding informed consent, purpose limitation, and the right to withdraw participation.

Two major legal frameworks govern the use and processing of personal data in Switzerland and Europe :

General Data Protection Regulation (GDPR) (EU) [1]:
- Defines video data containing identifiable persons as personal data.
- Requires informed and explicit consent before processing.
- Grants the right to withdraw consent at any time.
- Encourages privacy by design and privacy by default.

Swiss Federal Act on Data Protection (FADP) [2]:
- Aligns with GDPR in many aspects.
- Requires proportionality and purpose limitation in data collection.
- Grants individuals the right to access, correct, or demand the deletion of their data.	

Educational institutions must go beyond legal compliance and uphold ethical principles such as [5]:

- Respect for autonomy: Individuals must be informed and able to make decisions about their data.
- Data minimization: Only collect what is necessary.
- Transparency: Inform data subjects clearly about data use.
- Accountability: Keep records of consent and anonymization procedures.

In research, ethics boards often refuse or limit the use of video data that has not been properly anonymized, especially when it involves children. Consent withdrawal by a single participant can force the exclusion of entire video datasets. A reliable anonymization tool mitigates this risk and ensures research continuity.

### 2.3 Technical Challenges

Despite its importance, video anonymization remains a technically complex task. Automating the process requires overcoming several challenges [3][4]:

**Dynamic Elements**

Unlike static images, video frames contain:
- Moving faces that change in position, lighting, orientation, and size.
- Occlusions (e.g., students turning, hands covering faces).
- Multiple people entering and leaving the frame.

Robust anonymization must detect all faces across all frames, even in difficult conditions.

**Anonymization Quality vs. Data Usability**

Anonymizing video comes with trade-offs:
- Too much blurring can make pedagogical analysis (e.g., of student reactions or classroom behavior) impossible.
- Too little blurring may not sufficiently protect privacy.

The ideal solution strikes a balance between privacy protection and educational usefulness, with adjustable parameters.

**Real-Time Performance**

In practice, teachers or researchers may need to anonymize long videos quickly. The anonymization system must:
- Run on standard hardware (laptops, desktops)
- Process large video files without excessive delay
- Be easy to use with minimal technical knowledge

These constraints informed the technical choices in this project, including the selection of a real-time capable face detection model and the use of efficient video processing libraries.

### 2.4 Conclusion

An effective anonymization tool must address a wide range of ethical, legal, and technical challenges. In education, the stakes are particularly high: without anonymization, valuable video data may remain locked away, unusable due to privacy concerns. By grounding the tool in these theoretical foundations, this project ensures not only technical functionality but also alignment with the responsibilities and values of educational institutions.

# 3. Face Detection with RT-DETR

In this chapter, we detail the process of selecting, adapting, and training a deep learning model for the purpose of automatic face detection in video anonymization. This process was central to the development of our anonymization tool. It involved overcoming several hardware, dataset, and performance constraints to produce a model that performs reliably in real-world educational settings.

## 3.1 Why Face Detection Is Essential

In any anonymization pipeline involving video, the first and most critical step is accurate detection of personally identifiable information (PII). Among all types of PII, faces are the most prominent and sensitive, as they allow for direct identification of individuals. In educational contexts, this is especially important given that videos often contain footage of minors or teachers in classrooms.

A reliable face detector ensures that all faces appearing in video frames are located precisely. This enables downstream anonymization operations, such as blurring, pixelation, inapinting or masking, to be applied effectively. If detection fails or is inaccurate, anonymization cannot be guaranteed, potentially leading to privacy breaches or non-compliance with data protection laws.

For this reason, we decided to build a custom-trained face detection model, tailored for high accuracy in diverse environments.

## 3.2 Model Selection: RT-DETR-X

The selection of the object detection architecture was guided by two hard constraints:
- The model must run on standard Apple M1 MacBook Airs used by university staff (no discrete GPUs).
- The model must also be trainable and deployable on CUDA-compatible computers with NVIDIA GPUs.

After evaluating several object detection frameworks, two architectures stood out for their support of both CUDA and Apple’s Metal Performance Shaders (MPS):
- YOLO (You Only Look Once): a popular family of object detectors, known for speed and ease of deployment.
- RT-DETR (Real-Time Detection Transformer): a newer transformer-based model offering competitive accuracy and speed, especially on complex scenes.

While YOLO models are widely used and optimized, RT-DETR demonstrated superior accuracy in scenarios involving occlusions [6], dense scenes, and varied lighting conditions, which are common in real-world classroom footage. RT-DETR also integrates well with the Ultralytics framework, enabling consistent workflows across platforms.

<img src="yolovsrtdtr.png" alt="My image" width="500"/>


We selected RT-DETR-X, the extended and more capable variant of RT-DETR, as the base model for this project.

## 3.3 Dataset: WIDER Face
The WIDER Face dataset [7] was chosen for fine-tuning the RT-DETR-X model. 
It is widely regarded as a benchmark for evaluating face detection algorithms, due to its diversity and difficulty:
- Over 132,000 images and 393,000 annotated faces.
- Faces appear in various conditions: extreme poses, occlusions, crowded scenes, low resolution, and non-frontal angles.
- Divided into training, validation, and test sets.

<img src="wider-face.jpg" alt="My image" width="500"/>

This makes WIDER Face particularly suited for training a face detector that needs to generalize to diverse educational environments, such as group activities, low lighting, or partially visible students.

However, WIDER Face is not natively formatted for training with RT-DETR or Ultralytics. Therefore, we had to preprocess the data accordingly.

WIDER Face provides its annotations in a custom .txt format that lists bounding box coordinates for faces within each image. 
To use these annotations with RT-DETR via the Ultralytics interface, the following steps were necessary:

Custom Conversion Scripts:
- We wrote two scripts: train2yolo.py and val2yolo.py.
- These scripts parsed the WIDER Face annotation files and generated one YOLO-style .txt file per image, listing bounding boxes in normalized coordinates.

Dataset Organization:
- Images and labels were structured into folders following the YOLO convention: images/train, labels/train, images/val, labels/val.

Configuration File:
- A wider.yaml file was created to define the dataset class (face), training/validation image paths, and number of classes.

This preprocessing ensured seamless compatibility with the Ultralytics RT-DETR training pipeline.

## 3.4 Training the Model
Due to the limitations of local hardware, training was not possible on a MacBook Air. We rented a cloud server via Lambda Labs, equipped with a NVIDIA GH200 GPU and 96 GB of GPU RAM. Despite this powerful setup, training a transformer-based model like RT-DETR-X on a large dataset remained computationally expensive.

<img src="nvidia.png" alt="My image" width="500"/>

**Initial Strategy: Multi-Task Training**

We initially attempted to combine COCO and WIDER Face datasets, training RT-DETR-X to detect both the original 80 COCO classes and an additional “face” class. However, this approach failed due to:
- Memory constraints: Batch sizes had to be reduced drastically, slowing training dramatically.
- Data imbalance: WIDER Face provides far more images per class than COCO, leading to catastrophic forgetting. The model began to ignore the COCO classes and focused almost exclusively on faces.

**Final Strategy: Dedicated Face Model**

To resolve this, we retrained RT-DETR-X exclusively on WIDER Face, resulting in a dedicated face detection model. The training notebook, train-RT-DETR-X-Face.ipynb, defines the process:
- Epochs: 300
- Image size: 640×640
- Batch size: 8 
- AMP: Enabled for mixed-precision speedup
- Optimizer: Default Ultralytics settings

The training process was monitored with loss curves and validation metrics.

**Results**

The model achieved:
- mAP@0.5 = 0.579 on the WIDER Face validation set.

<img src="train_results.png" alt="My image" width="500"/>

<img src="train_inference.jpeg" alt="My image" width="500"/>

While this score is lower than state-of-the-art models like RetinaFace [8] (which reach ~0.9 mAP@0.5), it is acceptable for anonymization purposes. Additional improvements (e.g.hyperparameter tuning, longer training, better augmentation) could improve performance, but were not feasible due to cost and time constraints.

## 3.5 Deployment architecture

Due to the limitations encountered during multi-task training, we adopted a dual-model architecture:
- One default RT-DETR-X model pre-trained on COCO for general object detection.
- One fine-tuned RT-DETR-X model for face detection only, trained on WIDER Face.

At runtime, both models can be used sequentially or selectively:
- For anonymization, only the face detector is needed.
- For educational analysis, the COCO model can provide additional contextual detections.

Both models are lightweight enough to run on Apple Silicon using MPS acceleration, making the application usable on standard university hardware.

## Conclusion

This chapter has shown the step-by-step process used to build a custom face detection model tailored for educational video anonymization. By selecting RT-DETR-X for its balance of speed and accuracy, and by training it on the WIDER Face dataset, we developed a robust detector capable of operating in real-time on constrained hardware. Despite the computational cost and training challenges, the resulting model is fit for purpose and can be seamlessly integrated into the anonymization pipeline.

# 4. Application Development

The anonymization application developed for this project is composed of two main components: a frontend responsible for user interaction, and a backend that handles all computational tasks, including video analysis and anonymization. This architecture was chosen to balance flexibility, performance, and usability, particularly in environments where computational resources vary significantly.

## 4.1 Architecture overview

**Frontend (Electron app with Quasar Framework)**

The frontend is developed using Electron and the Quasar Framework (Vue.js) using Typescipt. It provides an intuitive and responsive user interface through which users can:
- Upload and preview video files
- Define detection and anonymization settings
- Select specific objects and regions to anonymize
- Monitor the progress of detection and processing tasks

**Backend (Python)**

The backend is built in Python, using:
- PyTorch for model inference
- OpenCV and ffmpeg for video decoding and encoding
- Additional libraries for object tracking and mask-based anonymization

The backend is responsible for:
- Object detection
- Video frame extraction and processing
- Anonymization using segmentation and inpainting
- Rebuilding the final anonymized video

**Communication via WebSocket**

Frontend and backend components communicate using WebSocket protocol, which allows for real-time, bidirectional data exchange. This enables:
- Continuous feedback during detection and processing
- Dynamic object preview and annotation
- Asynchronous task handling

This separation also offers deployment flexibility:
- The backend can run locally on powerful machines (e.g., M1/M2 laptops with sufficient RAM).
- It can also run remotely on a dedicated server, enabling lightweight clients to access advanced processing.

<img src="architecture.png" alt="My image" width="500"/>

## 4.2 Anonymization workflow

The anonymization pipeline follows a structured and interactive and linear process:

### Step 0: Initialization and Project Setup
- The user launches the application.
- A system check evaluates whether the local machine has sufficient compute capabilities.
- The user is prompted to select or create a workspace, which is then initialized with all required resources (models, dependencies).
- The user creates a project, selecting a video to process.

### Step 1: Video Trimming

In most cases, only specific segments of a video contain sensitive data.
- The app includes a video trimming tool that allows users to extract only the relevant portion of the footage.
- This reduces processing time and resource usage while improving precision.

### Step 2: Object Detection Setup

- The user selects which objects they want to detect.

Options include:
- 80+ COCO classes (e.g., person, backpack, chair)
- Faces (using the custom RT-DETR-X face model)
- Once selection is made, the backend performs object detection across all frames.
- A tracking algorithm (ByteTrack) is applied to track object instances across time.

This allows users to:
- Follow the same object across frames
- Interact with and select objects more intuitively in the next step

### Step 3: Object Selection for Anonymization
- The user is presented with a fully annotated version of the video, where all detections are visible.
- Using the UI, the user can:
- Select specific objects (e.g., only certain people or only faces)
- Select all detections of a certain type
- Apply filters to exclude or include only tracked objects

This manual validation step is essential to ensure that only relevant detections are anonymized, avoiding over-blurring or loss of useful pedagogical content.

### Step 4: Anonymization
- Once object selection is complete, the frontend sends the video and associated detection metadata to the backend.
- The backend processes the video frame by frame, and for each bounding box, the following pipeline is applied:

**Mask Generation with FastSAM-X**
- FastSAM-X is a real-time segmentation model that produces precise masks from bounding boxes [9].
- It is lightweight, fast, and compatible with Apple Silicon (M1/M2).
- The model outputs a binary segmentation mask corresponding to the exact shape of the detected object.

**Anonymization Method: Blur or Inpaint**

Depending on the user’s settings, one of two anonymization strategies is applied:

- Blurring
	- A Gaussian blur is applied within the mask.
	- This is fast and effective but retains some visual presence of the object.

- Inpainting with Big-Lama
	- For complete anonymization, we use Big-Lama, a state-of-the-art deep image inpainting model [10].
	- Big-Lama reconstructs the masked area with plausible background content, making it appear as if the object was never present.
	- Like FastSAM, Big-Lama is Apple Silicon–compatible, enabling local processing on modern macOS laptops.

- Video Reconstruction
	- Once all frames are processed:
	- The backend reassembles the video using ffmpeg, preserving original frame rate and audio.
	- The output video is stored in the project workspace and made available for download via the frontend.


⸻

5. Evaluation
	•	Usability testing (if any was done)
	•	Performance analysis (speed, CPU/GPU usage)
	•	Accuracy of anonymization
	•	Feedback from teachers or testers

⸻

6. Discussion
	•	Strengths of the approach
	•	Limitations (e.g., non-face PII not handled)
	•	Opportunities for improvement (e.g., voice anonymization, active learning)

⸻

7. Conclusion
	•	Summary of contributions
	•	Impact on educational data privacy
	•	Future directions (e.g., full pipeline anonymization, open-source release)

⸻

8. Appendices
	•	Code snippets and usage examples
	•	Installation instructions
	•	User guide / quick start
	•	Model training logs or charts

⸻

9. References

[1]: https://egonym.com/blog/best-video-anonymization-software-for-privacy-compliance
[2]: https://forscenter.ch/wp-content/uploads/2020/06/kleinerstam_fg11_anonymisation1_v1.0-1.pdf
[3]: https://datafromsky.com/news/dynamic-anonymization-anonymized-video-streaming
[4]: https://www.immuta.com/blog/data-anonymization-techniques
[5]: https://arxiv.org/abs/1710.06881
[6]: https://arxiv.org/pdf/2304.08069
[7]: http://shuoyang1213.me/WIDERFACE/
[8]: https://github.com/serengil/retinaface
[9]: https://docs.ultralytics.com/fr/models/fast-sam/
[10]: https://github.com/advimman/lama