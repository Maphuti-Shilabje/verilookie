
## **Slide 1: Title**

**Verilookie: A Deepfake and AI-Generated Media Detector**
Prototype – Web-based Solution

---

## **Slide 2: Problem Statement**

* The rise of **deepfakes and AI-generated media** poses threats to:

  * Individuals (identity theft, fraud, blackmail)
  * Society (misinformation, fake news, propaganda)
* Lack of tools for **easy verification** of media authenticity
* Users need both **protection and education** to stay resilient

---

## **Slide 3: What is Verilookie?**

* A **web-based application** to detect deepfakes and AI-generated media
* Provides **image, video, and audio analysis**
* Educates users via **gamified learning** (quizzes, rewards, progress tracking)
* Prototype stage, focused on core functionality

---

## **Slide 4: Key Features**

* **Image & Video Analysis**: Upload media for authenticity check
* **Unified Detection Pipeline**: Combines multiple detectors for stronger results
* **AI-Powered Quizzes**: Personalized, adaptive learning
* **Gamification**: Badges, progress tracking, achievements

---

## **Slide 5: Significance in Cybersecurity**

* **Combating Misinformation** – verifying authenticity reduces fake news spread
* **Fraud Prevention** – detect manipulated media used for scams
* **Protecting Identity** – users check if their likeness is misused
* **Education** – gamification teaches detection skills

---

## **Slide 6: System Architecture (Prototype)**

**Frontend (React):**

* UploadBox – file uploads
* ResultCard – shows analysis results
* QuizPage – interactive quizzes
* Gamification UI – progress & achievements

**Backend (FastAPI):**

* API endpoints for analysis & quizzes
* Detection pipeline:

  * **Deepfake Detector** – ViT model
  * **AI-Generated Detector** – NVIDIA API
* Quiz Generator – powered by Gemini API
* Data stored in JSON files

---

## **Slide 7: AI/ML Models**

* **Deepfake Detection**: ViT model (Hugging Face)
* **AI-Generated Media**: NVIDIA AI API
* **Quiz Generation**: Google Gemini API

---

## **Slide 8: Prototype Limitations**

* Data stored in **JSON** (no database yet)
* User management = **single user\_id**
* Not built for **production scalability**
* Future work: database integration, authentication, cloud deployment

---

## **Slide 9: Conclusion**

* Verilookie = prototype tackling a **critical cybersecurity problem**
* Provides detection + **education through gamification**
* Strong foundation, with clear paths for **scaling and improvement**

---