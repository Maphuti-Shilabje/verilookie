# Verilookie: A Deepfake and AI-Generated Media Detector

---

## What is Verilookie?

Verilookie is a web-based application designed to detect deepfakes and AI-generated images and videos. It provides users with a tool to analyze media files and determine their authenticity.

**Key Features:**

*   **Image and Video Analysis:** Upload media files for deepfake and AI-generation detection.
*   **Unified Detection Pipeline:** A multi-layered approach to analysis for more accurate results.
*   **Gamified Learning:** An integrated quiz and reward system to educate users about deepfakes and AI-generated content.
*   **Personalized Quizzes:** AI-powered quizzes that adapt to the user's knowledge level.

---

## Significance in Cybersecurity

The proliferation of deepfakes and AI-generated content poses a significant threat to individuals and society. Verilookie addresses these threats by:

*   **Combating Misinformation:** Helping users identify and verify the authenticity of media, thus reducing the spread of fake news and propaganda.
*   **Preventing Fraud:** Detecting manipulated media that could be used for financial fraud, identity theft, or blackmail.
*   **Protecting Digital Identity:** Empowering users to verify if their own image or likeness has been used in a deepfake.
*   **Educating Users:** The gamification features are not just for engagement; they actively teach users how to spot the signs of manipulated media, making them more resilient to this form of attack.

---

## System Architecture (Prototype)

Verilookie is a full-stack web application with a Python backend and a React frontend.

### Frontend (React)

*   **User Interface:** A simple and intuitive interface for uploading files and viewing results.
*   **Components:**
    *   `UploadBox`: Handles file selection and submission.
    *   `ResultCard`: Displays the analysis results.
    *   `QuizPage`: Presents the interactive quizzes.
    *   `Gamification`: Shows the user's progress, level, and achievements.
*   **Communication:** Interacts with the backend via REST APIs.

### Backend (Python - FastAPI)

*   **API Server:** A FastAPI server that exposes endpoints for file analysis and quizzes.
*   **Detection Pipeline:**
    1.  **Deepfake Detector:** A primary model (`prithivMLmods/Deep-Fake-Detector-v2-Model`) to analyze for deepfakes.
    2.  **AI-Generated Content Detector:** A secondary service (NVIDIA AI API) to check for AI-generated content if the primary model is inconclusive.
    3.  **Unified Analysis:** The results from both are combined to provide a single, unified result.
*   **Quiz Generator:**
    *   Uses the Gemini AI API to dynamically generate personalized quizzes based on user performance.
    *   Manages quiz data and user scores.
*   **Data Storage:** Uses JSON files for storing quiz and user performance data (a database would be used in a production environment).

### AI/ML Models

*   **Deepfake Detection:** A pre-trained Vision Transformer (ViT) model from Hugging Face.
*   **AI-Generated Content:** NVIDIA's AI-generated image detection API.
*   **Quiz Generation:** Google's Gemini API.

---

## Prototype Nature

This project is a prototype. The core functionalities are implemented, but some aspects are simplified for this demonstration:

*   **Data Storage:** Uses JSON files instead of a robust database.
*   **User Management:** A single `user_id` is used for demonstration purposes.
*   **Scalability:** The current setup is not designed for high-traffic, production use.

Future work would involve integrating a proper database, implementing a full user authentication system, and deploying the application in a scalable cloud environment.
