# Verilookie

Verilookie is a deepfake and scam detector with a gamified twist. Upload a file to see if it's real or fake, and earn XP to level up and unlock badges!

---

## 🚀 Setup

### Backend

1.  Navigate to the `src/backend` directory:
    ```bash
    cd src/backend
    ```
2.  Install the required Python packages:
    ```bash
    pip install fastapi "uvicorn[standard]" python-multipart
    ```

### Frontend

1.  Navigate to the `src/frontend` directory:
    ```bash
    cd src/frontend
    ```
2.  Install the required npm packages:
    ```bash
    npm install
    ```

---

## 🏃‍♂️ How to Run the Demo

1.  **Start the backend server:**

    In the `src/backend` directory, run:
    ```bash
    uvicorn main:app --reload
    ```
    The backend will be running at `http://localhost:8000`.

2.  **Start the frontend application:**

    In a new terminal, navigate to the `src/frontend` directory and run:
    ```bash
    npm start
    ```
    The frontend will open in your browser at `http://localhost:3000`.

3.  **Use the application:**

    -   Upload a file using the upload box.
    -   See the detection result.
    -   Watch your XP and level increase.
    -   Take the scam awareness quiz.

---

## 📝 Limitations & Future Work

-   **Mocked Detection:** The current version uses mocked detection and does not have a real deepfake detection model integrated.
-   **Limited File Types:** The backend is set up to handle file uploads, but the detection is not specific to any file type.

Future work could include:

-   Integrating a real deepfake detection model.
-   Adding more quiz questions.
-   Expanding the gamification features with more badges and a leaderboard.
-   Supporting more file types for detection.