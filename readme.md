🚫 Full-Stack AI Spam Detection System

An end-to-end Machine Learning application designed to classify SMS/Messages as Spam or Not Spam. This project features a robust 3-tier architecture with a dedicated Backend API, a MySQL Database for persistence, and an interactive Frontend dashboard.

🚀 Key Features

AI-Powered Classification: Uses a Multinomial Naive Bayes model with TF-IDF vectorization for high-accuracy text detection.

Idempotent API Design: Built with FastAPI, the backend ensures no duplicate messages are stored in the database through custom normalization logic.

Data Sanitization: Implements Regex-based cleaning to handle redundant spaces, newlines, and hidden characters before processing.

Database Persistence: Integrated with MySQL via SQLAlchemy to permanently store prediction history with UTC timestamps.

Interactive UI: A modern Streamlit dashboard featuring:

Predict Tab: Real-time message analysis with detailed system feedback.

History Tab: Dynamic data fetching from MySQL to view past 15 predictions in a clean tabular format.

Security Focused: Sensitive credentials (like DB passwords) are managed using Environment Variables (.env).

🛠️ Tech Stack

Layer

Technology Used

Frontend

Streamlit, Pandas, Requests

Backend

FastAPI, Uvicorn, Pydantic, Dotenv

Database

MySQL, SQLAlchemy, PyMySQL

Machine Learning

Scikit-Learn, NLTK, Pickle

Language

Python 3.10+

📂 Project Structure

Spam_Detection/
├── app.py                # Streamlit Frontend (UI & History Retrieval)
├── main.py               # FastAPI Backend (ML Logic & Database ORM)
├── model.pkl             # Trained Naive Bayes Model (Binary)
├── vectorizer.pkl        # TF-IDF Vectorizer
├── .env                  # Environment Variables (DB Credentials)
├── .gitignore            # Files excluded from Version Control
├── requirements.txt      # Python Dependencies
└── README.md             # Project Documentation


⚙️ Installation & Setup

1. Clone the Repository

git clone [https://github.com/Utkarsh-Singh-30/spam_detection.git](https://github.com/Utkarsh-Singh-30/spam_detection.git)
cd spam-detection-system


2. Install Dependencies

pip install -r requirements.txt


3. Database Configuration

Open your MySQL server and create the database:

CREATE DATABASE spam_db;


Create a .env file in the root directory and add your credentials:

DB_PASSWORD=your_mysql_password_here


4. Running the Application

You need to start both the backend and frontend in separate terminals:

Terminal 1 (Backend - FastAPI):

uvicorn main:app --reload


Terminal 2 (Frontend - Streamlit):

streamlit run app.py


🧠 Model Pipeline

Text Preprocessing: Tokenization, Lowercasing, Stopword Removal, and Stemming using NLTK's PorterStemmer.

Vectorization: TF-IDF transformation to convert cleaned text into numeric feature vectors.

Prediction: Classification using the Multinomial Naive Bayes algorithm.

Database Logic: * Input text is normalized using RegEx.

System checks if the message exists in MySQL (Idempotency).

New messages are saved; existing ones are fetched from the database to save compute resources.