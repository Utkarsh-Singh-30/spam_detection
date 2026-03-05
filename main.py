# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# from sqlalchemy import create_engine, Column, Integer, String, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from datetime import datetime
# import urllib.parse

# import re

# app=FastAPI()
# ps=PorterStemmer()

# password="Utkarsh@30"
# # 2. Encode the password so special characters like '@' are safe
# safe_password = urllib.parse.quote_plus(password)
# DATABASE_URL=f"mysql+pymysql://root:{safe_password}@localhost:3306/spam_db"


# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # --- 2. THE DATABASE TABLE MODEL ---
# class PredictionRecord(Base):
#     __tablename__ = "predictions"
    
#     id = Column(Integer, primary_key=True, index=True)
#     message = Column(String(500))  # Saving the original message
#     result = Column(String(50))    # Saving 'Spam' or 'Not Spam'
#     timestamp = Column(DateTime, default=datetime.utcnow)



# # This line creates the table in your MySQL automatically
# Base.metadata.create_all(bind=engine)


# try:
#     stopwords.words('english')
#     var = string.punctuation
#     nltk.word_tokenize('')
# except LookupError:
#     nltk.download('stopwords')
#     nltk.download('punkt')
#     nltk.download('punkt_tab')


# def transform_text(text):
#     text=text.lower()
#     text=nltk.word_tokenize(text)

#     y=[]

#     for i in text:
#         if i.isalnum():
#             y.append(i)
    
#     text=y[:]
#     y.clear()


#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#     text=y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)




# tfidf=pickle.load(open("vectorizer.pkl", 'rb'))
# model=pickle.load(open("model.pkl", 'rb'))



# class SpamRequest(BaseModel):
#     text:str

# # --- 4. THE API ROUTE (Updated to save to DB) ---
# # @app.post("/predict")
# # def predict(request: SpamRequest):
# #     # A. The Prediction Logic
# #     transformed_sms = transform_text(request.text)
# #     vector_input = tfidf.transform([transformed_sms])
# #     result = model.predict(vector_input)[0]
# #     label = "Spam" if result == 1 else "Not Spam"

# #     # B. The Database Logic (Saving the data)
# #     db = SessionLocal()
# #     try:
# #         new_entry = PredictionRecord(message=request.text, result=label)
# #         db.add(new_entry)
# #         db.commit()
# #         db.refresh(new_entry)
# #     finally:
# #         db.close() # Always close to prevent MySQL 'too many connections' error
    
# #     return {"prediction": label}

# # @app.post("/predict")
# # def predict(request: SpamRequest):
# #     db = SessionLocal()
    
# #     # 1. CHECK IF MESSAGE ALREADY EXISTS
# #     existing = db.query(PredictionRecord).filter(PredictionRecord.message == request.text).first()
    
# #     # 2. IF IT EXISTS, JUST RETURN THE OLD RESULT (Don't save again)
# #     if existing:
# #         db.close()
# #         return {"prediction": existing.result, "status": "Fetched from Database"}

# #     # 3. IF NEW, PROCEED WITH PREDICTION
# #     transformed_sms = transform_text(request.text)
# #     vector_input = tfidf.transform([transformed_sms])
# #     result = model.predict(vector_input)[0]
# #     label = "Spam" if result == 1 else "Not Spam"

# #     # 4. SAVE NEW RECORD
# #     try:
# #         new_entry = PredictionRecord(message=request.text, result=label)
# #         db.add(new_entry)
# #         db.commit()
# #     finally:
# #         db.close()
    
# #     return {"prediction": label, "status": "New Prediction Saved"}


# @app.post("/predict")
# def predict(request: SpamRequest):
#     db = SessionLocal()
#     try:
#         # 1. DATA SANITIZATION (The 'SDET' way)
#         # re.sub(r'\s+', ' ', ...) replaces tabs, newlines, and multiple spaces with one single space
#         # .strip() removes spaces from the very beginning and very end
#         user_msg = re.sub(r'\s+', ' ', request.text).strip()

#         # 2. VALIDATION: Don't process empty strings
#         if not user_msg:
#             return {"prediction": "N/A", "status": "Error: Empty message"}

#         # 3. DUPLICATE CHECK (Idempotency)
#         # Check if this exact normalized message already exists in MySQL
#         existing_record = db.query(PredictionRecord).filter(PredictionRecord.message == user_msg).first()
        
#         if existing_record:
#             # If found, return the saved result immediately without running the AI model
#             return {
#                 "prediction": existing_record.result, 
#                 "status": "Fetched from Database (Duplicate Ignored)"
#             }

#         # 4. AI MODEL PREDICTION (If message is new)
#         # Step A: Preprocess (Lowercase, Tokenize, Stem)
#         transformed_sms = transform_text(user_msg)
        
#         # Step B: Vectorize (TF-IDF)
#         vector_input = tfidf.transform([transformed_sms])
        
#         # Step C: Predict (MultinomialNB)
#         prediction_result = model.predict(vector_input)[0]
#         label = "Spam" if prediction_result == 1 else "Not Spam"

#         # 5. PERSISTENCE (Saving to MySQL)
#         new_entry = PredictionRecord(
#             message=user_msg, 
#             result=label, 
#             timestamp=datetime.utcnow()
#         )
#         db.add(new_entry)
#         db.commit()
#         db.refresh(new_entry) # Gets the auto-generated ID back
        
#         return {
#             "prediction": label, 
#             "status": "New Prediction Saved to Database"
#         }
        
#     except Exception as e:
#         # Log the error and return a safe message
#         print(f"Error occurred: {e}")
#         return {"prediction": "Error", "status": f"System Failure: {str(e)}"}
        
#     finally:
#         # ALWAYS close the connection to prevent 'Too many connections' error in MySQL
#         db.close()

# # Add this new route at the bottom of main.py
# @app.get("/history")
# def get_history():
#     db = SessionLocal()
#     try:
#         # 1. Query the database for the last 10 records
#         # .desc() means newest messages come first
#         records = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(10).all()
        
#         # 2. Convert the database objects into a simple list of dictionaries
#         history = []
#         for rec in records:
#             history.append({
#                 "id": rec.id,
#                 "message": rec.message,
#                 "result": rec.result,
#                 "timestamp": rec.timestamp.strftime("%Y-%m-%d %H:%M:%S")
#             })
#         return history
#     finally:
#         db.close()





import pickle
import string
import nltk
import re
import urllib.parse
import os
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# 1. Load Environment Variables (Security)
load_dotenv()
raw_password = os.getenv("DB_PASSWORD")

if not raw_password:
    print("⚠️ WARNING: DB_PASSWORD not found in .env file!")

# --- DATABASE SETUP ---
safe_password = urllib.parse.quote_plus(raw_password) if raw_password else ""
DATABASE_URL = f"mysql+pymysql://root:{safe_password}@localhost:3306/spam_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    message = Column(String(500))
    result = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- APP & ML SETUP ---
app = FastAPI()
ps = PorterStemmer()

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join([ps.stem(i) for i in text])

tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

class SpamRequest(BaseModel):
    text: str

# --- ROUTES ---

@app.post("/predict")
def predict(request: SpamRequest):
    db = SessionLocal()
    try:
        # Data Normalization (Handling extra spaces)
        user_msg = re.sub(r'\s+', ' ', request.text).strip()
        
        if not user_msg:
            return {"prediction": "N/A", "status": "Empty message"}

        # Idempotency Check (Duplicate Check)
        existing = db.query(PredictionRecord).filter(PredictionRecord.message == user_msg).first()
        if existing:
            return {"prediction": existing.result, "status": "Fetched from DB (Duplicate)"}

        # ML Prediction
        transformed_sms = transform_text(user_msg)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        label = "Spam" if result == 1 else "Not Spam"

        # Save to DB
        new_entry = PredictionRecord(message=user_msg, result=label)
        db.add(new_entry)
        db.commit()
        
        return {"prediction": label, "status": "New record saved"}
    finally:
        db.close()

@app.get("/history")
def get_history():
    db = SessionLocal()
    try:
        records = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(15).all()
        return [
            {
                "id": r.id, 
                "message": r.message, 
                "result": r.result, 
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M")
            } for r in records
        ]
    finally:
        db.close()