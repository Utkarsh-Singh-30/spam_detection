# SMS Spam Detection Project
  This project implements a machine learning model to classify SMS messages as either "ham" (legitimate) or "spam"). The project involves data cleaning, preprocessing, feature extraction, model training and evaluation, hyperparameter tuning, and model serialization for potential deployment.
  
# Project Description
  The goal of this project is to build an effective SMS spam detection system. By analyzing the text content of SMS messages, the model learns to distinguish between unwanted spam messages and legitimate ham        messages. This can be useful for filtering spam and improving the user experience of messaging applications.

# Setup
  To run this notebook and replicate the results, you will need to have the following libraries installed:
  
  pandas
  numpy
  matplotlib
  seaborn
  nltk
  scikit-learn
  You can install these libraries using pip:
  !pip install pandas numpy matplotlib seaborn nltk scikit-learn
  You will also need to download the necessary NLTK data:
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')

# Data
  The dataset used in this project is a collection of SMS messages labeled as either "ham" or "spam".
  
  Source: The data is loaded from a CSV file named spam.csv.
  Initial State: The raw data contains several columns, including the message text and the target label. It also includes unnecessary columns (Unnamed: 2, Unnamed: 3, Unnamed: 4) and duplicate entries.
  Cleaning: The data was cleaned by:
  Dropping the irrelevant Unnamed columns.
  Renaming the relevant columns to target and text for clarity.
  Removing duplicate SMS messages to ensure a unique dataset.
  Transformation: The target column was transformed using LabelEncoder to convert the categorical labels ("ham", "spam") into numerical values (0, 1).

# Methodology
  The project follows a standard machine learning pipeline for text classification:
  
  Data Loading and Initial Exploration: The data is loaded and a preliminary inspection is performed to understand its structure and identify issues like missing values and duplicates.
  Data Cleaning: Irrelevant columns and duplicate entries are removed.
  Text Preprocessing: A custom function transformed_text is applied to the SMS messages. This function performs:
  Lowercasing the text.
  Tokenization (splitting text into words).
  Removing non-alphanumeric characters.
  Removing common English stop words and punctuation.
  Stemming words to their root form using PorterStemmer.
  Feature Extraction: The preprocessed text data is converted into numerical features using TfidfVectorizer. This technique assigns a weight to each word based on its frequency in a document and across the entire dataset. The max_features parameter is set to 3000 to consider the most important words.
  Model Selection and Training: Three Naive Bayes models (Gaussian Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes) are trained on the processed data. Naive Bayes is chosen for its simplicity and effectiveness in text classification tasks.
  Model Evaluation: The trained models are evaluated using a test set (20% of the data). Performance is assessed using accuracy, precision, and confusion matrices.
  Hyperparameter Tuning: GridSearchCV and RandomizedSearchCV are used to optimize the alpha hyperparameter of the Multinomial Naive Bayes model to potentially improve its performance.

# Results
  The models were evaluated on the test set, and the following performance metrics were observed:
  
  Gaussian Naive Bayes:
  Accuracy: {{accuracy_score(y_test,y_pred1)*100:.2f}}%
  Precision: {{precision_score(y_test,y_pred1)*100:.2f}}%
  Multinomial Naive Bayes:
  Accuracy: {{accuracy_score(y_test,y_pred2)*100:.2f}}%
  Precision: {{precision_score(y_test,y_pred2)*100:.2f}}%
  Bernoulli Naive Bayes:
  Accuracy: {{accuracy_score(y_test,y_pred3)*100:.2f}}%
  Precision: {{precision_score(y_test,y_pred3)*100:.2f}}%
  The Bernoulli Naive Bayes model achieved the highest accuracy and precision on the test set.
  
  After hyperparameter tuning, the Multinomial Naive Bayes model with alpha = {{gs.best_params_['alpha']}} achieved a cross-validation score of {{gs.best_score_*100:.2f}}% on the training data. The performance on the test set with the tuned model was:
  
  Tuned Multinomial Naive Bayes:
  Accuracy: {{accuracy_score(y_test,y_predF)*100:.2f}}%
  Precision: {{precision_score(y_test,y_predF)*100:.2f}}%
  Deployment
  The trained TfidfVectorizer and the best-performing model (Multinomial Naive Bayes with tuned hyperparameters) are serialized using the pickle library. This allows for saving the trained components and loading them later for making predictions on new, unseen data without retraining.
  
  The requirements.txt file lists the necessary libraries with their versions, which is useful for recreating the environment for deployment.

# Files
  spam.csv: The dataset containing the SMS messages.
  vectorizer.pkl: The serialized TfidfVectorizer object.
  model.pkl: The serialized trained Multinomial Naive Bayes model.
  requirements.txt: A file listing the project dependencies.
  sms_spam_detection.ipynb: The Jupyter notebook containing the project code.
