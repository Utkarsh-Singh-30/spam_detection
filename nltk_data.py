import nltk
import os

# Define a local directory to save NLTK data
local_nltk_data_dir = './nltk_data' # This will create the folder in your current directory

# Create the directory if it doesn't exist
os.makedirs(local_nltk_data_dir, exist_ok=True)

# Download the required NLTK data to the specified directory
print(f"Downloading NLTK data to: {os.path.abspath(local_nltk_data_dir)}")
nltk.download('stopwords', download_dir=local_nltk_data_dir)
nltk.download('punkt', download_dir=local_nltk_data_dir)
nltk.download('punkt_tab', download_dir=local_nltk_data_dir)

print("NLTK data download complete.")
