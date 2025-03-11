#imports
import re
from bs4 import BeautifulSoup
import requests
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

#download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

#initialize tools
lemmatizer = WordNetLemmatizer()

#cleaning text
def clean_text(text):
    #remove html tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    #remove square brackets and numbers inside them
    text = re.sub(r'\[.*?\]', '', text)

    #normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    #tokenize
    tokens = word_tokenize(text)

    #lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) if word not in ["was", "has"] else word for word in tokens]
    #convert tokens back to string
    return ' '.join(tokens)


#remove duplicates
def remove_duplicates(data):
    return list(dict.fromkeys(data))  

def preprocess_data(scraped_data):
    cleaned_data = [clean_text(paragraph.strip()) for paragraph in scraped_data if paragraph.strip()]
    return remove_duplicates(cleaned_data)

#read scraped data
def read_scraped_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()  

#save clean data to new file
def save_cleaned_data(file_path, cleaned_data):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(cleaned_data))

#get current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

#set the files save in current directory
scraped_data_path = os.path.join(current_directory, "scrapedData.txt")
cleaned_data_path = os.path.join(current_directory, "cleanedData.txt")

#preprocess the data
scraped_data = read_scraped_data(scraped_data_path)
processed_data = preprocess_data(scraped_data)

#save the data in current directory
save_cleaned_data(cleaned_data_path, processed_data)
print(f"Cleaned data saved to {cleaned_data_path}")
