import os

import PyPDF2
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


def textHandler(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = reader.pages

        for i in num_pages:
            text += i.extract_text()
    return clear_text(text)


def urlHandler(web_url):
    response = requests.get(web_url)
    if response.status_code == 200:
        text = response.text
        return clear_text(text)
    else:
        return f"fail to generate with status code {response.status_code}"


def clear_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text
