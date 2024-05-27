import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def calculatesentiment(text):
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Fetching labels
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)

    # Get the highest scoring sentiment
    top_score_index = np.argmax(scores)
    top_label = labels[top_score_index]

    return top_label


# Streamlit App
st.title("Sentiment Analysis")
temp = """
     <div style="background-color:tomato; padding:10px">
     <h2 style="color:white; text-align:center;"> Real Time Sentiment Analysis</h2>
     </div>
"""
st.markdown(temp, unsafe_allow_html=True)
text = st.text_input("Text", "")
if st.button("predict"):
    result = calculatesentiment(text)
    st.success(f"Sentiment: {result}")
