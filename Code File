pip install -r requirements.txt
python text_summarization.py

### Task 1: Text Summarization Tool
# This script summarizes a given text using the Hugging Face transformers pipeline.

import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# Download necessary NLTK data
nltk.download('punkt')

def summarize_text(text, max_length=150):
    """Summarizes the given text using a pre-trained transformer model."""
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    text = """Paste your lengthy article here for summarization."""
    summary = summarize_text(text)
    print("Summary:\n", summary)
