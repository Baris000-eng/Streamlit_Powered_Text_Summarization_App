import streamlit as st
from transformers import pipeline, AutoTokenizer
import pdfplumber
import docx
import re
import time

MODEL_OPTIONS = {
    "BART": "facebook/bart-large-cnn",
    "T5": "t5-small",
    "DistilBART": "sshleifer/distilbart-cnn-12-6"
}

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def read_txt(file):
    return file.read().decode("utf-8")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    text = ''.join(c for c in text if c.isprintable())
    return text

def chunk_text_by_tokens(text, tokenizer, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def recursive_summarize(text, summarizer, tokenizer, max_tokens=512, min_len=30, max_len=150):
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=max_tokens)
    summaries = []
    for i, chunk in enumerate(chunks):
        st.write(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
        summaries.append(summary[0]["summary_text"])
        time.sleep(0.1)

    combined_summary = " ".join(summaries)
    combined_tokens = tokenizer.tokenize(combined_summary)
    if len(combined_tokens) > max_tokens:
        st.write("Summarizing combined summaries recursively...")
        return recursive_summarize(combined_summary, summarizer, tokenizer, max_tokens, min_len, max_len)
    else:
        return combined_summary


# Streamlit UI
st.set_page_config(page_title="Smart Summarizer", layout="wide")
st.title("📄 AI-Powered Text Summarizer")
st.markdown("Paste text or upload a file, then click Summarize.")

input_type = st.radio("Select input method:", ["Paste Text", "Upload File"], horizontal=True)

text = ""
if input_type == "Paste Text":
    text = st.text_area("Enter or paste your text here:", height=250)
else:
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            text = read_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            text = read_docx(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            text = read_txt(uploaded_file)

st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose model:", list(MODEL_OPTIONS.keys()))
min_len = st.sidebar.slider("Min summary length", 10, 100, 30, step=10)
max_len = st.sidebar.slider("Max summary length", 50, 500, 150, step=10)

@st.cache_resource
def load_model_and_tokenizer(model_name):
    summarizer = pipeline("summarization", model=MODEL_OPTIONS[model_name])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OPTIONS[model_name])
    return summarizer, tokenizer

if st.button("✨ Summarize"):
    if not text.strip():
        st.warning("⚠️ Please enter or upload some text first.")
    else:
        summarizer, tokenizer = load_model_and_tokenizer(model_choice)
        st.info("Summarizing, please wait...")
        summary = recursive_summarize(text, summarizer, tokenizer,
                                      max_tokens=tokenizer.model_max_length,
                                      min_len=min_len, max_len=max_len)
        summary = clean_text(summary)
        st.subheader("📌 Summary")
        st.text_area("Summary:", value=summary, height=300)
        st.download_button("📥 Download Summary", summary, file_name="summary.txt", mime="text/plain")
