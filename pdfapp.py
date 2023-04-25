import streamlit as st
import openai
import pdfplumber
from io import BytesIO
from transformers import pipeline

openai.api_key = st.secrets.OpenAIAPI.openai_api_key

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def answer_question(question, context):
    response = openai.Completion.create(engine="gpt-3.5-turbo", prompt=f"{question}\n{context}\nAnswer:", max_tokens=50)
    answer = response.choices[0].text.strip()
    return answer

st.title("PDF 要約 and Q&A Powered by GPT")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    text = read_pdf(uploaded_file)
    summary = summarize_text(text)
    st.write("Summary:", summary)
    question = st.text_input("Ask a question based on the summary:")
    if question:
        answer = answer_question(question, summary)
        st.write("Answer:", answer)