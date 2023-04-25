import streamlit as st
import openai
import PyPDF2
from summarizer import Summarizer
from transformers import pipeline

openai.api_key = st.secrets.OpenAIAPI.openai_api_key

def read_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page).extractText()
    return text

def summarize_text(text):
    summarizer = Summarizer()
    summary = summarizer(text)
    return summary

def answer_question(question, context):
    response = openai.Completion.create(engine="text-davinci-002", prompt=f"{question}\n{context}\nAnswer:", max_tokens=50)
    answer = response.choices[0].text.strip()
    return answer

st.title("PDF Summarizer and Q&A")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    text = read_pdf(uploaded_file)
    summary = summarize_text(text)
    st.write("Summary:", summary)
    question = st.text_input("Ask a question based on the summary:")
    if question:
        answer = answer_question(question, summary)
        st.write("Answer:", answer)
