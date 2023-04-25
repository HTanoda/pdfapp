import streamlit as st
import openai
import pdfreader
from io import BytesIO
from summarizer import Summarizer
from transformers import pipeline

openai.api_key = st.secrets.OpenAIAPI.openai_api_key

def read_pdf(file):
    text = ""
    file_buffer = BytesIO(file.getvalue())
    pdf = pdfreader.SimplePDFViewer(file_buffer)
    num_pages = len(list(pdfreader.SimplePDFViewer(file_buffer).get_pages()))
    for page in range(num_pages):
        pdf.navigate(page + 1)
        pdf.render()
        text += " ".join(pdf.canvas.strings)
    return text

def summarize_text(text):
    summarizer = Summarizer()
    summary = summarizer(text)
    return summary

def answer_question(question, context):
    response = openai.Completion.create(engine="gpt-3.5-turbo", prompt=f"{question}\n{context}\nAnswer:", max_tokens=50)
    answer = response.choices[0].text.strip()
    return answer

st.title("PDF 要約 and Q&A")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    text = read_pdf(uploaded_file)
    summary = summarize_text(text)
    st.write("Summary:", summary)
    question = st.text_input("Ask a question based on the summary:")
    if question:
        answer = answer_question(question, summary)
        st.write("Answer:", answer)