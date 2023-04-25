import os
import openai
import streamlit as st
import langchain

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain import PromptTemplate

os.environ["OPENAI_API_KEY"] = st.secrets.OpenAIAPI.openai_api_key

def load_pdf(file):
    loader = PyPDFLoader(file)
    documents = loader.load_and_split()

def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embeddings)
    return vectordb

def ask_question(vectordb, question):
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=vectordb.as_retriever())
    prompt = PromptTemplate(
        input_variables=["question"],
        template="あなたは親切なアシスタントです。下記の質問に日本語で回答してください。\n質問：{question}\n回答：",
    )
    answer = qa.ask(prompt.fill({"question": question}))
    return answer

def generate_summary(documents, language):
    if language == 'Japanese':
        summary_prompt = "この文書の要約を提供してください：\n\n{text}\n\n要約："
    elif language == 'English':
        summary_prompt = "Please provide a summary of this document:\n\n{text}\n\nSummary:"

    summaries = []
    for doc in documents:
        summary = ask_question(vectordb, summary_prompt.format(text=doc))
        summaries.append(summary)
    return '\n'.join(summaries)

st.title('PDF Summary and Q&A')
uploaded_file = st.file_uploader('Upload a PDF file', type=['pdf'])

if uploaded_file is not None:
    with st.spinner('Processing PDF...'):
        documents = load_pdf(uploaded_file)
        vectordb = process_documents(documents)
    st.success('PDF processed.')

    summary_language = st.selectbox("Select summary language:", ['Japanese', 'English'])

    with st.spinner('Generating summary...'):
        summary = generate_summary(documents, summary_language)
    st.subheader('Summary:')
    st.write(summary)

    question = st.text_input('Ask a question about the document:')
    if question:
        with st.spinner('Generating answer...'):
            answer = ask_question(vectordb, question)
        st.write(answer)

