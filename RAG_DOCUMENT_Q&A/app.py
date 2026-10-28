import os 
from dotenv import load_dotenv 
import streamlit as st 
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextsplitter
from langchain_classic.chains.combine_documents import (
    create_stuff_document_chain
)
from langchain_classic.chains import create_retrieval_chain 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY')
model=ChatGroq(model='openai/gpt-oss-20b',groq_api_key=groq_api_key)

embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

prompt=ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only.
please provide the most accurate response based on the question
{context}
<context>
Question:{input}
"""
)

def create_vector_embeddings():
    if "vecotors" not in st.sesstion_state:
        st.session_state.embeddings=HuggingFaceEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader()
        st.session_state.docs=st.session_state.loader().load()
        st.session_state.text_splitter=RecursiveCharacterTextsplitter(chunk_size=100,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session.final_documents,st.session_state.embeddings)

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(model,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrival_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrival_chain.invoke({'input':user_prompt})
    print(f'Response time:{time.process_time()-start}')

    st.write(response['answer'])

    # with a streamit expander 
    with st.expander('Document Similarity Search'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('--------------------------')