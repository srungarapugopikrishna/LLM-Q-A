import os
import streamlit as st
import pickle
import time
import langchain
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


def get_opneai_llm_object(temperature=0.9, max_tokens=500):
    llm = OpenAI(temperature=temperature, max_tokens=max_tokens)
    return llm


def load_urls(urls_list):
    loader = UnstructuredURLLoader(urls=urls_list)
    data = loader.load()
    return data


def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,)
    docs = text_splitter.split_documents(data)
    return docs


def get_open_ai_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    return embeddings


def get_faiss_vector_index(docs, embeddings):
    vector_index = FAISS.from_documents(docs, embeddings)
    return vector_index


def save_faiss_vector_index_to_local(vectorindex, destination_path):
    vectorindex.save_local(destination_path)


def get_faiss_vector_index_from_local(source_path, embeddings):
    vector_index = FAISS.load_local(source_path, embeddings, allow_dangerous_deserialization=True)
    return vector_index


def get_qa_chain(llm, vector_index):
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
    return chain


def query_chain(chain, query):
    chain_response = chain.invoke({"question": query}, return_only_outputs=True)
    return chain_response


def run(urls=None, query=None):
    # if not urls:
    #     urls = [
    #         "https://www.moneycontrol.com/news/technology/accentures-tepid-outlook-implies-more-pain-for-indian-it-even-in-fy25-12508381.html",
    #         "https://www.moneycontrol.com/news/technology/infosys-gives-revenue-guidance-of-1-3-for-fy25-12666111.html"]
    llm = get_opneai_llm_object()
    embeddings = get_open_ai_embeddings()

    # vector_index = get_faiss_vector_index_from_local("faiss_store", embeddings)

    data = load_urls(urls)
    docs = split_text(data)
    vector_index = get_faiss_vector_index(docs, embeddings)
    # save_faiss_vector_index_to_local(vector_index, "faiss_store")

    chain = get_qa_chain(llm, vector_index)
    # query = input("Questions::")
    # langchain.debug = True
    if not query:
        return "Ask some question"
    response = query_chain(chain, query)
    return response

# print(run())