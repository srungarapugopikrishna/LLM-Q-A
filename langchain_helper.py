import os

import pinecone
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI

load_dotenv()
# source_url = "https://hatkestory.com/about-us/"


def load_from_url(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data


def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True, )
    texts = text_splitter.split_documents(data)
    return texts


def get_conversation_chain(url):
    print("URL::::::::::::::::::")
    print(url)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    pc = pinecone.Pinecone(
        api_key=os.environ['PINECONE_API_KEY'],
        environment='us-east-1'
    )
    index_name = pc.Index('index-1')
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    data = load_from_url(url=url)
    texts = split_text(data)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name='index-1')
    retriever = vectordb.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conv_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    return conv_chain


# if __name__ == "__main__":
#     chain = get_conversation_chain(source_url)
#     query = 'where did gopi met rohit first?'
#     print(chain.invoke({'question': query}))
