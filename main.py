from langchain_helper import get_conversation_chain

import streamlit as st

st.title("Enter Url:")
url = st.text_input("url: ")

if url:
    st.title("Ask your queries ")

    question = st.text_input("Question: ")

    if question:
        chain = get_conversation_chain(url)
        answer = chain.invoke(question)
        st.header("Answer:")
        st.write(answer)

