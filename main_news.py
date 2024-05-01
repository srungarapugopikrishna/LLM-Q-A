from langchain_helper_FAISS import (get_opneai_llm_object, get_open_ai_embeddings, load_urls,
                                    query_chain, split_text, get_faiss_vector_index_from_local,get_faiss_vector_index, save_faiss_vector_index_to_local, get_qa_chain)

import streamlit as st


st.title("QnA from Urls")

st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}::")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

if process_url_clicked:
    main_placeholder.text("Data Loading....Started.....")
    data = load_urls(urls)
    main_placeholder.text("Text Splitter....Started.....")
    docs = split_text(data)

    llm = get_opneai_llm_object()
    embeddings = get_open_ai_embeddings()
    vector_index = get_faiss_vector_index(docs, embeddings)
    save_faiss_vector_index_to_local(vector_index, "faiss_store")

query = main_placeholder.text_input("Question::")
if query:
    llm = get_opneai_llm_object()
    embeddings = get_open_ai_embeddings()
    vector_index = get_faiss_vector_index_from_local("faiss_store", embeddings)
    chain = get_qa_chain(llm, vector_index)
    result = query_chain(chain, query)
    st.header("Answer")
    st.write(result["answer"])

    sources = result.get("sources")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")
        for source in  sources_list:
            st.write(source)