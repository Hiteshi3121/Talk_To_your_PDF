# Integration of Documents, vector embeddings, vector store, RAG model 

#Phase1 imports
import os #importing os module to interact with the operating system
import warnings
import logging #to log the activities i.e. events that happen during the execution of a program
import streamlit as st

#Phase 2: imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3: imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

from langchain.vectorstores import Chroma
import pickle

warnings.filterwarnings("ignore")     #this will ignore all the warnings
logging.getLogger("transformers").setLevel(logging.ERROR)  #ignore only transformers warnings

st.title("AI Chat-bot (RAG)")
df = st.file_uploader(label="Upload your PDF file here", type=["pdf"]) #accept_multiple_files=True) #upload the PDF file from UI

if "messages" not in st.session_state:   #Keep the chat history in session state
    st.session_state.messages = []

for message in st.session_state.messages:   #Display the chat History
    st.chat_message(message['role']).markdown(message['content'])

# Phase 3 (Pre-requisite): Load PDF, create embeddings, vector store
@st.cache_resource
def get_vectorstore():
    #pdf_name = "" #hardcoded PDF file path
    pdf_name = df
    persist_dir = "./vectorstore_cache"
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    

    loader = PyPDFLoader(pdf_name)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=90)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    vectordb.persist()  # save to disk
    return vectordb

    
prompt = st.chat_input("Enter your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt}) #to append the user message to the session state
    
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
                                            the most accurate and most precise answers. Answer the following Question: {user_prompt}.
                                            Start the answer directly. No small talk please""")
    
    #model = "openai/gpt-oss-20b"
    model = "groq/compound-mini"
    #model = "llama-3.3-70b-versatile"
    groq_chat = ChatGroq(
        groq_api_key = "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",
        model_name = model
    )
#phase 3: RAG model - Retrieval Augmented Generation
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Vectorstore is not initialized. Please check the PDF file path and try again.")
        
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        result = chain({"query": prompt})
        response = result['result']
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role':'assistant', 'content':response})
    except Exception as e:
        st.error(f"An error occurred: {e}")
        response = "I'm sorry, but I couldn't process your request at the moment."
    

    if st.button("Clear the Cache memory & Restart"):
        st.cache_resource.clear()
        Chroma.reset()
        st.experimental_rerun() # Rerun to re-initialize Chroma from scratch