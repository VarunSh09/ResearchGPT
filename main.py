import os

import streamlit as st
import time
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv() #take enironment variables from .env.

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

import tempfile

st.sidebar.header("📂 Sources")
st.sidebar.title("📄 Upload Sources")
st.sidebar.markdown("Add URLs or upload a PDF, then click **Process**.")

st.set_page_config(
    page_title="ResearchGPT",
    page_icon="📚",
    layout="wide"
)

st.title("📚 ResearchGPT")
st.caption("Ask questions from News URLs and PDF documents using AI")
st.sidebar.header("Knowledge Sources")

if "messages" not in st.session_state:
    st.session_state.messages = []


urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

all_docs = []

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


valid_url = [u for u in urls if u.strip()]


process_url_clicked =  st.sidebar.button("🚀 Process")
#file_path = "FAISS_index.pkl"

if not valid_url or  uploaded_pdf is None:
    st.sidebar.markdown("Please enter valid url at least one or Upload any pdf...")
    
elif valid_url or uploaded_pdf:
     st.sidebar.markdown("✅ Ready to process! Click the button below.")




main_placeholder = st.empty()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2,max_tokens=300)

if process_url_clicked :
    with st.spinner("Processing documents..."):  
        try:
            if urls:
                loader = UnstructuredURLLoader(urls=urls)
                url_docs = loader.load()
                all_docs.extend(url_docs)
                st.success("URLs loaded successfully!")
         except Exception as e:
             st.error(f"Failed to load URL!")
            


        if uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                temp_pdf_path = tmp.name

            pdf_loader = PyPDFLoader(temp_pdf_path)
            pdf_docs = pdf_loader.load()

            all_docs.extend(pdf_docs)

        if not all_docs:
            st.error("Please enter a URL or upload a PDF before processing.")
        
        main_placeholder.info("Loading data from URLs...")
        time.sleep(2)
        main_placeholder.info("Loading data from PDF...")

        # Split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
                                        separators= ['\n\n','\n','.',',',' '],
                                        chunk_size=1000,
                                        chunk_overlap=200)
        main_placeholder.info("Splitting data into chunks...")
        docs = text_splitter.split_documents(all_docs)
        #create embeddings and save it into to FAISS index
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


        #FAISS_index = FAISS.from_documents(docs, embeddings) 
        vectorstore = FAISS.from_documents(docs, embeddings)

        main_placeholder.info("Embedding vector started building...")
        vectorstore.save_local("faiss_index")
        time.sleep(2)
        main_placeholder.empty()
        st.success("✅ Documents processed successfully! You can now ask questions.")

        
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])      
    

if valid_url or uploaded_pdf:
    query = st.chat_input("Ask anything about the uploaded content...")
else:
    query = st.chat_input("Please enter a valid URL or upload a PDF to ask questions about the content...",disabled=True)

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })



if query:
   
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            vectorstore = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)

            chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(),return_source_documents=True)
            with st.spinner("Generating answer..."):
                try: 
                    result = chain({"question": query},return_only_outputs=True)
                    sources = result.get("sources"," ")
                    st.session_state.messages.append({
                                "role": "assistant",
                                "content": result["answer"]})                  
                    
                    with st.chat_message("user"):
                        st.write(query)
          
                    with st.chat_message("assistant"):
                        st.subheader("🤖 Answer")
                    
                        st.write(result["answer"])
                        st.write("Sources:")
                        if valid_url:
                            st.subheader("🔗 URLs")
                            for url in valid_url:
                                st.write(url)
                        if uploaded_pdf:
                            st.subheader("📄 Uploaded PDF")
                            st.write(uploaded_pdf.name)
                except Exception as e:
                        st.error("Sorry for inconvience. Problem generating answer. Please try later.")
                        error_message = str(e)
                        if "rate_limit" in error_message.lower() or "429" in error_message:
                            st.error("⚠️ Groq API rate limit reached. Please wait a few minutes or use a different API key.")
                        else:
                            st.error(f"⚠️ Error: {error_message}")
            #st.header("Answer:")
            #st.write(result["answer"])
            
          
           # if sources:
              #  st.subheader("Sources:")
             #   sources_list = sources.split("\n")
             #   for source in sources_list:
             #       st.write(source)
            
            
                    

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
