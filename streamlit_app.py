import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

# Streamlit app
st.title('DOChain')

# Get OpenAI API key and source document input
openai_api_key = os.environ['OPENAI_API_KEY']
source_doc = st.file_uploader("Upload Source Document", type="pdf")
question = st.text_input("Question")

# Check if the 'Answer' button is clicked
if st.button("Answer"):
    # Validate inputs
    if not openai_api_key.strip() or not source_doc:
        st.error(f"Please provide the missing fields.")
    else:
        try:
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            for i in range(3):
                pages.append(Document(page_content="null", metadata={}))
            os.remove(tmp_file.name)
            
            # Create embeddings for the pages and insert into Chroma database
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectordb = Chroma.from_documents(pages, embeddings)

            # Initialize the OpenAI module, load and run the QA chain
            llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-4")
            search = vectordb.similarity_search(" ")
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            answer = qa_chain.run(input_documents=search, question=question)
            st.success(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
