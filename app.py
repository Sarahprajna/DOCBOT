import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer
from chroma import chroma_database

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_text_chunks(text):
    chunk_size = 256
    chunk_overlap = 128

    chunked_texts = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        chunked_texts.extend(chunks)
    return chunks
def get_embeddings(chunked_texts):
    embeddings = model.encode_multi_process(chunked_texts)
    sentence_embeddings = []
    for chunk_idx in range(0, len(chunked_texts), chunk_overlap):
        chunk_embeddings = embeddings[chunk_idx:chunk_idx + chunk_overlap]
        sentence_embeddings.append(np.mean(chunk_embeddings, axis=0))
    return embeddings

def get_vectordb(embeddings):
    db = chroma.connect('localhost', 'chroma_database')
    # Create a table in the Chroma database to store the embeddings
    db.create_collection('embeddings', columns=['id', 'embedding'])
    for i, embedding in enumerate(embeddings):
        db.insert('embeddings', {
        'id': i,
        'embedding': embedding
    })
    return db


st.set_page_config(
    page_title="DOCBOT",
    page_icon="👾",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom#DC7C98,#FFF5EE);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("CHAT WITH DOCBOT 👾")

user_input = st.text_area("ASK YOUR QUERY....")
if st.button("SEND"):
    st.write("User's Query:", user_input)
if user_input:
    response = "DocBot At Your Service."
    st.write("DocBot:", response)

with st.sidebar:
    st.sidebar.title("DOCBOT 👾")
    pdfs = st.file_uploader("Upload Your Documents", accept_multiple_files=True)

    # Check if the PDFs are not None
    if pdfs is not None:
        for pdf in pdfs:
            file_extension = pdf.name.split(".")[-1].lower()
            if file_extension == "pdf":
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                uploaded_text = text
                st.write(uploaded_text)
                is_pdf_uploaded = True

    submit_button = st.sidebar.button("SUBMIT")
    if submit_button:
        st.sidebar.text("Files submitted and processed.")

    text_chunks = get_text_chunks(text)
    st.write(text_chunks)

    # Get the embeddings for the text chunks
    embeddings = get_embeddings(chunked_texts)
    st.write(embeddings)

    #TO STORE IN CHROMA VECTOR DATA BASE
    db = get_vectordb(embeddings)
   
   

previously_asked_queries = []
st.sidebar.markdown("## Previously Asked Queries")
