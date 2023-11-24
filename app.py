import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch
from langchain.llms import CTransformers
import chromadb
from chromadb.config import Settings


device = torch.device('cuda')


custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


def chroma_settings():
    settings = Settings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        persist_directory='db',
        chroma_server_host='localhost',
        chroma_server_http_port='8000',
        anonymized_telemetry=False
    )
    return settings


def load_model():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm


def textsplitter():
    # Pass uploaded_text as an argument to the function
    global uploaded_text
    # Convert each item in uploaded_text to a Document object
    texts = []
    for txt in uploaded_text:
        doc = Document(page_content=txt)
        texts.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    tests = text_splitter.split_documents(texts)
    return tests


def embeddings():
    # create embeddings here
    sent_embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
    return sent_embeddings


def database():

    settings = chroma_settings()
    tests = textsplitter()
    sent_embeddings = embeddings()
    client = chromadb.HttpClient(host='localhost', port='8000')
    cdb = Chroma.from_documents(tests, sent_embeddings, persist_directory="db", client_settings=settings, client=client)
    cdb.persist()
    cdb = None
    return cdb


def vector_database():
    database()
    sent_embeddings = embeddings()
    vector_db = Chroma(persist_directory='db', embedding_function=sent_embeddings)
    return vector_db


# creates a Question Answering (QA) model based on the LLM pipeline and the Chroma database.
@st.cache_resource
def qa_llm():
    qa_prompt = set_custom_prompt()
    embedding = embeddings()
    llm = load_model()
    settings = chroma_settings()
    db = Chroma(persist_directory="db", embedding_function=embedding, client_settings=settings)
    qa = RetrievalQA.from_llm(llm, qa_prompt, db)
    return qa


def process_answer():
    global instruction
    global answer
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


st.set_page_config(
    page_title="DOCBOT",
    page_icon="👾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
    <style>
    .sidebar.fixed {{
        background-image: linear-gradient(to right, #8E236A, #4A148C);
    }}
    </style>
    """, unsafe_allow_html=True)
with st.expander("About Docbot"):
    st.markdown(
        """
        DOCBOT can read and understand any type of document including Pdfs,Word Documents and many more.
        DOCBOT is still under development this is just a demo of communications with multiple Documents.
        """
    )

st.header("CHAT WITH DOCBOT 👾")
user_input = st.text_area("ASK YOUR QUERY....")
st.write("Query:" + user_input)
st.write("DocBot:")


def handle_send_button_click():
    if not user_input:
        st.error("Please enter a query to proceed.")
    return


instruction = user_input
if user_input:
    answer, metadata = process_answer()
    st.write("DocBot:", answer)
    st.write("DocBot:", metadata)
if st.button("SEND"):
    handle_send_button_click()

with st.sidebar:
    st.sidebar.title("DOCBOT 👾",)
    pdfs = st.file_uploader("Upload Your Documents", accept_multiple_files=True)

    # Check if the PDFs are not None
    if pdfs is not None:
        for pdf in pdfs:
            file_extension = pdf.name.split(".")[-1].lower()
            if file_extension == "pdf":
                pdf_reader = PdfReader(pdf)
                document = []
                for page in pdf_reader.pages:
                    document += page.extract_text()
                uploaded_text = document
                is_pdf_uploaded = True
    submit_button = st.sidebar.button("SUBMIT")

    if submit_button:
        vector_database()
        # Pass the uploaded_text variable as an argument
        st.sidebar.text("Files submitted and processed.")

previously_asked_queries = []

st.sidebar.markdown("## Previously Asked Queries")


