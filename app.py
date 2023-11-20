import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import chromadb
from langchain.document_loaders import DirectoryLoader
from streamlit_lottie import st_lottie
import requests

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
    model = AutoModelForSeq2SeqLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float32,
        from_tf=True
    )
    return base_model
def get_embeddings(pdfs):
    loaders = []
    loaders1 = ""
    pdf_loader1 = DirectoryLoader(loaders, glob="**/*.documents")
    for pdf in pdfs:
        loaders.append(pdf)


    # lets create document
    documents = []
    tests = []
    for loader in loaders:
        documents.append(loader.load())
    for doc in documents:
        print("splitting into chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts1 = text_splitter.split_documents(doc)
        tests += texts1
    # create embeddings here
    print("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    sent_embeddings = []

    for text in tests:
        # Generate embeddings
        sent_embeddings1 = embeddings.embed_documents(text)

        # Flatten the nested list
        sent_embeddings_flat = []
        for embedding in sent_embeddings1:
            for inner_embedding in embedding:
                sent_embeddings_flat.append(inner_embedding)

        sent_embeddings.append(sent_embeddings_flat)

    # Process the embeddings
    print(sent_embeddings)
    print(f"Creating embeddings. May take some minutes...")
    client = chromadb.Client()
    collection = client.create_collection("My_Collection")
    id = []
    for n in range(0, len(sent_embeddings)):
        x = f"id{n}"
        y = str(x)
        id.append(y)
    collection.add(
        embeddings=sent_embeddings,
        documents=tests,
        ids=id
    )

    return tests, collection,

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=load_model(),
        max_length=256,
        do_sample="True",
        temperature=0.5,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# creates a Question Answering (QA) model based on the LLM pipeline and the Chroma database.
def qa_llm():
    llm = llm_pipeline()
    # Connect to the ChromaDB database
    db = chromadb.connect()

    # Create a retriever object
    retriever = db.as_retriever()

    # Initialize the QA model
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa

#this function takes an instruction, loads a QA model,
# generates an answer using the model,
# extracts the answer from the model's response,
# and returns both the answer and the generated text.

def process_answer(instruction):
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
if st.button("SEND"):
    st.info("Your Question" + user_input)
    st.info("Your Answer:")
    instruction = user_input
    if user_input:
        answer, metadata = process_answer(instruction)
        st.write("DocBot:", answer)
        st.write("DocBot:", metadata)

# Display the title and sidebar

with st.sidebar:
    st.sidebar.title("DOCBOT 👾",)
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

    if submit_button :
        vector_store = get_embeddings(pdfs)
        st.sidebar.text("Files submitted and processed.")

previously_asked_queries = []

st.sidebar.markdown("## Previously Asked Queries")

