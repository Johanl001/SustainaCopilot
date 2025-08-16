import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the paths for your documents and the vector store
DOC_PATH = "docs"
CHROMA_PATH = "chroma_db"

def ingest_documents():
    """
    Loads, splits, and embeds documents from the 'docs' folder into a ChromaDB vector store.
    """
    documents = []
    for filename in os.listdir(DOC_PATH):
        file_path = os.path.join(DOC_PATH, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        elif filename.load()
            documents.extend(loader.load())
    
    if not documents:
        print("No documents found to ingest. Please add files to the 'docs' folder.")
        return

    print(f"Loaded {len(documents)} document pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Use a local, open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared existing data from {CHROMA_PATH}.")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Successfully created a new ChromaDB instance at {CHROMA_PATH} with {len(chunks)} chunks.")
    
    db.persist()

if __name__ == "__main__":
    ingest_documents()