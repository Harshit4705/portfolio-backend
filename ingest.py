import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Configuration
RESUME_PATH = r"..\resume.pdf" # Path relative to backend/
CHROMA_PATH = r"chroma_db"

def ingest_resume():
    if not os.path.exists(RESUME_PATH):
        print(f"Error: Resume not found at {RESUME_PATH}")
        return

    print("Loading resume...")
    loader = PyPDFLoader(RESUME_PATH)
    docs = loader.load()
    
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    print("Embedding and storing in ChromaDB...")
    # Using FastEmbed for free, local, good quality embeddings.
    # Alternatives: OpenAIEmbeddings, HuggingFaceEmbeddings
    embedding_function = FastEmbedEmbeddings() 
    
    if os.path.exists(CHROMA_PATH):
        print("Clearing existing vector store...")
        import shutil
        shutil.rmtree(CHROMA_PATH)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH,
    )
    print(f"Successfully toasted resume into {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_resume()
