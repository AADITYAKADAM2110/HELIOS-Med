import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from preprocess import preprocess_text

def ingest_medical_data(data_dir="./data", persist_dir="./chroma_db"):
    """
    Reads PDFs from data_dir, chunks them, and saves to a local Vector DB.
    """

    embeddings = HuggingFaceEmbeddings(model = "all-MiniLM-L6-v2")

    documents = []

    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created {data_dir}. Please drop your PDFs there and run again.")
            return None
        
        for file in os.listdir(data_dir):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(data_dir, file))
                documents.extend(loader.load())

        if not documents:
            print("No PDFs found in the data directory.")
            return None
    
    except Exception as e:
        print(f"Error creating data directory: {e}")
        return None
    
    preprocess_texts = [preprocess_text(doc.page_content) for doc in documents]
    for i, doc in enumerate(documents):
        doc.page_content = preprocess_texts[i]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1300,
        chunk_overlap = 300,
        length_function = len,
        separators = ["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Indexing {len(chunks)} chunks into ChromaDB at {persist_dir}...")
    vector_db = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_dir
    )

    print("Ingestion Complete. Vector DB is ready.")
    return vector_db

if __name__ == "__main__":
    ingest_medical_data()