import os
import openai

# Env import
from dotenv import load_dotenv

# PDF utlity
from PyPDF2 import PdfReader

# Langchain imports
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

folder_path = "./data/pdfs"

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_texts():
    text = ""
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.pdf'):
            print(f"Skipping non-PDF file: {filename}")
            continue

        print("Loading file ",filename)
        pdf_path = os.path.join(folder_path, filename)
        
        # with open(pdf_path, 'rb') as file:
            
        # print("File: ", file)
        pdf_reader = PdfReader(pdf_path)
        
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    return text

# Function to split text into chunks
def split_text_into_chunks(text: str):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# Function to load and process PDFs from a specific folder
def load_pdfs():
    knowledge_base = None
    embeddings = OpenAIEmbeddings()
    
    text = get_pdf_texts()

    chunks = split_text_into_chunks(text)

    if knowledge_base is None:
                # Initialize the FAISS index with embeddings from the first PDF
        knowledge_base = FAISS.from_texts(chunks, embeddings)
    else:
                # Update the FAISS index with new embeddings
                # Replace this line with your actual FAISS update logic
        new_index = FAISS.from_texts(chunks, embeddings)
        knowledge_base.index.add(new_index.index)
                        
    return knowledge_base

