import os
import openai
from flask import Flask, request, jsonify
from pdf_ingester import load_pdfs
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Initialize Flask app
app = Flask(__name__)

# Initialize the knowledge base
knowledge_base = load_pdfs()

# REST API endpoint to accept user's question
@app.route('/ask', methods=['POST'])
def ask_user():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "Question is required"}), 400
    
    # Answer the question using the knowledge base
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
    
    return jsonify({"answer": response})

