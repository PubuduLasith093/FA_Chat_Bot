from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf_file, text_split, downlaod_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAI
import pinecone
from langchain.prompts import PromptTemplate
# from langchain_community.llms import CTransformers
# from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY="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"



embeddings = downlaod_hugging_face_embeddings()

#Initializing the Pinecone
#pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "fachatbot"

#Loading the index
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

retrever = docsearch.as_retriever(search_type = "similarity", search_kwargs={"k":3})

llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}')
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retrever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    data = request.get_json()

    if not data or 'msg' not in data:
        return jsonify({"error": "Invalid request, 'msg' field is required"}), 400

    input = data['msg']
    print(input)
    result=rag_chain.invoke({"input": input})
    print("Response : ", result["answer"])
    return jsonify({"response": result["answer"]})



if __name__ == '__main__':
    #app.run(host="0.0.0.0", port= 8080, debug= True)
    app.run(host="0.0.0.0")