from flask import Flask, request, jsonify
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import logging
# import constants

app = Flask(__name__)

# Your existing code setup
# os.environ["OPENAI_API_KEY"] = constants.APIKEY

PERSIST = False
loader = DirectoryLoader("data/")
index = VectorstoreIndexCreator().from_loaders([loader])
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)
chat_history = []

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data['query']
    result = chain({"question answer only in english": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return jsonify(answer=result['answer'])

@app.route('/', methods=['GET'])
def home():
    return "home works"

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error('Server Error: %s', error)
    return error, 500

@app.errorhandler(Exception)
def handle_exception(error):
    app.logger.error('Unhandled Exception: %s', error)
    return error, 500


if __name__ == '__main__':
    app.run(debug=True)
