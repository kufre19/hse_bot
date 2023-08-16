from flask import Flask, request, jsonify
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
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return jsonify(answer=result['answer'])

if __name__ == '__main__':
    app.run(debug=True)
