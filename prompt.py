from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever

import langchain
langchain.debut=True

load_dotenv()

embeddings = OpenAIEmbeddings()
db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings
)
retriever = RedundantFilterRetriever(
  embeddings=embeddings, 
  chroma=db
  )
# retriever = db.as_retriever()

chat = ChatOpenAI()
chain = RetrievalQA.from_chain_type(
  llm=chat,
  retriever=retriever,
  chain_type="stuff"
)

result = chain.run("What is my dog's name")

print(result)