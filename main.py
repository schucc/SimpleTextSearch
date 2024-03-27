from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size=200,
  chunk_overlap=0
)
loader = TextLoader("facts.txt")
docs = loader.load_and_split(
  text_splitter=text_splitter
)

# below code calls openAI to calculate embeddings, will incur cost
db = Chroma.from_documents(
  docs,
  embedding=embeddings,
  persist_directory="emb"
)


results = db.similarity_search_with_score(
  "What is an interesting fact abou the English language?",
  k=2
  )

for result in results:
  print("\n")
  print(result[1]) #search score
  print(result[0].page_content)
