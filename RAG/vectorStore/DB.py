from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Python is a popular programming language used in AI, web development, and data science.",
        metadata={"source": "python.txt", "topic": "Python"}
    ),
    Document(
        page_content="Machine learning enables computers to learn patterns from data and make predictions.",
        metadata={"source": "ml.txt", "topic": "Machine Learning"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) combines document retrieval with language models to generate context-aware answers.",
        metadata={"source": "rag.txt", "topic": "Generative AI"}
    )
]

embeddingModel = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorStore = Chroma.from_documents(
    documents = docs,
    embedding = embeddingModel,
    persist_directory = "chroma-langchain-db"
)

result = vectorStore.similarity_search("Which one predict the data with the help of patterns",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorStore.as_retriever()

docs = retriver.invoke("Explain Deep Learning")

for d in docs:
    print(d.page_content) 
# = -  _  ""  ''  