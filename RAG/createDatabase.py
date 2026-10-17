# load the pdf 
# split into chunks
# create the embeddings
# store in vectore Store
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader

load_dotenv()

data = PyPDFLoader("document-loaders/DeepLearning.pdf") 
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 200
) 

chunks = splitter.split_documents(docs)

print("Total PDF pages:", len(docs))
print("Total chunks:", len(chunks))

embeddingModel = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001"
)

vectorStore = Chroma.from_documents(
    documents = chunks[:20],
    embedding = embeddingModel,
    collection_name = "deep_learning",
    persist_directory = "chromaDBLang"
)

