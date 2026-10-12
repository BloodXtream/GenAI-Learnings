from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-Mini-L6-v2",
)

text=[
    "Hello this is Pragyanam Tiwari",
    "Hello your name is youtube",
    "And you all are very beautiful",
]

vector=embeddings.embed_documents(text)

print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ")

print(vector)