from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=64
)

text=[
    "Hello this is Pragyanam Tiwari",
    "Hello your name is youtube",
    "And you all are very beautiful",
]

vector=embeddings.embed_documents(text)

print('-----------------------------------------------------------')
print(vector)