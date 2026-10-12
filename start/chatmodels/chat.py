from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

# model=init_chat_model("groq:openai/gpt-oss-safeguard-120b")
model=ChatGroq(model="openai/gpt-oss-120b",max_tokens=1000,temperature=0.5)

response=model.invoke("Write a -poem on laptops")

print(response.content)