from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
hf_token=os.getenv("HUGGINGFACE_API_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=hf_token,
)

model=ChatHuggingFace(llm=llm)
response=model.invoke("who are you...?")
print(response.content)