from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

message=[]

model=GoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    Rating: Optional[float]
    summary: str

parser=PydanticOutputParser(pydantic_object=Movie)

# = -  _  ""  ''  

prompt=ChatPromptTemplate.from_messages([(
    'system',"""
    Extract Movie Information from the paragraph 
    {format_instruction}"""),
    ('human','{text}')
])

para=input("Enter Your Paragraph :")
final_prompt=prompt.invoke(
    {
        "text":para,
        "format_instruction":parser.get_format_instructions()
    }
)
response=model.invoke(final_prompt)

print(response)