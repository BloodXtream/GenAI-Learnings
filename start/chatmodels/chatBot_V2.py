from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage 

load_dotenv()

# = -  _  ""  ''  

model=GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

messages=[
    SystemMessage(content="you are a funny ai agent")
]

print('---------------------------------------------')
print('Welcome to the prototype of Gemini model...')
print("Type 0 to Exit")
print("---------------------------------------------")

while True:

    prompt=input("You : ")
    messages.append(HumanMessage(content=prompt))

    if prompt == "0":
        print("Your suck bitch")
        break

    response=model.invoke(messages)
    messages.append(AIMessage(content=response))

    print(f"Response: {response}")

print(messages)