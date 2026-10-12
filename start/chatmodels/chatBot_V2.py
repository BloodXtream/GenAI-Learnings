from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage 

load_dotenv()

# = -  _  ""  ''  

model=GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

print("Choose your AI agent mode: ")
print("press 1 for Angry mode: ")
print("press 2 for funny mode:")
print("press 3 for sad mode:")

choise=int(input("tell yor response:"))
if choise == 1:
    mode = "You are a angry ai agent. Your response have agnre and response are impatient"
elif choise == 2:
    mode = "You are a very funny ai agent. Your response have vity humor  and jokes "
elif choise == 3:
    mode = "You are a sad ai agent. Your response have saddist and negativity in it"

messages=[
    SystemMessage(content=mode)
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