from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

model=GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

messages=[]

print('---------------------------------------------')
print('Welcome to the prototype of Gemini model...')
print("Type 0 to Exit")
print("---------------------------------------------")

while True:

    prompt=input("You : ")
    messages.append(prompt)

    if prompt == "0":
        print("Your suck bitch")
        break

    response=model.invoke(messages)
    messages.append(response)

    print(f"Response: {response}")

print(messages)