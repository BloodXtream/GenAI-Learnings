from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddingModel = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001"
) 

vectorStore = Chroma(
    persist_directory = "chromaDBLang",
    embedding_function = embeddingModel
)

retriever = vectorStore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)

llm = GoogleGenerativeAI(
    model = "gemini-3.1-flash-lite" 
) 

# Prompt Template 

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant that answers questions
        using the provided document context.

        Follow these rules strictly:

        1. Answer the user's question using only the information
           available in the provided context.

        2. Do not use your own external knowledge or make up
           facts that are not supported by the context.

        3. If the context contains the answer, provide a clear,
           accurate, and easy-to-understand explanation.

        4. If the context contains only part of the answer,
           explain the available information and clearly state
           what information is missing.

        5. If the answer cannot be found in the context, respond:
           "I am not able to find the answer in the documents."

        6. If multiple parts of the context are relevant, combine
           them to provide a complete answer.

        7. Avoid unnecessary repetition. Use bullet points when
           they make the answer easier to understand.

        Remember: The provided context is your only source of
        factual information for answering the question.
        """
    ),
    (
        "human",
        """
        DOCUMENT CONTEXT:
        {context}

        USER QUESTION:
        {question}

        ANSWER:
        """
    )
])


print("Rag System is Created")
print("Press 0 to Exit")

while True:
    query = input("You : ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    finalPrompt = prompt.invoke({
        "context" : context,
        "question" : query
    })

    response = llm.invoke(finalPrompt)

    print(f"\n  AI: {response}")

# = -  _  ""  ''  