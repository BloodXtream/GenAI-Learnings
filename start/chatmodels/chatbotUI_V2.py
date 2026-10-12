
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Gemini AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Existing Gemini model — unchanged
model = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

# Initialize conversation history only once per session
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="you are a funny ai agent")
    ]

if "chat_ended" not in st.session_state:
    st.session_state.chat_ended = False

# Page heading
st.title("🤖 Gemini AI Chatbot")
st.caption("Your funny AI conversation partner")

st.divider()

# Display previous conversation
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Message input
if not st.session_state.chat_ended:
    prompt = st.chat_input("Type your message here...")

    if prompt:
        # Preserve the original exit command
        if prompt.strip() == "0":
            st.session_state.chat_ended = True
            st.rerun()

        # Add and display the user's message
        st.session_state.messages.append(
            HumanMessage(content=prompt)
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate the AI response using existing history
        try:
            with st.chat_message("assistant"):
                with st.spinner("Gemini is thinking..."):
                    response = model.invoke(
                        st.session_state.messages
                    )

                st.markdown(response)

            # Save the AI response to conversation history
            st.session_state.messages.append(
                AIMessage(content=response)
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")

else:
    st.info("This chat session has ended. Refresh the page to start a new session.")
