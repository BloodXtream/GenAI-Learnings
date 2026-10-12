
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

st.set_page_config(
    page_title="Gemini AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Existing Gemini model
model = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

# Available AI personalities
modes = {
    "Angry 😡": (
        "You are a angry ai agent. Your response have agnre "
        "and response are impatient"
    ),
    "Funny 😂": (
        "You are a very funny ai agent. Your response have "
        "vity humor and jokes"
    ),
    "Sad 😢": (
        "You are a sad ai agent. Your response have saddist "
        "and negativity in it"
    )
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = None

if "chat_ended" not in st.session_state:
    st.session_state.chat_ended = False

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Funny 😂"

# Page heading
st.title("🤖 Gemini AI Chatbot")
st.caption("Your AI conversation partner")

st.divider()

# Personality selection
selected_mode = st.selectbox(
    "Choose your AI agent mode:",
    options=list(modes.keys()),
    index=list(modes.keys()).index(
        st.session_state.selected_mode
    ),
    disabled=st.session_state.messages is not None
)

# Start the chat with the selected personality
if st.session_state.messages is None:
    st.session_state.selected_mode = selected_mode
    st.session_state.messages = [
        SystemMessage(content=modes[selected_mode])
    ]

# Display the selected personality
st.caption(f"Current personality: {st.session_state.selected_mode}")

st.divider()

# Display conversation history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input
if not st.session_state.chat_ended:
    prompt = st.chat_input("Type your message here...")

    if prompt:
        # Exit command
        if prompt.strip() == "0":
            st.session_state.chat_ended = True
            st.rerun()

        # Save and display user message
        st.session_state.messages.append(
            HumanMessage(content=prompt)
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        try:
            with st.chat_message("assistant"):
                with st.spinner("Gemini is thinking..."):
                    response = model.invoke(
                        st.session_state.messages
                    )

                st.markdown(response)

            # Save AI response
            st.session_state.messages.append(
                AIMessage(content=response)
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")

else:
    st.info(
        "This chat session has ended. "
        "Refresh the page to start a new session."
    )
