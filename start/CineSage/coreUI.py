
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)

# Gemini model
model = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        """
        You are an expert movie information extraction assistant.

        Your task is to analyze the provided paragraph and extract
        the most useful information about the movie.

        Extract the following fields:

        1. Movie Name
        2. Director
        3. Writers / Source Material
        4. Main Cast and Character Names
        5. Genre (only if explicitly mentioned or clearly supported)
        6. Plot Summary (2-3 sentences)
        7. Key Themes
        8. Main Protagonist
        9. Setting / Location
        10. Awards, Rankings, or Recognition
        11. Interesting Facts
        12. Overall Takeaway (one concise sentence)

        Instructions:
        - Use only information supported by the provided paragraph.
        - Do not invent missing details or use outside knowledge.
        - If a field is unavailable, return "Not specified".
        - Keep the output concise, clear, and useful.
        - Distinguish the director from the writer.
        - Distinguish actors from the characters they portray.
        - Do not repeat the same information unnecessarily.
        - Write the plot summary in simple language.
        - Follow the exact output format below.

        Output Format:

        Movie Title:
        Director:
        Writer / Source Material:

        Main Cast:
        - Actor Name — Character Name

        Genre:
        Release Year:

        Setting:
        Main Protagonist:

        Key Themes:
        - Theme 1
        - Theme 2

        Awards / Rankings:
        Interesting Facts:

        Overall Takeaway:

        Quick Summary:
        Write a clear and concise summary in 2-3 sentences.

        Paragraph:
        {text}
        """
    )
])

# Page heading
st.title("🎬 Movie Information Extractor")
st.caption(
    "Extract useful movie details and summaries from any paragraph."
)

st.divider()

# Paragraph input
paragraph = st.text_area(
    "Enter Movie Paragraph",
    placeholder="Paste your movie description or paragraph here...",
    height=220
)

# Extract information
if st.button("Extract Information", type="primary", use_container_width=True):
    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")
    else:
        try:
            with st.spinner("Analyzing the paragraph..."):
                final_prompt = prompt.invoke(
                    {"text": paragraph}
                )

                response = model.invoke(final_prompt)

            st.subheader("Extracted Movie Information")
            st.markdown(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")