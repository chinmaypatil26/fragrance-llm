import validators
import time
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import ast
from typing import List

# Load API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Set Streamlit page configurations
st.set_page_config(page_title="Fragrance Videos Summary", page_icon="🫧🧴✨")
st.title("Fragrance Video Summarizer 🫧🧴✨")
st.subheader("Summarize YouTube videos about fragrances and get product details.")

# Define prompt templates
perfume_list_prompt = PromptTemplate(
    template="""Create a Python list object with the names of the perfumes mentioned in the video. 
    Use accurate names, especially for Arabic or French names. Output a valid Python list object.
    Content: {text}""",
    input_variables=["text"],
)

notes_summary_prompt = PromptTemplate(
    template="""You are given the notes for the perfume. Create pointers with the labels 
    "perfume name", "top", "middle", and "base".
    Content: {text}""",
    input_variables=["text"],
)

summary_prompt = PromptTemplate(
    template="""You are a fragrance expert. Create a table with the following columns:
    - PERFUME NAME
    - NOTES (top, middle, base)
    - REVIEW
    - RATING
    If the perfume is a clone of an expensive perfume, include it in the review.
    Use the context for notes and content for reviews/ratings.
    Content: {text}
    Context: {context}""",
    input_variables=["text", "context"],
)

# Define DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# Input field for YouTube URL
youtube_url = st.text_input("Enter the YouTube URL")

if st.button("GO"):
    if not validators.url(youtube_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            # Load video transcript
            loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
            docs = loader.load()

            # Extract perfume names
            perfume_chain = load_summarize_chain(llm, "stuff", prompt=perfume_list_prompt)
            perfume_list_output = perfume_chain.invoke(docs)
            try:
                perfumes = ast.literal_eval(perfume_list_output["output_text"])
            except Exception:
                st.error("Failed to parse perfume names. Please check the video content.")
                st.stop()

            # Initialize context for notes
            context = ""

            # Extract notes for each perfume
            notes_chain = load_summarize_chain(llm, "stuff", prompt=notes_summary_prompt)
            for perfume in perfumes:
                try:
                    search_result = search.invoke(f"{perfume} perfume's top, middle, base notes")
                    document = Document(
                        page_content=search_result,
                        metadata={"source": "https://fragrantica.com"},
                    )
                    notes_output = notes_chain.invoke([document])
                    context += notes_output["output_text"] + "\n\n"
                    time.sleep(1)  # Avoid rate-limiting
                except Exception as e:
                    st.warning(f"Failed to retrieve notes for {perfume}: {e}")
                    continue

            # Generate summary
            summary_chain = load_summarize_chain(llm, "stuff", prompt=summary_prompt)
            summary_output = summary_chain.invoke({"input_documents": docs, "context": context})

            # Display the final summary
            st.success(summary_output["output_text"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
