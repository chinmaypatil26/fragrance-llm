import validators
import time
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_groq import ChatGroq

groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model="Llama-3.3-70b-Versatile", groq_api_key=groq_api_key, max_tokens=2000)

# Define prompts

st.set_page_config(page_title="Fragrance Videos Summary", page_icon="🫧🧴✨")
st.title("Summarize Youtube videos about fragrances and get product details. 🫧🧴✨")
st.subheader("Summarize URL")

perfume_list = """
Create a python list object with the names of the perfumes mentioned in the video.
Note: The video may contain arabic names if it's a clone video (title contains "clone") or a lot of french names if it isn't, please do not mess up the names.
You are given the Content below.

Content: {text}
"""

notes_summary = """
You are given the notes for the perfume, create pointers with the labels "perfume name", "top", "middle" and "base".

Content: {text}
"""

list_template = """
Convert the following text into a valid Python list of strings. Give only the python extract, no extra characters.
Text: {text}
"""

list_prompt = PromptTemplate(template=list_template, input_variables=["text"])

search = DuckDuckGoSearchRun()

perfumes_list_template = PromptTemplate(template=perfume_list, input_variables=['text'])

notes_template = PromptTemplate(template=notes_summary, input_variables=['text'])

generic_url = st.text_input("Enter the youtube url", label_visibility="collapsed")


if st.button("GO"):
    if not validators.url(generic_url):
        st.error("Please enter a valid url")
    else:
        try:
            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
            docs = loader.load()

            perfume_list_summary_chain = load_summarize_chain(llm, "stuff", prompt=perfumes_list_template)
            perfumes = perfume_list_summary_chain.invoke(docs)

            chain = list_prompt | llm
            print(chain.invoke(input=perfumes['output_text']).content)
            perfumes = eval(chain.invoke(input=perfumes['output_text']).content)
            notes_summary_chain = load_summarize_chain(llm, "stuff", prompt=notes_template)

            context = ""

            for perfume in perfumes:
                search_result = search.invoke(f"{perfume} perfume's name, top, middle, base notes")
                document = Document(
                    page_content=search_result,
                    metadata={"source": "https://fragrantica.com"}
                )
                notes = notes_summary_chain.invoke([document])['output_text']
                context += notes
                time.sleep(1)

            summary_prompt = """
            You are a fragrance expert with a lot of knowledge from the web.
            Create a table with the PERFUME NAME, NOTES of the perfume, the REVIEW and the RATING. 
            Also mention the clone in the review if it is a clone of an expensive perfume.
            Note: The content/context may contain arabic names if it's a clone video or a lot of french names if it isn't, do not mess up the names.
            You are given the Content and Context below.
            Use the perfume names, notes from the context and context only. Use the content for review and rating. 
            Provide only the table as the output. This output will be outputted directly in streamlit success method.

            content: {text}
            context: {context}
            """
            summary_prompt_template = PromptTemplate(template=summary_prompt, input_variables=['text', 'context'])

            chain = load_summarize_chain(llm, "stuff", prompt=summary_prompt_template)

            output_summary = chain.invoke({"input_documents": docs, 'context':context})

            st.success(output_summary['output_text'])
        except Exception as e:
            st.error(f'Exception: {e}')
