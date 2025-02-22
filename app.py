__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import validators
import time
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    model="Llama-3.3-70b-Versatile", groq_api_key=groq_api_key, max_tokens=20000
)
torch.classes.__path__ = []
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_name = 'intfloat/multilingual-e5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

chroma_client = chromadb.PersistentClient(path="chroma_db", settings=chromadb.config.Settings(
    chroma_db_impl="duckdb"
))
embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
collection_name = "fragrantica_embeddings"
collection = chroma_client.get_or_create_collection(name=collection_name)
vectorstore = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function=embeddings_model
)

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
Convert the following text into a valid Python list of strings. Give only the PYTHON LIST OBJECT that can directly be evaluated as a list, no extra characters like ```.
Text: {text}
"""

list_prompt = PromptTemplate(template=list_template, input_variables=["text"])

perfumes_list_template = PromptTemplate(template=perfume_list, input_variables=["text"])

notes_template = PromptTemplate(template=notes_summary, input_variables=["text"])

generic_url = st.text_input("Enter the youtube url", label_visibility="collapsed")


if st.button("GO"):
    if not validators.url(generic_url):
        st.error("Please enter a valid url")
    else:
        try:
            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
            docs = loader.load()

            perfume_list_summary_chain = load_summarize_chain(
                llm, "stuff", prompt=perfumes_list_template
            )
            perfumes = perfume_list_summary_chain.invoke(docs)

            chain = list_prompt | llm
            print(chain.invoke(input=perfumes["output_text"]).content)
            perfumes = eval(chain.invoke(input=perfumes["output_text"]).content)
            notes_summary_chain = load_summarize_chain(
                llm, "stuff", prompt=notes_template
            )

            context = ""
            for perfume in perfumes:
                print(perfume)
                inputs = tokenizer(perfume, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().detach().numpy()

                # Ensure query_embedding is a 1D NumPy array
                query_embedding = np.array(query_embedding).squeeze()
                print(query_embedding.shape)
                stored_docs = collection.query(query_embeddings=query_embedding, n_results=1)
                print(stored_docs)
                for i in range(1):
                    context += stored_docs['metadatas'][0][i]['text']
                    context += "\n"
                context += "\n"
                time.sleep(1)

            summary_prompt = """
            You are a fragrance expert with a lot of knowledge from the web.
            Create a table with the PERFUME NAME, NOTES of the perfume, the REVIEW and the RATING. 
            Also mention the clone in the review if it is a clone of an expensive perfume.
            Note: The content/context may contain arabic names if it's a clone video or a lot of french names if it isn't, do not mess up the names.
            You are given the Content and Context below.
            Use the perfume names, notes from the context and context only. Use the content for review and rating. 
            If a table can be created, output the table or else output the text with the perfume name, notes, review and rating.

            content: {text}
            context: {context}
            """
            summary_prompt_template = PromptTemplate(
                template=summary_prompt, input_variables=["text", "context"]
            )

            chain = load_summarize_chain(llm, "stuff", prompt=summary_prompt_template)

            output_summary = chain.invoke({"input_documents": docs, "context": context})

            st.success(output_summary["output_text"])
        except Exception as e:
            st.error(f"Exception: {e}")
