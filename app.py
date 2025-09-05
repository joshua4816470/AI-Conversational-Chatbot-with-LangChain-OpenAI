import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # loads .env next to app.py

# Ensure env vars are present & correctly cased
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true").lower()
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Simple Q&A Chatbot with OPENAI")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Quick sanity checks (won’t print any secrets)
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY missing"
assert os.getenv("LANGCHAIN_API_KEY"), "LANGCHAIN_API_KEY missing"


## PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries"),
    ("user", "question : {question}")
])

def generate_response(question, model_name, temperature, max_tokens):
    chat = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    parser = StrOutputParser()
    chain = prompt | chat | parser
    return chain.invoke({"question": question})

## title of the app
st.title("Enhanced Q&A CHATBOT With OpenAI")

## drop down to select various open ai models
llm = st.sidebar.selectbox("select an Open AI model ", ["gpt-4o","gpt-4-turbo", "gpt-4"])

## adjust response parameter
temperature = st.sidebar.slider("temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## main interface 
st.write("Go ahead and ask any question")

question = st.text_input("You:")

if question:
    response = generate_response(question, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide query")
