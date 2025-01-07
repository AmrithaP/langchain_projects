import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Langchain Demo with OpenAI API')

# Input text field
input_text = st.text_input("Search the topic you want")

# OpenAI LLM models
llm = OpenAI(temperature=0.8)
# Temp - range 0 to 1, controls randomness of the response

if input_text:
    # Generate the response using the LLM
    response = llm(input_text)
    st.write(response)
