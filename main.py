## integrate code with openAI API
import os
from constants import apiai_key
from langchain.llms import OpenAI

import streamlit as st

os.environ["OPENAI_API_KEY"] = apiai_key
# streamlit framework

st.title('Langchain Demo With openai API')
input_text = st.text_input("search the topic you want")



# openapi llms models
llm = OpenAI(temperature=0.8)


if input_text:
    st.write(llm(input_text))





