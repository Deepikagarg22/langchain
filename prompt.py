## integrate code with openAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = openai_key
# streamlit framework

st.title('Celebrity Search Results')
input_text = st.text_input("search the celebrity you want")


# prompt templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template = "Tell me about {name}"
)
# Memory
person_memeory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memeory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memeory = ConversationBufferMemory(input_key='dob',memory_key='description_history')

# openapi llms models
llm = genai.GenerativeModel('gemini-pro-vision')
chain = LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,memory=person_memeory,output_key='person')

# prompt templates
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template = "When was {person} born"
)
chain2 = LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,memory=dob_memeory,output_key='dob')


# prompt template
third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template = "Mention five major events happen around {dob} in the world"
)

chain3 = LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,memory=descr_memeory,output_key='description')




parent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)




if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memeory.buffer)
    with st.expander('Major Events'):
        st.info(descr_memeory.buffer)



