from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st






# streamlit ui setup
st.title('Celebrity Search Results')
input_text = st.text_input("Search for the celebrity you want")


# initialising hugging face question answering pipeline
qa_pipeline = pipeline("question-answering",model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased")



# Memory
person_memeory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memeory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memeory = ConversationBufferMemory(input_key='dob',memory_key='description_history')

# prompt templates -1
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template = "Tell me about {name}"
)
chain = LLMChain(
    llm=qa_pipeline,prompt=first_input_prompt,verbose=True,memory=person_memeory,output_key='person')

# prompt templates - 2
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template = "When was {person} born"
)
chain2 = LLMChain(
    llm=qa_pipeline,prompt=second_input_prompt,verbose=True,memory=dob_memeory,output_key='dob')



# prompt template -3
third_input_prompt = PromptTemplate(input_variables=['dob'],
                               template="what are five major events happened around {dob} in the world?")



chain3=LLMChain(
    llm=qa_pipeline,prompt=third_input_prompt,verbose=True,memory=descr_memeory,output_key='description'
)


parent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)



# Check for user input and run the conversation chain
if input_text:
    results = parent_chain({'name': input_text})

    # Display results using Streamlit
    st.write(results)

    with st.expander('Person Name'):
        st.info(person_memeory.buffer)
    with st.expander('Major Events'):
        st.info(descr_memeory.buffer)




