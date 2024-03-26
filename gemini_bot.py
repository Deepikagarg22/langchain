import os
from typing import Any, List, Mapping, Optional
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import google.generativeai as genai
import streamlit as st


os.environ['GOOGLE_API_KEY'] = "AIzaSyDh9gNFTxY9d3Oz9QWdTPy6SG854-ZU3p8"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro')


class GeminiProLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-pro"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              **kwargs: Any,) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted")

        gemini_pro_model = model

        model_response = gemini_pro_model.generate_content(
            prompt,
            generation_config={"temperature": 0.9}
        )
        if len(model_response.candidates[0].content.parts) > 0:
            return model_response.candidates[0].content.parts[0].text
        else:
            return "<No answer given by gemini-pro"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_id": "gemini-pro", "temperature": 0.9}

@st.cache_resource
def load_chain():
    llm = GeminiProLLM()
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

chatchain = load_chain()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.sidebar.title("Sidebar")

if st.sidebar.button("Clear Conversation", key="clear"):
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    role = message['role']
    content = message['content']
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("You:")

if prompt:
    st.session_state['messages'].append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = chatchain(prompt)['response']
    st.session_state['messages'].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)



        '''Please add option, when user select 'mid cycle alert' then ---- ask question 'do u want to mid cycle alert?' yes/no.'''

