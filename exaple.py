# Import necessary libraries
import streamlit as st
import google.generativeai as genai



# Initialize Streamlit app
st.set_page_config(page_title="CODE GPT")
st.header("Stack overflow")

# Sidebar for Gemini Pro API Key input
with st.sidebar:
    # Input for Gemini Pro API Key
    gemini_api_key = st.text_input("Gemini Pro API Key", key="api_key", type="password")
    # Link to obtain Gemini Pro API Key
    "[Get a Gemini Pro API key](https://makersuite.google.com/u/2/app/apikey)"

# Function to load Gemini Pro model and get responses
def get_gemini_response(question):
    # Configure Gemini API
    genai.configure(api_key=gemini_api_key)
    # Initialize Gemini Pro model
    model = genai.GenerativeModel('gemini-pro')
    # Generate content using the model
    response = model.generate_content(question)
    return response.text

# Text area for user input
input = st.text_area(".", key="input", height=250)

# Button to generate documentation
submit = st.button("Comment")

# If the generate documentation button is clicked
if submit:
    # Check if Gemini Pro API key is provided
    if not gemini_api_key:
        st.info("Please add your Gemini Pro API key to continue.")
        st.stop()

    # Create a prompt for the Gemini model
    prompt = "ANSWER THIS " + input
    # Get Gemini's response based on the prompt
    response = get_gemini_response(prompt)
    
    # Display the generated documentation
    st.write(response)


