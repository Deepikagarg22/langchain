import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import fitz  
paraphrase_model_name = "humarin/chatgpt_paraphraser_on_T5_base"
paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_name)
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name)

qa_model_name = "t5-large"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)

st.set_page_config(
    page_title="interview",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f'''
    <style>
        .sidebar .sidebar-content {{
            width: 375px;
        }}
    </style>
''', unsafe_allow_html=True)

st.markdown("<h1 style='font-family:Courier; color:Brown; font-size: 70px;text-align: center;'>Resume Q&A Generator</h1>", unsafe_allow_html=True)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'preprocess' not in st.session_state:
    st.session_state['preprocess'] = 0
if 'questions' not in st.session_state:
    st.session_state['questions'] = ''
if 'pre_prompt' not in st.session_state:
    st.session_state['pre_prompt'] = []
if 'iterator' not in st.session_state:
    st.session_state['iterator'] = 0
if 'question_index' not in st.session_state:
    st.session_state['question_index'] = 0

uploaded_file = st.file_uploader('Choose your resume file', type=["pdf", "txt"])


def generate_response(prompt):
    inputs = qa_tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True)
    outputs = qa_model.generate(inputs, max_length=150, num_return_sequences=1,do_sample=True)
    response = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


# def generate_response(prompt):
#     input_text = f"generate interview questions: {prompt}"
#     input_ids = qa_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
#     output_ids = qa_model.generate(input_ids, max_length=80, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
#     response = qa_tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return response


def generate_questions(segment, question_type):
    if question_type == "behavioral questions":
        template = f"As a candidate, how would you approach the following behavioral questions based on your experience:\n{segment}\n"
    elif question_type == "technical questions":
        template = f"As a candidate, how would you address the following technical questions based on your experience:\n{segment}\n"
    else:
        template = f"As a candidate, how would you respond to the following general questions based on your experience:\n{segment}\n"
    
    return template

def paraphrase_text(text):
    input_ids = paraphrase_tokenizer(
        f'paraphrase: {text}',
        return_tensors="pt", padding="longest",
        max_length=128,
        truncation=True,
    ).input_ids
    
    outputs = paraphrase_model.generate(
        input_ids, repetition_penalty=10.0,
        num_return_sequences=1, no_repeat_ngram_size=2,
        max_length=128
    )

 
    paraphrased_text = paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # cleaned_paraphrased_text = " ".join(sorted(set(paraphrased_text.split()), key=paraphrased_text.split().index))

    return paraphrased_text


def preprocess_resume(file_content):
    paraphrased_content = paraphrase_text(file_content)
    
    return paraphrased_content

def extract_text_from_pdf(file_path):
    pdf_doc = fitz.open(file_path)
    pdf_text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc[page_num]
        pdf_text += page.get_text()
    return pdf_text



def query():
    if st.session_state['question_index'] < len(st.session_state['pre_prompt']):
        current_prompt = st.session_state['pre_prompt'][st.session_state['question_index']]
        response = generate_response(current_prompt)
        bot = response
        user = get_text()

        st.session_state.past.append(user)
        st.session_state.generated.append(bot)
        prompt = st.session_state['pre_prompt'][-1] + ' ' + bot + 'Candidate: ' + user + '\nInterviewer:'
        st.session_state['pre_prompt'].append(prompt)
        st.session_state['iterator'] += 1


        if user.lower() == 'stop' or st.session_state['question_index'] == len(st.session_state['pre_prompt']) - 1:
            st.write("Interview stopped. If you have more questions, please restart.")
        else:
            st.session_state['question_index'] += 1
            st.button("Next Question")

def get_text():
    input_text = st.text_input(label='Answer my questions:', key="input")
    return input_text


if uploaded_file is not None:
    if st.session_state['preprocess'] == 0:
        temp_file_path = "temp_resume_file" + uploaded_file.name[-4:]
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        if temp_file_path.endswith('.pdf'):
            resume_text = extract_text_from_pdf(temp_file_path)
        elif temp_file_path.endswith('.txt'):
            with open(temp_file_path, 'r', encoding='utf-8') as txt_file:
                resume_text = txt_file.read()
        else:
            st.error("Unsupported file format. Please upload a PDF or TXT file.")
            st.stop()

        preprocessed_resume = preprocess_resume(resume_text)

        passage_segments = [segment.strip() for segment in preprocessed_resume.split('.') if segment]
        prompts = [f"extract the question for the interview preparation from the uploaded file: {segment}?" for segment in passage_segments]
        # prompts =''' [f"As an interviewer, what questions would you pose to a candidate with the following experience:\n{segment}?" for segment in passage_segments"]'''
        # prompts = [f"As an interviewer, what questions would you pose to a candidate with the following experience:\n{segment}?" for segment in passage_segments]

        # behavioral_template = "behavioral questions"
        # technical_template = "technical questions"
        # general_template = "questions"

        # prompts = [generate_questions(segment, behavioral_template) for segment in passage_segments]
        # prompts += [generate_questions(segment, technical_template) for segment in passage_segments]
        # prompts += [generate_questions(segment, general_template) for segment in passage_segments]

        # prompts = [
        # f'''Generate technical questions based on the candidate's programming skills and experience.based on{preprocessed_resume}",'''
        # # "Formulate questions about the candidate's problem-solving abilities as highlighted in the resume.based on{resume_text}",
        # # "Generate questions related to the candidate's project experience and contributions.",
        # # "Formulate questions about the candidate's communication and teamwork skills based on the resume.",
        # # "Generate questions about the candidate's educational background and relevant coursework.",
        # # "Formulate questions related to the candidate's work experience, focusing on their achievements and responsibilities.",
    
    # ]
        generated_questions = []
        for prompt in prompts:
            question = generate_response(prompt)  
            # generated_questions.append(question)
    
        for question in generated_questions:
            st.write(question)
        st.session_state['preprocess'] = 1
        st.session_state['pre_prompt'] = prompts
        st.session_state['iterator'] = 0


output = query()

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.write(st.session_state["generated"][i])
        st.write(st.session_state['past'][i])