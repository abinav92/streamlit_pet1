import streamlit as st
from openai import OpenAI

st.set_page_config(page_title = "Are You Ready For A Pet? 🐶🐱🦜")
st.title("Are You Ready For A Pet? 🐶🐱🦜")
st.sidebar.success("Current page: Are You Ready For A Pet? 🐶🐱🦜")
st.write("Take this short questionnaire to discover more❗")

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
    
def clicked_check_answer():
    st.session_state.quiz_submitted = True

def clicked_retry():
    st.session_state.quiz_submitted = False

q1_r = st.radio("Does your family agree to having a pet?", ["Yes", "No"], key=10, horizontal = True, disabled=st.session_state.quiz_submitted)
q2_r = st.radio("Are you committed to looking after it for life?", ["Yes", "No"], key=20, horizontal = True, disabled=st.session_state.quiz_submitted)
q3_r = st.radio("Can you afford to pay for pet's veterinary bills, food, and grooming?", ["Yes", "No"], key=30, horizontal = True, disabled=st.session_state.quiz_submitted)
q4_r = st.radio("Do we have enough time to care for pet?", ["Yes", "No"], key=40, horizontal = True, disabled=st.session_state.quiz_submitted)
q5_r = st.radio("Do we have enough space at home for a pet?", ["Yes", "No"], key=50, horizontal = True, disabled=st.session_state.quiz_submitted)
q6_r = st.radio("Do we know how to care for your pet properly?", ["Yes", "No"], key=60, horizontal = True, disabled=st.session_state.quiz_submitted)

response = [q1_r == "Yes", q2_r == "Yes", q3_r == "Yes", q4_r == "Yes", q5_r == "Yes", q6_r == "Yes"]
questions = ["Does your family agree to having a pet?",
             "Are you committed to looking after it for life?",
             "Can you afford to pay for pet's veterinary bills, food, and grooming?",
             "Do we have enough time to care for pet?",
             "Do we have enough space at home for a pet?",
             "Do we know how to care for your pet properly?"]

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

def generate_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You are a helpful and funny pet keeper who gives constructive feedbacks\n"
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                }
            ]
            },
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content


# st.write(yes_responses)
# st.write(no_responses)
# st.write(completion)

if st.button("Check Answer", type="primary", key="check_answer", on_click = clicked_check_answer):
    
    if not all(response):
        
        no_responses = [questions[i] for i in range(len(response)) if not response[i]]
        yes_responses = [questions[i] for i in range(len(response)) if response[i]]
        
        prompt_msg = f"The user took a pet readines quiz. Provide a constructive reply encouraging/commending the user. Even if one of the response is no, reply along the lines of please consider the decision to get a pet in a constructive manner.\\\
        The negative response is {no_responses} and positive response is {yes_responses}. If there is no negative response. Just commend the user. Keep your response short in 1 paragraph."
        completion = generate_completion(prompt_msg)
        st.warning(completion)
        if st.button("Re-try", type="secondary", key = "re-try", on_click = clicked_retry):
            st.rerun()
  
    else:
        
        no_responses = [questions[i] for i in range(len(response)) if not response[i]]
        yes_responses = [questions[i] for i in range(len(response)) if response[i]]
        
        prompt_msg = f"The user took a pet readines quiz. Provide a constructive reply encouraging/commending the user. Even if one of the response is no, reply along the lines of please consider the decision to get a pet in a constructive manner.\\\
        The negative response is {no_responses} and positive response is {yes_responses}. If there is no negative response. Just commend the user. Keep your response short in 1 paragraph."
        completion = generate_completion(prompt_msg)
        st.success(completion)
        st.balloons()
        if st.button("Re-try", type="secondary", key = "re-try", on_click = clicked_retry):
            st.rerun()


# .\capstone_project\scripts\activate
# streamlit run I_Love_Pets.py
