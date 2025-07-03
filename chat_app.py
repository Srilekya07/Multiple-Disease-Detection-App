import google.generativeai as genai
import streamlit as st

# Replace with your actual API keygenai.configure(api_key="AIzaSyDnMsQoYi5N0weH_uFpDazUc6B039YFC1A")

def chat_with_gemini(prompt, chat_history):
    model = genai.GenerativeModel("gemini-2.0-flash")  # Using Pro model for better responses
    response = model.generate_content(prompt)
    chat_history.append(("You", prompt))
    chat_history.append(("health bot", response.text))
    return response.text

st.sidebar.title("Chatbot")
page = st.sidebar.radio("SL_Chatbot", ["SL_Chatbot"])
menu = ["SL_Chatbot", ""]
choice = st.sidebar.selectbox("Select Option", menu)

if choice == "SL_Chatbot":
    st.subheader("Ask anything about queries !")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
chat_container = st.container()
with chat_container:
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

# Stylish input box (like the reference image)
with st.container():
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        user_input = st.text_input(
            "", "",
            key="user_input",
            placeholder="Ask anything...",
            help="Type your message and press Enter"
        )
    with col3:
        send_button = st.button("🎤", help="getting as soon as possible")

if user_input:
    response = chat_with_gemini(user_input, st.session_state.chat_history)
    with chat_container:
        with st.chat_message("health bot"):
            st.write(response)
