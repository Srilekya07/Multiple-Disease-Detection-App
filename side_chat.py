import google.generativeai as genai
import streamlit as st

# Replace with your actual API key
genai.configure(api_key="AIzaSyDnMsQoYi5N0weH_uFpDazUc6B039YFC1A")

def chat_with_gemini(prompt, chat_history):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    chat_history.append(("You", prompt))
    chat_history.append(("health bot", response.text))
    return response.text

# Sidebar Chatbot UI
st.sidebar.title("SL_Chatbot")
st.sidebar.subheader("Ask anything about queries!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history inside the sidebar
with st.sidebar:
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(message)

    # Stylish input box within the sidebar
    user_input = st.text_input(
        "", "",
        key="user_input",
        placeholder="Ask anything...",
        help="Type your message and press Enter"
    )
    send_button = st.button("🎤", help="Getting as soon as possible")

    if user_input:
        response = chat_with_gemini(user_input, st.session_state.chat_history)
        # Keep only the latest Q&A pair
        st.session_state.chat_history = [("You", user_input), ("health bot", response)]
        with chat_container:
            with st.chat_message("health bot"):
                st.write(response)
