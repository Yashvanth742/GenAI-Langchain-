import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="AI Mood Chatbot",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# Model
# -----------------------------

model = ChatOllama(
    model="llama3:latest"
)


# -----------------------------
# AI Modes
# -----------------------------

modes = {
    "Angry 😠": "You are an angry assistant. You respond to user queries with a frustrated and irritable tone.",
    "Happy 😊": "You are a happy assistant. You respond to user queries with a cheerful and positive tone.",
    "Sad 😔": "You are a sad assistant. You respond to user queries with a melancholic and sorrowful tone.",
    "Normal 🙂": "You are a normal assistant. You respond to user queries in a neutral and balanced tone."
}


# -----------------------------
# Session state
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = None


# -----------------------------
# Mode selection
# -----------------------------

st.title("🤖 AI Mood Chatbot")
st.caption("Choose how you want your AI assistant to behave")


selected_mode = st.radio(
    "Choose AI mode:",
    list(modes.keys()),
    horizontal=True
)


# Reset conversation when mode changes
if st.session_state.mode != selected_mode:

    st.session_state.mode = selected_mode

    st.session_state.messages = [
        SystemMessage(
            content=modes[selected_mode]
        )
    ]


# -----------------------------
# Display conversation
# -----------------------------

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.write(message.content)


# -----------------------------
# User input
# -----------------------------

prompt = st.chat_input("Type your message...")


if prompt:

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Add user message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # Get AI response
    response = model.invoke(
        st.session_state.messages
    )

    answer = response.content

    # Display AI response
    with st.chat_message("assistant"):
        st.write(answer)

    # Save AI response
    st.session_state.messages.append(
        AIMessage(content=answer)
    )