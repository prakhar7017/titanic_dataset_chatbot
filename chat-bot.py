import streamlit as st
from langchain_util import generate_response  

st.set_page_config(page_title="Titanic Chatbot", layout="centered")

st.title("🚢 Titanic Chatbot")
st.markdown("Ask me anything about the Titanic passengers!")


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question...")

if user_input:
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    
    response_type, response_content = generate_response(user_input)

    if response_type == "text":
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        with st.chat_message("assistant"):
            st.markdown(response_content)

    elif response_type == "visual":
        st.pyplot(response_content)


if st.button("Show Passenger Embarkation Data 📊"):
    _, embarked_plot = generate_response("Show embarked passenger data")
    st.pyplot(embarked_plot)
