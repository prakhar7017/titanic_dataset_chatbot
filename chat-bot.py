import streamlit as st
import os
from langchain_util import query_titanic, visualize_embarked_data  
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

    response = query_titanic(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

if st.button("Show Passenger Embarkation Data 📊"):
    visualize_embarked_data()
