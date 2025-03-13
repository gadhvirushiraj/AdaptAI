"""
Tone-Adaptive Conversational Agent main pipeline file.
"""

import sqlite3
import streamlit as st
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from prompts import PERSONALIZED_LLM_PROMPT
from crud_db import get_timetable_w_stress_lvl


def tca(groq_api_key):
    """Main function to set up the tone-adaptive conversational interface and handle interactions."""

    # Streamlit UI setup
    st.title("Chat with AdaptAI!")
    st.write(
        "Hello! I'm your friendly personalized chatbot. Let's start our conversation!"
    )

    # Sidebar customization
    st.sidebar.title("Customization")
    model = st.sidebar.selectbox(
        "Choose a model", ["llama-3.3-70b-versatile, llama3-70b-8192, llama3-8b-8192"]
    )
    conversational_memory_length = st.sidebar.slider(
        "Conversational memory length:", 1, 10, value=5
    )

    # Memory for conversation history
    memory = ConversationBufferWindowMemory(
        k=conversational_memory_length, memory_key="chat_history", return_messages=True
    )

    user_question = st.text_input("Ask a question:")

    # Restore chat history if available
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({"input": message["human"]}, {"output": message["AI"]})

    # Initialize Groq chat model
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    # Fetch timetable data
    tables = get_timetable_w_stress_lvl("task.db")

    if user_question:
        # Format query with context
        query = PERSONALIZED_LLM_PROMPT.format(input_context=tables)
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=query),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        # Create conversation chain
        conversation = LLMChain(
            llm=groq_chat, prompt=prompt, verbose=True, memory=memory
        )

        # Get chatbot response
        response = conversation.predict(human_input=user_question)
        message = {"human": user_question, "AI": response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)


if __name__ == "_main_":
    GROQ_API_KEY = "your-groq-api-key"
    tca(GROQ_API_KEY)
