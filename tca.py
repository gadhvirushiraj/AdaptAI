"""
Tone-Adaptive Conversational Agent main pipeline file.
"""

import sqlite3
from groq import Groq
import streamlit as st
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from prompts import PERSONALIZED_LLM_PROMPT


def timetable_database(db_path):
    """Fetches timetable data from the database and formats it as CSV."""
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM timetable")
    tables = cursor.fetchall()

    header = "time_interval,Desk_Work,Commuting,Eating,In_Meeting,pnn50,hr\n"
    rows = [",".join(map(str, row)) for row in tables]

    return header + "\n".join(rows)


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
        "Choose a model", ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
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
    tables = timetable_database("task.db")

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
