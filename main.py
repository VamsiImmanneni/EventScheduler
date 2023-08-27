import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import *
from langchain.chains.summarize import load_summarize_chain
import tempfile
from langchain.docstore.document import Document

st.title("Email Event Scheduler")

email_content = st.text_area("Enter email content")

def eventExtractor(email_content):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    system_template = """You are an assistant designed to extract events from email content."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please extract any events mentioned in the email content: '{email_content}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(email_content=email_content)
    return result

def calendarInviteGenerator(events):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    system_template = """You are an assistant responsible for generating Google Calendar invites for each event."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please generate a Google Calendar invite for each event in the following list: {events}."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(events=events)
    return result

if st.button("Submit"):
    if email_content:
        events = eventExtractor(email_content)
        if events:
            calendar_invites = calendarInviteGenerator(events)
            st.markdown(f"Generated Google Calendar invites: {calendar_invites}")

