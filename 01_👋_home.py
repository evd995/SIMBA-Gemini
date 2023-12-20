from trulens_eval import Tru
import logging
import sys

# Configure LlamaIndex logging to output to stdout at DEBUG level in a single line
logging.getLogger('llama_index').setLevel(logging.DEBUG)
logging.getLogger('llama_index').addHandler(logging.StreamHandler(sys.stdout))

import asyncio
# Create a new event loop
loop = asyncio.new_event_loop()
# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

import streamlit as st

st.set_page_config(
    page_title="SIMBA-Gemini Demo",
    page_icon="😸",
)

st.title("Welcome to the SIMBA-Gemini Demo")

if "tru_student" not in st.session_state:
    tru = Tru()
    tru.reset_database()

st.sidebar.success("Select a page above.")

