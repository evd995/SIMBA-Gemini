import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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
st.sidebar.success("Select a page above.")

