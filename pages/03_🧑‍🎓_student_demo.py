import asyncio
# Create a new event loop
loop = asyncio.new_event_loop()
# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

import streamlit as st

st.title("Student Demo")
GOOGLE_AI_STUDIO = st.secrets["GEMINI_API_KEY"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = "😸" if message["role"] == "model" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")
if prompt and 'agent' in st.session_state:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Start streaming model response
    with st.chat_message("model", avatar="😸"):
        message_placeholder = st.empty()
        full_response = ""
        full_response = st.session_state.agent.chat(prompt)
        # streaming_response = st.session_state.agent.stream_chat(prompt)
        # for response_chunk in streaming_response.response_gen:
        #     full_response += response_chunk
        #     message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "model", "content": full_response})
elif 'agent' not in st.session_state:
    st.write("Agent not initialized. Please initiate Teacher Demo.")