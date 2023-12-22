import asyncio
# Create a new event loop
loop = asyncio.new_event_loop()
# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

import streamlit as st
from trulens_eval import Tru


st.title("Student Demo")
GOOGLE_AI_STUDIO = st.secrets["GEMINI_API_KEY"]

col1, col2 = st.columns([3, 1])

if "activity_goal" in st.session_state:
    with col1:
        st.markdown(f"**The goal for this activity is: {st.session_state.activity_goal}**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = "😸" if message["role"] == "model" else None
    with col1:
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
        with st.session_state.tru_student as recording:
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

if "agent" in st.session_state:
    tru = Tru()
    records, _ = tru.get_records_and_feedback(app_ids=[])
    if '[METRIC] Groundedness' in records.columns:
        groundedness_scores = records['[METRIC] Groundedness']
        if (groundedness_scores < 0.3).any():
            with col2:
                st.markdown("🚨 **Low groundedness of the assistant's answers.**")
                st.markdown("The assistant may be hallucinating some facts, giving information that is not based on course context or related sources. I suggest looking directly at the course material to verify these facts or asking the teacher.")
                ungrounded_records = records.loc[groundedness_scores < 0.3, ['input', 'output', '[METRIC] Groundedness']]
                for ix, record in ungrounded_records.iterrows():
                    st.markdown(f"**Input**: {record['input']} \n **Output**: {record['output']} \n **Groundedness**: {record['[METRIC] Groundedness']}")
