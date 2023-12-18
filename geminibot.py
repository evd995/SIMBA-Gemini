import streamlit as st
import google.generativeai as genai

GOOGLE_AI_STUDIO = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GOOGLE_AI_STUDIO)

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}


model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config)
st.title("Gemini Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = "😸" if message["role"] == "model" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

chat = model.start_chat(history=[
                {"role": m["role"], 'parts': [m["content"]]}
                for m in st.session_state.messages
            ])

prompt = st.chat_input("What is up?")
if prompt:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Start streaming model response
    with st.chat_message("model", avatar="😸"):
        message_placeholder = st.empty()
        full_response = ""
        for response_chunk in chat.send_message(prompt, stream=True):
            full_response += response_chunk.text
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "model", "content": full_response})

    