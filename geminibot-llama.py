import asyncio

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

import streamlit as st
import google.generativeai as genai
from llama_index.llms import ChatMessage, Gemini
from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.embeddings import GeminiEmbedding
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.agent.react.formatter import ReActChatFormatter
from react_prompt import CUSTOM_REACT_CHAT_SYSTEM_HEADER

GOOGLE_AI_STUDIO = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GOOGLE_AI_STUDIO)

# Set up the model
generation_config = {
  "temperature": 0.3,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}


model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config)
st.title("Gemini Bot")

documents = SimpleDirectoryReader(
    input_files=["./Sample_Syllabus.pdf"]
).load_data()
document = Document(text="\n\n".join([doc.text for doc in documents]))

# Using the embedding model to Gemini
embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_AI_STUDIO
)
service_context = ServiceContext.from_defaults(
    llm=Gemini(api_key=GOOGLE_AI_STUDIO), embed_model=embed_model
)

index = VectorStoreIndex(
    [document],
    service_context=service_context
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_engine" not in st.session_state:
    st.session_state.query_engine = index.as_query_engine(
        service_context=service_context,
        verbose=True
        )
    query_engine_tools = [
        QueryEngineTool(
            query_engine=st.session_state.query_engine,
            metadata=ToolMetadata(
                name="Syllabus",
                description=(
                    "Provides information about the course, such as administrative issues, " +  
                    "bibliography, schedules, and important stuff to pass. "
                ),
            ),
        )
    ]
    react_formatter = ReActChatFormatter()
    react_formatter.system_header = CUSTOM_REACT_CHAT_SYSTEM_HEADER
    st.session_state.agent = ReActAgent.from_tools(
        query_engine_tools, 
        llm=Gemini(api_key=GOOGLE_AI_STUDIO), 
        verbose=True,
        react_chat_formatter=react_formatter
        )
    # print(st.session_state.chat_engine.agent_worker.__dict__)
    # print()
    # print()
    # # print(st.session_state.chat_engine.agent_worker.tools)
    # print(st.session_state.chat_engine.agent_worker._llm.__dict__)
    # print(st.session_state.chat_engine.agent_worker._react_chat_formatter.system_header)
    # print()
    # print()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = "😸" if message["role"] == "model" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


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
        full_response = st.session_state.agent.chat(prompt)
        # streaming_response = st.session_state.chat_engine.stream_chat(prompt)
        # for response_chunk in streaming_response.async_response_gen():
        #     full_response += response_chunk.text
        #     message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "model", "content": full_response})

    