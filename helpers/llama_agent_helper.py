import streamlit as st
from llama_index.llms import Gemini
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings import GeminiEmbedding
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.agent.react.formatter import ReActChatFormatter
from prompts.react_prompt import CUSTOM_REACT_CHAT_SYSTEM_HEADER
from llama_index.text_splitter import SentenceSplitter
from llama_index.ingestion import IngestionPipeline
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from gemini_config import SAFETY_SETTINGS, DEFAULT_TEMPERATURE


GOOGLE_AI_STUDIO = st.secrets["GEMINI_API_KEY"]
llm = Gemini(api_key=GOOGLE_AI_STUDIO, temperature=DEFAULT_TEMPERATURE, safety_settings=SAFETY_SETTINGS)


def create_query_engine(documents):
    # Create pipeline
    text_splitter = SentenceSplitter(
        chunk_size=512, chunk_overlap=128
    )

    extractors = [
        #TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
        #SummaryExtractor(summaries=["prev", "self"], llm=llm),
        KeywordExtractor(keywords=3, llm=llm),
    ]

    transformations = [text_splitter] + extractors
    pipeline = IngestionPipeline(transformations=transformations)

    # Pass documents through pipeline
    document_nodes = pipeline.run(documents=documents)

    # Create vector store with Gemini embeddings
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=GOOGLE_AI_STUDIO
    )
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, text_splitter=text_splitter
    )

    index = VectorStoreIndex(
        nodes=document_nodes,
        service_context=service_context
    )

    # Create query engine
    query_engine = index.as_query_engine(
        service_context=service_context,
        verbose=True,
        similarity_top_k=4
    )
    return query_engine

def create_default_query_engine_tool(query_engine):
    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="course_info",
                description=(
                    "Access relevant documents from the course, such as the syllabus, " +  
                    "bibliography, and lecture notes." + 
                    "Requires the input parameter be a phrase summarizing the information to be retreived."
                ),
            ),
        )
    ]
    return query_engine_tools

def create_agent_from_tools(tools):
    react_formatter = ReActChatFormatter(system_header=CUSTOM_REACT_CHAT_SYSTEM_HEADER)
    agent = ReActAgent.from_tools(
        tools, 
        llm=llm, 
        verbose=True,
        react_chat_formatter=react_formatter
    )
    return agent

def create_agent_from_documents(documents):
    query_engine = create_query_engine(documents)
    query_engine_tools = create_default_query_engine_tool(query_engine)
    agent = create_agent_from_tools(query_engine_tools)
    return agent