import streamlit as st
from llama_index.llms import Gemini
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings import GeminiEmbedding
from llama_index.tools import QueryEngineTool, ToolMetadata, FunctionTool
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


DEFAULT_ACTIVITY_GOAL = "Help the student reflect on their study habits and plan for future study session."
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.question_gen.llm_generators import LLMQuestionGenerator

GOOGLE_AI_STUDIO = st.secrets["GEMINI_API_KEY"]
llm = Gemini(api_key=GOOGLE_AI_STUDIO, temperature=DEFAULT_TEMPERATURE, safety_settings=SAFETY_SETTINGS)


def create_query_engine(documents):
    # Create pipeline
    text_splitter = SentenceSplitter(
        chunk_size=256, chunk_overlap=64
    )

    extractors = [
        #TitleExtractor(nodes=5, llm=llm),
        #QuestionsAnsweredExtractor(questions=3, llm=llm),
        SummaryExtractor(summaries=["prev", "self"], llm=llm),
        #KeywordExtractor(keywords=3, llm=llm),
    ]

    transformations = [text_splitter] + extractors
    pipeline = IngestionPipeline(transformations=transformations)

    # Pass documents through pipeline
    document_nodes = pipeline.run(documents=[documents])

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
        similarity_top_k=5
    )
    return query_engine, document_nodes


def create_default_query_engine_tool(query_engine, document_nodes, metadata):
    query_engine_tool = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="_".join(metadata.title.split(" ")),
                description=(
                    f"Access relevant documents from the course, this one is summarized as: {document_node.metadata['section_summary']}." + 
                    "Use information from this document if the summary has topics of the question." + 
                    "Requires the input parameter be a phrase summarizing the information to be retreived."
                ),
            ),
        ) for document_node in document_nodes
    ]
    return query_engine_tool


def create_default_educational_tools(activity_goal=None):
    activity_goal = activity_goal or DEFAULT_ACTIVITY_GOAL

    GREETING_PROMPT = (
        "You are an assistant for students called SIMBA (Student Interactive Mentoring Bot Assistant)." +
        "You are here to help students with their study habits and provide advice on how to improve them." +
        f"In particular, you must help them complete the following task: '{activity_goal}'" +
        "\n"
        "Create a very short message to greet the student and introduce yourself." +
        "Start your message with 'Hi! 😸 I am SIMBA, your educational assistant.'" +
        f"At the end, include a question to kickstart the conversation towards the completing the task: '{activity_goal}'" 
    )

    greeting_tool = FunctionTool.from_defaults(
        fn=lambda input: f"Use EXACTLY this message to greet the student: ''{llm.complete(GREETING_PROMPT)}''. Do not use any more tools, you are now ready to respond. PLEASE DO NOT CHANGE THE MESSAGE.",
        name="greet_student",
        description=(
            "Create an appropiate message to introduce yourself to the student." +
            "Always use this tool to answer to messages like: 'Hi!' or 'Hello!'" + 
            "Follow the output of this tool exactly as it says."
        )
    )


    EXPERT_PROMPT = (
        "You are an expert in providing educational advice. Use your knowledge in self-regulated learning to provide support." +
        "You will receive a question from another tutor about how to support the students" + 
        "You must answer informatively and with detail. Start your answer with 'I suggest...'" + 
        "If more context is needed (e.g., information from the course syllabus or academic material), follow the next steps:" +
        "In this case, include in the answer 'It would be good to have information about...', then follow one of the following two:" +
        "If you need context from the specific from course (e.g., syllabus or academic material) also include: 'Use another tool to get this context, DO NOT ask the user'."+
        "If you need context of the student's emotions (e.g., motivation, frustrations) or want to clarify something (e.g., asking to go more in depth), also include: 'Ask the student for this information, DO NOT use another tool'."+
        "Finally, after asking for the context, finish with 'Come back when you have this information.'"
        "\nINPUT: {input}"
    )
    expert_tool = FunctionTool.from_defaults(
        fn=lambda input: llm.complete(EXPERT_PROMPT.format(input=input)),
        name="educational_advice",
        description=(
            "Ask for advice from an expert educational assistant. They can help with student planning, goal setting, and reflection." + 
            "They do not have access to the conversation or course context, so be as detailed as possible in your question.")
    )
    return [expert_tool, greeting_tool]


def create_agent_from_tools(tools, activity_goal=None):
    activity_goal = activity_goal or DEFAULT_ACTIVITY_GOAL
    system_header = CUSTOM_REACT_CHAT_SYSTEM_HEADER.format(teacher_goal=activity_goal)
    react_formatter = ReActChatFormatter(system_header=system_header)
    agent = ReActAgent.from_tools(
        tools, 
        llm=llm, 
        verbose=True,
        react_chat_formatter=react_formatter
    )
    return agent


def create_agent_from_documents(documents, metadata, activity_goal=None):
    query_engine_tools = []
    for i, document in enumerate(documents):
        query_engine, document_nodes = create_query_engine(document)
        query_engine_tools += create_default_query_engine_tool(query_engine, document_nodes, metadata[i])
    educational_tools = create_default_educational_tools(activity_goal=activity_goal)    
    agent = create_agent_from_tools(educational_tools + query_engine_tools, activity_goal=activity_goal)

    return agent