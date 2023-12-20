import streamlit as st
import PyPDF2
from llama_index import Document, SimpleDirectoryReader
from llama_index import download_loader
from utils import create_agent_from_documents

from trulens_eval import Tru, Feedback, TruLlama, OpenAI as fOpenAI
from trulens_eval.feedback import Groundedness
import openai
import numpy as np
openai.api_key = st.secrets["OPENAI_API_KEY"]

tru = Tru()

def build_tru_recorder(agent):
    provider = fOpenAI()
    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()

    context_selection = TruLlama.select_source_nodes().node.text
    f_qs_relevance = (
        Feedback(provider.qs_relevance_with_cot_reasons,
                name="Context Relevance")
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    )

    grounded = Groundedness(groundedness_provider=provider)
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons,
                name="Groundedness"
                )
        .on(context_selection)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    tru_recorder = TruLlama(
        agent,
        app_id="Students Agent",
        feedbacks=[
            f_qa_relevance,
            f_qs_relevance,
            f_groundedness
        ]
    )
    return tru_recorder


st.title("Teacher Demo")

if st.button("Use demo documents."):
    full_documents = []
    for filename in ['lecture-1.pdf', 'lecture-2.pdf', 'syllabus.pdf']:
        documents = SimpleDirectoryReader(
            input_files=[f"PDFs/{filename}"],
            # required_exts=[".pdf"],
            # recursive=True,
        ).load_data()
        document = Document(text="\n\n".join([doc.text for doc in documents]))
        full_documents.append(document)
    with st.spinner('Indexing documents...'):
        st.session_state.agent = create_agent_from_documents(full_documents)
        st.session_state.tru_student = build_tru_recorder(st.session_state.agent)
    st.success("Documents uploaded and indexed.")

# Add documents with streamlit
uploaded_files= st.file_uploader("Upload a document", type=["pdf"], accept_multiple_files=True)
if len(uploaded_files):
    for uploaded_file in uploaded_files:
        print(uploaded_file)
        # Process the uploaded file using PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        metadata = pdf_reader.metadata

        st.write("Metadata:")
        #st.write(metadata)
        st.write(metadata.title)

        # Process the uploaded file using llama_index (https://llamahub.ai/l/file-pdf?from=all)
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        documents = loader.load_data(file=uploaded_file)


    # Add submit button
    if st.button("Submit documents"):
        with st.spinner('Indexing documents...'):
            st.session_state.agent = create_agent_from_documents(documents)
            st.session_state.tru_student = build_tru_recorder(st.session_state.agent)

        st.success("Documents uploaded and indexed.")

if "tru_student" in st.session_state:
    if st.button("Check Performance (TruLens)"):
        tru = Tru()
        records, _ = tru.get_records_and_feedback(app_ids=[])
        st.write(records)

    

    

