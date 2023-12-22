import streamlit as st
import PyPDF2
from llama_index import Document, SimpleDirectoryReader
from llama_index import download_loader
from helpers.llama_agent_helper import create_agent_from_documents, DEFAULT_ACTIVITY_GOAL
from helpers.tru_helper import build_tru_recorder
from trulens_eval import Tru

st.title("Teacher Demo")

DEMO_ACTIVITY_GOAL = 'Reflect on your understanding on Linear Algebra before the next lecture.'

st.header("Use demo activity")
st.write(f"The goal for this demo activity is: '{DEMO_ACTIVITY_GOAL}'")
st.write("The demo will use some pre-loaded PDFs.")

if st.button("Set-up demo activity"):
    full_documents = []
    full_metadata = []
    for filename in ['lecture-2.pdf', 'syllabus.pdf']:
        documents = SimpleDirectoryReader(
            input_files=[f"PDFs/{filename}"],
            # required_exts=[".pdf"],
            # recursive=True,
        ).load_data()
        pdf_reader = PyPDF2.PdfReader(f"PDFs/{filename}")
        metadata = pdf_reader.metadata
        document = Document(text="\n\n".join([doc.text for doc in documents]))
        full_documents.append(document)
        full_metadata.append(metadata)
    with st.spinner('Indexing documents...'):
        st.session_state.activity_goal = DEMO_ACTIVITY_GOAL
        st.session_state.agent = create_agent_from_documents(full_documents, full_metadata, activity_goal=st.session_state.activity_goal)
        st.session_state.tru_student = build_tru_recorder(st.session_state.agent)
    st.success("Documents uploaded and indexed.")

st.divider()
st.header("Create custom activity")
activity_goal = st.text_input('Create an activity goal', placeholder=f"Ex: {DEFAULT_ACTIVITY_GOAL}")

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
    if st.button("Create activity"):
        with st.spinner('Indexing documents...'):
            st.session_state.activity_goal = activity_goal
            st.session_state.agent = create_agent_from_documents(documents, activity_goal=activity_goal)
            st.session_state.tru_student = build_tru_recorder(st.session_state.agent)

        st.success("Documents uploaded and indexed.")

st.divider()
st.header("Monitor Assistant Performance")
if "tru_student" in st.session_state:
    if st.button("Check Performance (TruLens)"):
        tru = Tru()
        records, _ = tru.get_records_and_feedback(app_ids=[])
        metric_cols_ix = records.columns.str.startswith("[METRIC]") & ~records.columns.str.endswith("_calls")
        metric_cols = records.columns[metric_cols_ix]
        mean_metrics = records[metric_cols].mean()

        # Show alerts for metrics that are below 0.3
        if mean_metrics['[METRIC] Answer Relevance'] < (1/3):
            st.markdown("🚨 **Low relevance of the assistant's answers.** The assistant may not have all the information needed to answer the question. You can try added more files to the activity.")
        
        if mean_metrics['[METRIC] Groundedness'] < (1/3):
            st.markdown("🚨 **Low groundedness of the assistant's answers.** The assistant may be hallucinating some facts. Try addressing them in class to avoid misconceptions.")
        
        if mean_metrics['[METRIC] Insensitivity'] > (2/3):
            st.markdown("🚨 **Insensitive answers from the assistant.** The assistant may be giving insensitive answers. In the activity goal you can try adding your desired tone for the bot (friendly, format, etc).")
        
        if mean_metrics['[METRIC] Input Maliciousness'] > (2/3):
            st.markdown("🚨 **Malicious input from the user detected.** The user may be trying to trick the assistant. You can modify the assisant's goal or address this issue in your class.")
        

        records['ts'] = records['ts'].apply(lambda x: x[:16])
        records['input'] = records['input'].apply(lambda x: eval(x)[0])
        records['output'] = records['output'].apply(lambda x: eval(x))
        config = {
            'input' : st.column_config.TextColumn('input', width="small"),
            'output' : st.column_config.TextColumn('output', width="small"),
        }
        for col in metric_cols:
            config[col] = st.column_config.TextColumn(col.replace('[METRIC] ', '').replace(' ', '\n'), width="small")
        records = records[["ts", "input", "output", *metric_cols]]
        records[metric_cols] = records[metric_cols].round(3)
        def color_code(val):
            if val < 0.3:
                color = '#d7481d'
            elif (0.3 <= val <= 0.6):
                color = '#fff321'
            else:
                color = '#59f720'
            return f'background-color: {color}'

        # Apply color coding to the DataFrame
        styled_records = records.style.map(color_code, subset=metric_cols)
        styled_records = styled_records.map(lambda x: color_code(1 - x), subset=['[METRIC] Input Maliciousness', '[METRIC] Insensitivity'])

        st.dataframe(styled_records, use_container_width=True, column_config=config)


    

    

