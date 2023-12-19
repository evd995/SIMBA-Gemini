import streamlit as st
from llama_index import download_loader
import PyPDF2

st.title("Teacher Demo")

# Add documents with streamlit
uploaded_files= st.file_uploader("Upload a document", type=["pdf"], accept_multiple_files=True)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        print(uploaded_file)
        # Process the uploaded file using PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        metadata = pdf_reader.metadata

        st.write("Metadata:")
        st.write(metadata)
        st.write(metadata.title)

        # Process the uploaded file using llama_index (https://llamahub.ai/l/file-pdf?from=all)
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        documents = loader.load_data(file=uploaded_file)
        print(documents[:3])
        st.write(documents[:3])

    

