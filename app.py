import streamlit as st
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import SummaryIndex
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from io import BytesIO

# Apply async support
nest_asyncio.apply()

# Set up LLM and embeddings
GROQ_API_KEY = "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"
llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding()

# Streamlit UI
st.title("Document Summarizer")
st.write("Upload a PDF document to generate a summary.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Use BytesIO to keep the file in memory
    file_buffer = BytesIO(uploaded_file.read())
    
    # Read document and prepare for summarization directly from in-memory BytesIO object
    documents = SimpleDirectoryReader(input_files=[file_buffer]).load_data()
    splitter = SentenceSplitter(chunk_size=2048)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Build summarization index
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    
    # Run summarization query
    with st.spinner("Generating summary..."):
        response = summary_query_engine.query("Summarize the given document")
    
    # Display the result
    st.subheader("Document Summary")
    st.write(response)
