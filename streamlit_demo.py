import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

# API endpoints
API_URL = "http://localhost:8080"
ADD_URL = f"{API_URL}/add"
SEARCH_URL = f"{API_URL}/search"
HEALTH_URL = f"{API_URL}/health"

st.set_page_config(
    page_title="Two-Tower Search Demo",
    page_icon="🔍",
    layout="wide"
)

# App title
st.title("Two-Tower Search Demo")
st.markdown("Semantic search powered by the Two-Tower model")

# Check if API is available
try:
    response = requests.get(HEALTH_URL, timeout=5)
    health_info = response.json()
    if health_info.get("status") == "healthy":
        st.sidebar.success("API is healthy ✅")
    else:
        st.sidebar.error("API is not healthy ❌")
except Exception as e:
    st.sidebar.error(f"API is not available: {str(e)}")
    st.error("Cannot connect to the API. Please make sure your Docker services are running.")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["🔍 Search", "➕ Add Documents"])


# Search Tab
with tab1:
    st.header("Semantic Search")
    
    search_query = st.text_input("Enter your search query", placeholder="e.g., vector similarity search")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)
    
    if st.button("Search") and search_query:
        with st.spinner("Searching..."):
            try:
                payload = {"text": search_query, "top_k": top_k}
                response = requests.post(SEARCH_URL, json=payload)
                result = response.json()
                
                if "results" in result and result["results"]:
                    st.success(f"Found {len(result['results'])} documents")
                    
                    # Display results
                    for i, doc in enumerate(result["results"]):
                        st.markdown(f"**Result {i+1}:** {doc['id']}")
                        st.markdown(f"**Text:** {doc['text']}")
                        st.markdown(f"**Similarity score:** {doc['score']:.4f}")
                        st.markdown("---")
                    
                    # Create a simple bar chart of scores
                    fig, ax = plt.subplots()
                    doc_ids = [doc["id"] for doc in result["results"]]
                    scores = [doc["score"] for doc in result["results"]]
                    ax.barh(doc_ids, scores)
                    ax.set_xlabel("Score (lower is better)")
                    ax.set_title("Document Similarity Scores")
                    st.pyplot(fig)
                else:
                    st.warning("No results found.")
            except Exception as e:
                st.error(f"Error during search: {str(e)}")


# Add Documents Tab
with tab2:
    st.header("Add Documents to Search Index")
    
    # Sample document sets
    sample_sets = {
        "Machine Learning": [
            "Deep learning techniques revolutionized natural language processing",
            "Neural networks can learn complex patterns in data",
            "Transformer models have advanced the state of text understanding",
            "Vector embeddings enable semantic search capabilities",
            "Machine learning algorithms improve with more training data"
        ],
        "Software Engineering": [
            "Code refactoring improves maintainability without changing behavior",
            "Unit tests ensure components work as expected",
            "Continuous integration automates the build and test process",
            "Version control systems track changes to source code",
            "Design patterns provide reusable solutions to common problems"
        ]
    }
    
    input_method = st.radio("Add documents by:", ["Text Input", "Sample Sets"])
    
    if input_method == "Text Input":
        documents_text = st.text_area(
            "Enter documents (one per line)",
            height=200,
            placeholder="Enter each document on a new line..."
        )
        documents = [doc.strip() for doc in documents_text.split("
") if doc.strip()]
    else:
        selected_set = st.selectbox("Choose a sample set:", list(sample_sets.keys()))
        documents = sample_sets[selected_set]
        st.write("Preview:")
        for doc in documents:
            st.markdown(f"- {doc}")
    
    if st.button("Add Documents") and documents:
        with st.spinner("Adding documents..."):
            try:
                payload = {"texts": documents}
                response = requests.post(ADD_URL, json=payload)
                result = response.json()
                if result.get("success", False):
                    st.success(f"Successfully added {result.get('added', 0)} documents")
                    st.write(f"Document IDs: {', '.join(result.get('ids', []))}")
                else:
                    st.error("Failed to add documents")
            except Exception as e:
                st.error(f"Error adding documents: {str(e)}")


# Sidebar info
st.sidebar.title("About")
st.sidebar.info("""This demo allows you to interact with a Two-Tower semantic search model deployed via Docker.

You can add documents to the vector database and search for semantically similar content.

The search works best when you have added several documents with varied content.""")
