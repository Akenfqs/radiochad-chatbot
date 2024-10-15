# 019. Updated AI-Powered Podcast Chatbot App with OpenAI v1.0.0+
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from openai import OpenAI
from typing import List

# Initialize the OpenAI client
openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

def chunk_transcriptions(transcriptions, chunk_size=1000, overlap=100):
    chunked_texts = []
    for transcript in transcriptions:
        words = transcript.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunked_texts.append(chunk)
    return chunked_texts

def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings

def setup_vector_db(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def chatbot_query(query: str, index, chunks: List[str], model, k: int = 5) -> str:
    # Retrieve relevant chunks
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    # Prepare the prompt for GPT
    prompt = f"""Based on the following excerpts from a podcast transcript, please answer the question: "{query}"

Relevant excerpts:
{' '.join(relevant_chunks)}

Please provide a concise and informative answer based on the information in these excerpts. If the information to answer the question is not present in the excerpts, please state that clearly."""

    # Generate response using GPT
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # You can use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about a podcast based on its transcripts."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150  # Adjust as needed
    )
    
    return completion.choices[0].message.content

def process_new_episodes(new_transcripts, existing_chunks, existing_embeddings, model, index):
    new_chunks = chunk_transcriptions(new_transcripts)
    all_chunks = existing_chunks + new_chunks
    
    new_embeddings = create_embeddings(new_chunks)
    all_embeddings = np.vstack([existing_embeddings, new_embeddings])
    
    index.add(new_embeddings)
    
    return all_chunks, all_embeddings

def process_transcripts(transcript_folder):
    transcripts = []
    for filename in os.listdir(transcript_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(transcript_folder, filename), 'r', encoding='utf-8') as f:
                transcripts.append(f.read())
    
    chunks = chunk_transcriptions(transcripts)
    embeddings = create_embeddings(chunks)
    index = setup_vector_db(embeddings)
    
    return chunks, embeddings, index

# 023. Real Chat-like Interface for Podcast Chatbot
def main():
    st.title("Radio Chad Chatbot")

    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load or process data
    data_file = 'chatbot_data.pkl'
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            chunks, embeddings, index = pickle.load(f)
    else:
        with st.spinner("Processing transcripts... This may take a few minutes."):
            transcript_folder = 'transcripts'
            chunks, embeddings, index = process_transcripts(transcript_folder)
            with open(data_file, 'wb') as f:
                pickle.dump((chunks, embeddings, index), f)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a container for chat history
    chat_container = st.container()

    # Create a container for the input field at the bottom
    input_container = st.container()

    # Use the input container to display the input field at the bottom
    with input_container:
        user_query = st.text_input("Ask a question about the podcast:")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Process the user query
    if user_query:
        with st.spinner('Generating response...'):
            response = chatbot_query(user_query, index, chunks, model)
        
        # Add new Q&A to chat history
        st.session_state.chat_history.append((user_query, response))

    # Display chat history in the chat container
    with chat_container:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Toi :** *{question}*")
            st.markdown(f"**Chad Bot:** {answer}")
            st.markdown("---")

    # Scroll to the bottom of the page
    st.markdown('<script>window.scrollTo(0,document.body.scrollHeight);</script>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()