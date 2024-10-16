# 009. Comprehensive AI-Powered Podcast Chatbot with OpenAI Embeddings

import os
import pickle
import time
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI
from typing import List, Dict
import logging
import tiktoken

# Try to import NLTK, use a fallback if not available
try:
    import nltk
    nltk.download('punkt', quiet=True)
    use_nltk = True
except ImportError:
    use_nltk = False
    print("NLTK not available. Using a simple fallback for sentence tokenization.")

# Fallback sentence tokenization function
def simple_sentence_tokenize(text):
    return text.replace('!', '.').replace('?', '.').split('.')

# Initialize the OpenAI client
openai_api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=openai_api_key)

# 019. Improved Transcript Processing with Multiple Encoding Support
def process_transcripts(transcript_folder):
    transcripts = []
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'ascii']
    
    for filename in os.listdir(transcript_folder):
        file_path = os.path.join(transcript_folder, filename)
        if os.path.isfile(file_path):
            # Extract date and topic from filename, if possible
            parts = filename.rsplit('.', 1)[0].split('_', 1)
            if len(parts) == 2:
                date, topic = parts
            else:
                date = "Unknown"
                topic = parts[0]

            content = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break  # If successful, break the loop
                except UnicodeDecodeError:
                    continue  # Try the next encoding

            if content is None:
                logging.warning(f"Could not decode file {filename} with any of the attempted encodings. Skipping this file.")
                continue

            transcripts.append({
                'date': date,
                'topic': topic,
                'content': content,
                'type': 'txt' if filename.endswith('.txt') else 'srt'
            })
    
    return transcripts

# 024. Optimized Chunk Transcriptions Function
def chunk_transcriptions(transcripts, chunk_size=250, overlap=25):
    chunked_texts = []
    for transcript in transcripts:
        if use_nltk:
            sentences = nltk.sent_tokenize(transcript['content'])
        else:
            sentences = simple_sentence_tokenize(transcript['content'])
        current_chunk = []
        current_chunk_size = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_length = len(sentence.split())
            if current_chunk_size + sentence_length > chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunked_texts.append({
                        'date': transcript['date'],
                        'topic': transcript['topic'],
                        'content': chunk_text,
                        'type': transcript['type']
                    })
                    # Keep the last 'overlap' words for the next chunk
                    overlap_words = current_chunk[-overlap:]
                    current_chunk = overlap_words
                    current_chunk_size = len(" ".join(overlap_words).split())
            current_chunk.append(sentence)
            current_chunk_size += sentence_length
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunked_texts.append({
                'date': transcript['date'],
                'topic': transcript['topic'],
                'content': chunk_text,
                'type': transcript['type']
            })
    return chunked_texts

# 025. Robust Create Embeddings Function
def create_embeddings(chunks, batch_size=32, max_retries=3):
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=[chunk['content'] for chunk in batch]
                )
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if "maximum context length" in str(e):
                    if batch_size > 1:
                        # If batch size is greater than 1, reduce it and retry
                        batch_size = max(1, batch_size // 2)
                        logging.warning(f"Reducing batch size to {batch_size} due to token limit error.")
                    else:
                        # If batch size is already 1, skip this chunk
                        logging.error(f"Skipping chunk due to token limit error: {str(e)}")
                        break
                else:
                    logging.error(f"Error in creating embeddings: {str(e)}")
                    retry_count += 1
        if retry_count == max_retries:
            logging.error(f"Failed to create embeddings for batch after {max_retries} retries.")
    return all_embeddings

# 016. Updated FAISS Index Setup Function
def setup_vector_db(embeddings):
    dimension = len(embeddings[0])  # Get the dimension from the first embedding
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

# 022. Optimized Token-Aware Chatbot Query Function

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chatbot_query(query: str, index, chunks: List[Dict], k: int = 5, conversation_history: str = "") -> str:
    query_embedding = client.embeddings.create(
        model="text-embedding-3-large",
        input=[query]
    ).data[0].embedding
    
    # Ensure the query embedding has the same dimension as the index
    if len(query_embedding) != index.d:
        raise ValueError(f"Query embedding dimension ({len(query_embedding)}) does not match index dimension ({index.d})")
    
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    context_parts = []
    total_tokens = 0
    max_context_tokens = 6000  # Reduced from 7000 to leave more room for other parts

    for chunk in relevant_chunks:
        chunk_text = f"Date: {chunk.get('date', 'N/A')}\nTopic: {chunk.get('topic', 'N/A')}\nContent: {chunk.get('content', 'N/A')}"
        chunk_tokens = num_tokens_from_string(chunk_text)
        if total_tokens + chunk_tokens > max_context_tokens:
            break
        context_parts.append(chunk_text)
        total_tokens += chunk_tokens
    
    context = "\n\n".join(context_parts)
    
    system_message = "You are a knowledgeable and enthusiastic assistant for the Radio Chad podcast. You provide accurate, engaging, and context-aware responses about the podcast content."
    
    user_message = f"""Answer the following question about the Radio Chad podcast based on the provided transcript excerpts:

Question: {query}

Relevant excerpts:
{context}

Recent conversation history:
{conversation_history}

Guidelines:
1. Provide a concise, informative, and engaging answer based on the information in the excerpts.
2. If the information is not in the excerpts, clearly state that and offer general podcast information if available.
3. Use dates and topics from the excerpts when mentioning specific episodes.
4. Suggest related topics or episodes if appropriate.
5. Maintain a friendly, conversational tone.
6. If you need more information, ask a follow-up question.

Please formulate your response now."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    total_tokens = num_tokens_from_string(system_message) + num_tokens_from_string(user_message)
    
    if total_tokens > 8000:
        logging.warning(f"Total tokens ({total_tokens}) exceed limit. Truncating context.")
        while total_tokens > 8000:
            context_parts.pop()
            context = "\n\n".join(context_parts)
            user_message = user_message.replace("{context}", context)
            total_tokens = num_tokens_from_string(system_message) + num_tokens_from_string(user_message)
        messages[1]["content"] = user_message

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300
        )
        
        response = completion.choices[0].message.content
        
        # Check if the response contains a follow-up question
        if "?" in response and not response.endswith("?"):
            parts = response.split("?", 1)
            return f"{parts[0]}?\n\nTo provide a more complete answer, I need some additional information. {parts[1]}"
        
        return response
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {str(e)}")
        return f"I apologize, but I encountered an error while processing your request. Error: {str(e)}"

def process_new_episodes(new_transcripts, existing_chunks, existing_embeddings, index):
    new_chunks = chunk_transcriptions(new_transcripts)
    all_chunks = existing_chunks + new_chunks
    
    new_embeddings = create_embeddings(new_chunks)
    all_embeddings = np.vstack([existing_embeddings, new_embeddings])
    
    index.add(new_embeddings.astype('float32'))
    
    return all_chunks, all_embeddings

# 018. Enhanced Main Function with Error Handling and Logging
logging.basicConfig(level=logging.INFO)

def main():
    st.title("Radio Chad Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load or process data
    data_file = 'chatbot_data.pkl'
    rebuild = False  # Set this to True to force a rebuild of embeddings

    try:
        if os.path.exists(data_file) and not rebuild:
            with open(data_file, 'rb') as f:
                chunks, embeddings, index = pickle.load(f)
            logging.info("Loaded existing data from chatbot_data.pkl")
        else:
            with st.spinner("Processing transcripts and creating embeddings... This may take a few minutes."):
                transcript_folder = 'transcripts'
                transcripts = process_transcripts(transcript_folder)
                logging.info(f"Processed {len(transcripts)} transcripts")
                chunks = chunk_transcriptions(transcripts)
                logging.info(f"Created {len(chunks)} chunks")
                embeddings = create_embeddings(chunks)
                logging.info(f"Created {len(embeddings)} embeddings")
                if len(embeddings) > 0:
                    index = setup_vector_db(embeddings)
                    logging.info("Set up FAISS index")
                    with open(data_file, 'wb') as f:
                        pickle.dump((chunks, embeddings, index), f)
                    logging.info("Saved new data to chatbot_data.pkl")
                else:
                    raise ValueError("No embeddings were created. Please check your data and try again.")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What would you like to know about the podcast?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Pass the entire conversation history to the chatbot_query function
                    conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])  # Last 5 messages
                    assistant_response = chatbot_query(prompt, index, chunks, conversation_history=conversation_history)
                    
                    # Simulate stream of response with milliseconds delay
                    for chunk in assistant_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    message_placeholder.markdown(error_message)
                    full_response = error_message
                    logging.error(f"Error in chatbot_query: {str(e)}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")
        st.error(f"An error occurred while processing the transcripts: {str(e)}")
        return

if __name__ == "__main__":
    main()