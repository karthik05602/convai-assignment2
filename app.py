import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import pickle
import faiss
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --- 1. Cached Loading of All Models and Artifacts ---
from peft import PeftModel

@st.cache_resource
def load_resources(ft_repo_id):
    """Load all models and pre-built RAG artifacts from files and the Hub."""
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # --- Load RAG Artifacts (No changes here) ---
    try:
        faiss_index = faiss.read_index("faiss_index.bin")
        with open("bm25_index.pkl", "rb") as f:
            bm25_index = pickle.load(f)
        with open("chunks.json", "r") as f:
            chunks_with_metadata = json.load(f)
    except FileNotFoundError as e:
        st.error(f"A required artifact file was not found: {e}.")
        return None

    # --- RAG Model Loading (No changes here) ---
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    rag_generator = pipeline('text2text-generation', model="google/flan-t5-base") 
    
    # --- CORRECTED Fine-Tuned Model Loading ---
    ft_generator = None
    try:
        # Step 1: Load the original base model (GPT-2)
        base_model_name = "gpt2"
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Step 2: Load and apply your LoRA adapters from the Hub
        ft_model = PeftModel.from_pretrained(base_model, ft_repo_id)
        
        # Step 3: Use the same tokenizer as the base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token # IMPORTANT for GPT-2

        # Step 4: Create the pipeline with the combined model
        ft_generator = pipeline('text-generation', model=ft_model, tokenizer=tokenizer)
        
    except Exception as e:
        st.error(f"Could not load fine-tuned model from '{ft_repo_id}'. Error: {e}")

    return (
        embedding_model, rag_generator, faiss_index, bm25_index, chunks_with_metadata, 
        set(stopwords.words('english')), ft_generator
    )

# --- 2. Core RAG and FT Functions ---
def is_query_financially_relevant(query, stop_words):
    query_lower = query.lower()
    negative_keywords = ['ceo', 'stock price', 'who is', 'products', 'what are']
    if any(keyword in query_lower for keyword in negative_keywords): return False
    financial_keywords = ['revenue', 'income', 'asset', 'liability', 'equity', 'cash', 'profit',
        'loss', 'balance sheet', 'financial', 'cost', 'expense', 'debt',
        'rieter', 'inventories', 'receivables', 'cogs', 'payables', 'ebitda','goodwill','provisions']
    return any(keyword in query_lower for keyword in financial_keywords)

def preprocess_query(query, stop_words):
    query_lower = query.lower()
    tokens = word_tokenize(query_lower)
    return query_lower, [word for word in tokens if word.isalnum() and word not in stop_words]

def hybrid_retrieval(query, top_n=3):
    processed_query, tokenized_query = preprocess_query(query, stop_words)
    query_embedding = embedding_model.encode(processed_query)
    distances, dense_indices = faiss_index.search(np.array([query_embedding]), top_n)
    dense_results = [{"id": int(i), "score": 1/(1+d)} for i, d in zip(dense_indices[0], distances[0])]
    bm25_scores = bm25_index.get_scores(tokenized_query)
    sparse_indices = np.argsort(bm25_scores)[::-1][:top_n]
    sparse_results = [{"id": int(i), "score": bm25_scores[i]} for i in sparse_indices]
    combined = {res['id']: res['score'] for res in dense_results + sparse_results}
    retrieved_chunks = [chunks[idx] for idx in combined.keys()]
    return retrieved_chunks, combined

def merge_adjacent_chunks(retrieved_chunks):
    if not retrieved_chunks: return ""
    retrieved_chunks.sort(key=lambda x: x['id'])
    return " ".join(c['text'] for c in retrieved_chunks)

def answer_query_with_rag(query):
    start_time = time.time()
    if not is_query_financially_relevant(query, stop_words):
        return "This question is not relevant to the financial documents.", 0.0, 0.0
    retrieved, scores = hybrid_retrieval(query)
    context = merge_adjacent_chunks(retrieved)
    if not context: return "Could not find relevant information.", 0.0, time.time() - start_time
    
    prompt = f"Based on the context, provide a short, direct answer to the question.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = rag_generator(prompt, max_new_tokens=60, repetition_penalty=1.2)[0]['generated_text']
    
    duration = time.time() - start_time
    confidence = np.mean(list(scores.values())) if scores else 0
    return answer, confidence, duration

def answer_query_with_ft(query):
    start_time = time.time()
    if not is_query_financially_relevant(query, stop_words):
        return "This question is not relevant to the financial documents.", 0.0, 0.0
    if not ft_generator:
        return "Fine-tuned model is not available.", 0.0, 0.0
    
    prompt = f"Question: {query}\nAnswer:"
    response = ft_generator(prompt, max_new_tokens=50, pad_token_id=ft_generator.tokenizer.eos_token_id)
    answer = response[0]['generated_text'].replace(prompt, "").strip()
    
    duration = time.time() - start_time
    return answer, 0.95, duration # Use a fixed confidence for FT model

# --- 3. Streamlit User Interface ---
st.set_page_config(page_title="Financial Q&A Bot", layout="wide")
st.title("📈 Financial Report Q&A Bot")

# IMPORTANT: Replace "YourUsername/..." with the actual repo ID of your fine-tuned model on the Hub
FT_REPO_ID = "2023ac05602/gpt2-lora-financial-qa"

# Load all resources
resources = load_resources(FT_REPO_ID)
if resources:
    embedding_model, rag_generator, faiss_index, bm25_index, chunks, stop_words, ft_generator = resources
    
    st.sidebar.header("Controls")
    mode = st.sidebar.selectbox("Select Mode", ["RAG", "Fine-Tuned"])
    st.sidebar.info(
        "**RAG Mode**: Answers using a general model by retrieving relevant document sections.\n\n"
        "**Fine-Tuned Mode**: Answers using a GPT-2 model specially trained on this document's Q&A."
    )

    user_query = st.text_input("Enter your question:", "What were the total assets in 2023?")

    if st.button("Get Answer"):
        if user_query:
            if mode == "RAG":
                with st.spinner("Analyzing documents with RAG..."):
                    # --- ADD THIS DEBUGGING BLOCK ---
                    retrieved, scores = hybrid_retrieval(user_query)
                    context_in_streamlit = merge_adjacent_chunks(retrieved)
                
                    with st.expander("Click to see the context sent to the model"):
                        st.text_area("Context:", context_in_streamlit, height=200)
                # --- END DEBUGGING BLOCK ---
                    answer, confidence, duration = answer_query_with_rag(user_query)
            else: # Fine-Tuned Mode
                with st.spinner("Querying the Fine-Tuned model..."):
                    answer, confidence, duration = answer_query_with_ft(user_query)
            
            st.subheader("Answer")
            st.markdown(f"> {answer}")
            
            st.subheader("Response Details")
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Score", f"{confidence:.4f}")
            col2.metric("Response Time", f"{duration:.2f}s")
            col3.metric("Method Used", mode)
        else:
            st.warning("Please enter a question.")
