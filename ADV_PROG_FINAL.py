import streamlit as st
import threading
import logging
import requests
import shutil
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import base64
import json

from PIL import Image
from bs4 import BeautifulSoup

# Imports for LangChain chains and text splitting
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Imports for vector store and models
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from serpapi import GoogleSearch

# Imports for Telegram bot
from telegram.ext import Updater, CommandHandler

import chromadb
import ollama

logging.basicConfig(level=logging.INFO)
chromadb.api.client.SharedSystemClient.clear_system_cache()

# --- Keys and Constants ---
SERPAPI_KEY = "b04ea417f4b8624e5f6a579489d6ccf52c33827f02852a079c7d47b757e3a4a3"
TELEGRAM_BOT_TOKEN = "8118108907:AAGDX4XtmSqozlybU_yNREQn9m219dgXJoA"
TELEGRAM_CHAT_ID = "1142604273"
HISTORY_FILE = "chat_history.txt"

# --- Sidebar Settings ---
st.sidebar.title("Settings")
selected_model = st.sidebar.radio("Select LLaMA Version", ["llama3.1", "llama3.2"])
search_mode = st.sidebar.radio("Search Mode", ["Web Search", "Document Search"])

uploaded_file = st.sidebar.file_uploader("Upload Document (TXT, PDF, DOC/DOCX)", type=["txt", "pdf", "doc", "docx"])
docs_folder = "docs"
if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)
if uploaded_file is not None:
    file_path = os.path.join(docs_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")

# --- Navigation ---
page = st.sidebar.radio("Select Section", ["Query", "Chat with Ollama"])

# --- App Title ---
st.title("Chatbot with Web/Document Search, Image Analysis and Telegram Callback")

# --- Initialize Embeddings ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Initialize Memory for Web Search/Document Processing ---
if "short_term_memory" not in st.session_state:
    st.session_state["short_term_memory"] = []  # Short-term memory
if "long_term_memory" not in st.session_state:
    st.session_state["long_term_memory"] = []   # Long-term memory

# --- Helper: Check for Profanity ---
def contains_profanity(text):
    banned_words = [
        "fuck", "fucking", "shit", "bitch", "asshole", "dick", "cunt", "motherfucker",
        "crap", "damn", "bastard", "son of a bitch", "bollocks", "arsehole", "screw", "piss"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in banned_words)

# --- Common Functions (Web Search, Documents, Telegram, etc.) ---
def clear_chroma_db():
    chroma_path = ".chroma_db"
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
            logging.info("ChromaDB cache cleared successfully!")
        except Exception as e:
            logging.error(f"Error clearing ChromaDB: {e}")
    try:
        _ = Chroma(embedding_function=embeddings, persist_directory=".chroma_db")
        logging.info("ChromaDB in-memory instance created for reset")
    except Exception as e:
        logging.error(f"Failed to create in-memory Chroma instance: {e}")

def fetch_web_content(query, start=0, max_results=10):
    params = {"engine": "google", "q": query, "api_key": SERPAPI_KEY, "start": start}
    search = GoogleSearch(params)
    results = search.get_dict()
    processed_results = []
    seen_links = set()
    for result in results.get("organic_results", []):
        title = result.get("title", "").strip()
        link = result.get("link", "").strip()
        logging.info(f"Processing link: {link}")
        if link in seen_links or not link or not title:
            continue
        try:
            response = requests.get(link, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = ' '.join([p.text for p in soup.find_all('p')])
                headers = ' '.join([h.text for h in soup.find_all(['h1', 'h2', 'h3', 'h4'])])
                content = headers + "\n" + paragraphs
                if content.strip():
                    processed_results.append((title, link, content))
                    seen_links.add(link)
            if len(processed_results) >= max_results:
                break
        except Exception as e:
            logging.error(f"Error fetching {link}: {e}")
    return processed_results

def fetch_image_urls(query, max_results=3):
    params = {"engine": "google_images", "q": query, "api_key": SERPAPI_KEY}
    search = GoogleSearch(params)
    results = search.get_dict()
    return [img.get("original") for img in results.get("images_results", [])[:max_results]]

def image_to_base64(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return "data:image/png;base64," + img_str
    except Exception as e:
        logging.error(f"Error processing image from {image_url}: {e}")
        return None

def analyze_images(image_urls):
    results = []
    for url in image_urls:
        img_base64 = image_to_base64(url)
        if img_base64:
            try:
                response = ollama.chat(model="llava", messages=[{
                    "role": "user",
                    "content": "Please provide a detailed description of this image. Include any visible text, colors, objects, and context if available.",
                    "images": [img_base64]
                }])
                logging.info(f"Image analysis response for {url}: {response}")
                if "message" in response and "content" in response["message"]:
                    results.append((url, response["message"]["content"]))
                else:
                    results.append((url, f"Image analysis failed for {url}: Unexpected response format"))
            except Exception as e:
                logging.error(f"Error analyzing image {url}: {e}")
                results.append((url, f"Failed to analyze image: {url}"))
        else:
            results.append((url, f"Failed to load image: {url}"))
    return results

def load_documents():
    docs = []
    if os.path.exists(docs_folder):
        for filename in os.listdir(docs_folder):
            path = os.path.join(docs_folder, filename)
            if filename.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    docs.append(f.read())
            elif filename.endswith(".pdf"):
                try:
                    import PyPDF2
                    with open(path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        docs.append(text)
                except Exception as e:
                    logging.error(f"Error reading PDF {filename}: {e}")
            elif filename.endswith((".doc", ".docx")):
                try:
                    import docx
                    document = docx.Document(path)
                    text = "\n".join([para.text for para in document.paragraphs])
                    docs.append(text)
                except Exception as e:
                    logging.error(f"Error reading DOC/DOCX {filename}: {e}")
    return docs

def document_search(query, selected_model):
    docs = load_documents()
    if not docs:
        return "No documents uploaded."
    vectorstore = Chroma.from_texts(docs, embeddings, persist_directory=".doc_chroma_db")
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=10)
    threshold = 0.9
    retrieved_context = "\n".join([doc.page_content for doc, score in docs_with_scores if score >= threshold])
    llm_model = OllamaLLM(model=selected_model)
    prompt = f"Context:\n{retrieved_context}\n\nQuestion:\n{query}\n\nAnswer based only on the above context."
    response = llm_model(prompt)
    return response

profanity_template = "Check the following text for profanity. Answer 'Yes' if found, and 'No' if not. Text: {text}"
profanity_prompt = PromptTemplate(template=profanity_template, input_variables=["text"])
profanity_chain = LLMChain(llm=OllamaLLM(model=selected_model), prompt=profanity_prompt)

def send_telegram_notification(message: str):
    max_length = 4000
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for i in range(0, len(message), max_length):
        part = message[i:i + max_length]
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": part}
        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                logging.info("Telegram notification sent successfully.")
            else:
                logging.error(f"Failed to send Telegram notification. Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logging.error(f"Error sending Telegram notification: {e}")

def send_query_response_to_telegram(query: str, summary: str, image_analysis: str, conclusion: str):
    message = (
        f"User Query:\n{query}\n\n"
        f"Summary:\n{summary}\n\n"
        f"Image Analysis:\n{image_analysis}\n\n"
        f"Conclusion:\n{conclusion}"
    )
    send_telegram_notification(message)

def save_history(query: str, response: str):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"User Query:\n{query}\nBot Response:\n{response}\n{'-'*40}\n")

def chatbot_response(user_input, selected_model):
    if contains_profanity(user_input):
        warning_message = f"Swear words detected in query: {user_input}. Query not processed."
        logging.warning(warning_message)
        send_telegram_notification(warning_message)
        return "Your query contains profanity. Please edit it and try again."
    
    profanity_result = profanity_chain.run(text=user_input).strip().lower()
    if profanity_result == "yes":
        warning_message = f"Swear words detected in query: {user_input}. Query not processed."
        logging.warning(warning_message)
        send_telegram_notification(warning_message)
        return "Your query contains profanity. Please edit it and try again."
    
    clear_chroma_db()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=".chroma_db")
    search_results = fetch_web_content(user_input, max_results=10)
    if not search_results:
        return "No results found. Try rephrasing your query."
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for title, link, content in search_results:
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            vectorstore.add_texts(texts=[chunk], metadatas=[{"title": title, "url": link}])
    
    vectorstore.persist()
    docs_with_scores = vectorstore.similarity_search_with_score(user_input, k=10)
    threshold = 0.9
    retrieved_context = "\n".join([doc.page_content for doc, score in docs_with_scores if score >= threshold])
    
    llm_model = OllamaLLM(model=selected_model)
    prompt = f"Context:\n{retrieved_context}\n\nQuestion:\n{user_input}\n\nPlease provide an answer based only on the above context."
    response = llm_model(prompt)
    return response

def generate_conclusion(text_summary, image_summary, selected_model):
    prompt = (
        f"Generate a conclusion based on the following:\n"
        f"Text Summary:\n{text_summary}\n\n"
        f"Image Analysis:\n{image_summary}"
    )
    response = ollama.chat(model=selected_model, messages=[{"role": "user", "content": prompt}])
    return response["message"].get("content", "Failed to generate conclusion.")

async def main(user_query, search_mode, selected_model):
    with ThreadPoolExecutor() as executor:
        if search_mode == "Web Search":
            summary = await asyncio.get_event_loop().run_in_executor(
                executor, chatbot_response, user_query, selected_model
            )
            if "Your query contains profanity" in summary:
                image_urls = []
                image_summary = ""
                conclusion = summary
            else:
                image_urls = fetch_image_urls(user_query)
                image_summary_data = await asyncio.get_event_loop().run_in_executor(
                    executor, analyze_images, image_urls
                )
                image_summary = "\n".join([f"{url}: {s}" for url, s in image_summary_data])
                conclusion = await asyncio.get_event_loop().run_in_executor(
                    executor, generate_conclusion, summary, image_summary, selected_model
                )
        else:
            summary = document_search(user_query, selected_model)
            image_urls = []
            image_summary = ""
            conclusion = await asyncio.get_event_loop().run_in_executor(
                executor, generate_conclusion, summary, image_summary, selected_model
            )
    
    new_entry = {"query": user_query, "response": summary}
    st.session_state["short_term_memory"].append(new_entry)
    if len(st.session_state["short_term_memory"]) > 5:
        st.session_state["long_term_memory"].extend(st.session_state["short_term_memory"])
        st.session_state["short_term_memory"] = []
    
    save_history(user_query, summary)
    send_query_response_to_telegram(user_query, summary, image_summary, conclusion)
    
    return summary, image_urls, image_summary, conclusion

def start(update, context):
    update.message.reply_text(
        "Welcome! This bot allows you to view the history (/history) and delete it (/delete_history)."
    )

def history_cmd(update, context):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history_text = f.read()
        if not history_text.strip():
            update.message.reply_text("History is empty.")
        else:
            update.message.reply_text(history_text)
    except Exception as e:
        update.message.reply_text("Error retrieving history.")

def delete_history(update, context):
    try:
        open(HISTORY_FILE, "w", encoding="utf-8").close()
        update.message.reply_text("History deleted.")
    except Exception as e:
        update.message.reply_text("Error deleting history.")

def run_telegram_bot():
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("history", history_cmd))
    dp.add_handler(CommandHandler("delete_history", delete_history))
    updater.start_polling()
    updater.idle()

# Start the Telegram bot in a separate thread
bot_thread = threading.Thread(target=run_telegram_bot)
bot_thread.start()

# --- Page "Query" (Web Search / Document Processing) ---
if page == "Query":
    st.subheader("Query")
    user_query = st.text_input("Enter query:")
    if user_query:
        summary, image_urls, image_summary, conclusion = asyncio.run(
            main(user_query, search_mode, selected_model)
        )
        st.subheader("Result")
        st.write(summary)
        
        if search_mode == "Web Search":
            st.subheader("Image Analysis")
            for img in image_urls:
                st.image(img, caption="Analyzed Image")
            st.write(image_summary)
        
        st.subheader("Conclusion")
        st.write(conclusion)

# --- Page "Chat with Ollama" (Standalone Chat) ---
if page == "Chat with Ollama":
    st.markdown("## Chat with Ollama")
    # Initialize separate memory for Chat with Ollama
    if "ollama_chat_short" not in st.session_state:
        st.session_state["ollama_chat_short"] = []  # Short-term memory
    if "ollama_chat_long" not in st.session_state:
        # Load previous long-term memory from file, if it exists
        if os.path.exists("ollama_chat_long.json"):
            try:
                with open("ollama_chat_long.json", "r", encoding="utf-8") as f:
                    st.session_state["ollama_chat_long"] = json.load(f)
            except Exception as e:
                st.error(f"Error loading long-term memory: {e}")
                st.session_state["ollama_chat_long"] = []
        else:
            st.session_state["ollama_chat_long"] = []

    chat_memory_mode = st.radio("Chat Memory Mode", ["Short-term", "Long-term"], key="ollama_memory")
    chat_input = st.text_input("Enter message for Chat with Ollama", key="ollama_input")

    if st.button("Send", key="send_ollama_chat"):
        if chat_input:
            if contains_profanity(chat_input):
                st.error("Your message contains profanity. Please edit it and try again.")
            else:
                if chat_memory_mode == "Short-term":
                    history = "\n".join(st.session_state["ollama_chat_short"])
                else:
                    history = "\n".join(st.session_state["ollama_chat_long"])
                prompt = f"Conversation history:\n{history}\nUser: {chat_input}\nBot:"
                llm_model = OllamaLLM(model=selected_model)
                response = llm_model(prompt)
                bot_reply = response  # Adapt if needed
                st.write("Bot:", bot_reply)
                entry = f"User: {chat_input}\nBot: {bot_reply}"
                if chat_memory_mode == "Short-term":
                    st.session_state["ollama_chat_short"].append(entry)
                else:
                    st.session_state["ollama_chat_long"].append(entry)
                    with open("ollama_chat_long.json", "w", encoding="utf-8") as f:
                        json.dump(st.session_state["ollama_chat_long"], f, ensure_ascii=False, indent=2)

    st.markdown("### Chat History with Ollama")
    if chat_memory_mode == "Short-term":
        if st.session_state["ollama_chat_short"]:
            for msg in st.session_state["ollama_chat_short"]:
                st.write(msg)
        else:
            st.write("Short-term memory is empty.")
    else:
        if st.session_state["ollama_chat_long"]:
            for msg in st.session_state["ollama_chat_long"]:
                st.write(msg)
        else:
            st.write("Long-term memory is empty.")
