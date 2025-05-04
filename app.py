import os
import fitz  # PyMuPDF for PDF parsing
from flask import Flask, request, render_template_string
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document  # Import Document class

# Constants
DATA_DIR = "data"
INDEX_DIR = "index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Set up Flask
app = Flask(__name__)

# Load Ollama LLM with detailed config
llm = OllamaLLM(
    model="gemma3:1b",
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.1
)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to create FAISS vector store from PDF files
def create_vector_store():
    print("[INFO] Creating vector store from PDF files...")

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Loop through PDF files in the data directory
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            text = extract_text_from_pdf(path)
            # Split the text into chunks
            split_docs = splitter.split_text(text)
            
            # Wrap the text chunks into Document objects
            for chunk in split_docs:
                all_docs.append(Document(page_content=chunk))

    db = FAISS.from_documents(all_docs, embedding_model)
    db.save_local(INDEX_DIR)
    print("[INFO] Vector store saved to:", INDEX_DIR)
    return db

# Load or create the FAISS index
try:
    db = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
    print("[INFO] Loaded existing FAISS index.")
except Exception as e:
    print("[WARN] Index not found or corrupted. Rebuilding...")
    db = create_vector_store()

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html data-theme="{{ theme }}">
<head>
    <title>LegalBOT RAG</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
            transition: background-color 0.3s, color 0.3s;
        }
        
        :root {
            --bg-primary: #f9f9f9;
            --bg-secondary: #ffffff;
            --text-primary: #1f2937;
            --text-secondary: #6b7280;
            --border-color: #e5e5e5;
            --user-bubble: #2563eb;
            --user-text: white;
            --bot-bubble: #e5e7eb;
            --bot-text: #1f2937;
            --input-border: #d1d5db;
        }
        
        [data-theme="dark"] {
            --bg-primary: #111827;
            --bg-secondary: #1f2937;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --border-color: #374151;
            --user-bubble: #3b82f6;
            --user-text: white;
            --bot-bubble: #374151;
            --bot-text: #f3f4f6;
            --input-border: #4b5563;
        }
        
        body {
            background-color: var(--bg-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            color: var(--text-primary);
        }
        
        .header {
            background-color: var(--bg-secondary);
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .header h1 {
            color: var(--text-primary);
            font-size: 20px;
            font-weight: bold;
        }
        
        .theme-toggle {
            width: 48px;
            height: 24px;
            background-color: var(--border-color);
            border-radius: 12px;
            position: relative;
            cursor: pointer;
            display: inline-block;
        }
        
        .theme-toggle::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: var(--bg-secondary);
            top: 2px;
            left: 2px;
            transition: transform 0.3s;
        }
        
        [data-theme="dark"] .theme-toggle::after {
            transform: translateX(24px);
        }
        
        .theme-toggle-container {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .theme-label {
            font-size: 14px;
            color: var(--text-secondary);
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        
        .welcome-message {
            text-align: center;
            color: var(--text-secondary);
            margin: auto;
            max-width: 500px;
        }
        
        .welcome-message h2 {
            font-size: 24px;
            margin-bottom: 12px;
            color: var(--text-primary);
        }
        
        .welcome-message p {
            font-size: 16px;
        }
        
        .message {
            display: flex;
            max-width: 80%;
        }
        
        .user-message {
            margin-left: auto;
            justify-content: flex-end;
        }
        
        .bot-message {
            margin-right: auto;
            justify-content: flex-start;
        }
        
        .message-bubble {
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 16px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-width: 100%;
        }
        
        .user-bubble {
            background-color: var(--user-bubble);
            color: var(--user-text);
            border-bottom-right-radius: 4px;
        }
        
        .bot-bubble {
            background-color: var(--bot-bubble);
            color: var(--bot-text);
            border-bottom-left-radius: 4px;
        }
        
        .input-container {
            background-color: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
            padding: 16px 24px;
            position: sticky;
            bottom: 0;
        }
        
        .input-box {
            display: flex;
            border: 1px solid var(--input-border);
            border-radius: 8px;
            overflow: hidden;
            background-color: var(--bg-secondary);
        }
        
        .input-box textarea {
            flex: 1;
            border: none;
            outline: none;
            padding: 12px 16px;
            font-size: 16px;
            resize: none;
            min-height: 56px;
            background-color: var(--bg-secondary);
            color: var(--text-primary);
        }
        
        .input-box textarea::placeholder {
            color: var(--text-secondary);
        }
        
        .input-box input[type="submit"] {
            background-color: var(--user-bubble);
            color: var(--user-text);
            border: none;
            padding: 0 16px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
        }
        
        .input-box input[type="submit"]:hover {
            opacity: 0.9;
        }
        
        .helper-text {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 8px;
            text-align: center;
        }
        
        .controls {
            display: flex;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        
        .clear-btn {
            background: none;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .clear-btn:hover {
            background-color: var(--border-color);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 LegalBOT - Ask Legal Questions</h1>
        <a href="?toggle_theme=1" class="theme-toggle-container">
            <span class="theme-label">{{ 'Light' if theme == 'light' else 'Dark' }}</span>
            <span class="theme-toggle"></span>
        </a>
    </div>
    
    <div class="chat-container">
        {% if not conversation %}
        <div class="welcome-message">
            <h2>Welcome to LegalBOT</h2>
            <p>Ask any questions about legal documents and get instant answers based on the provided PDFs.</p>
        </div>
        {% else %}
            <div class="controls">
                <a href="/" class="clear-btn">Clear Conversation</a>
            </div>
            
            {% for message in conversation %}
                <div class="message {{ 'user-message' if message.role == 'user' else 'bot-message' }}">
                    <div class="message-bubble {{ 'user-bubble' if message.role == 'user' else 'bot-bubble' }}">
                        {{ message.content }}
                    </div>
                </div>
            {% endfor %}
        {% endif %}
    </div>
    
    <div class="input-container">
        <form method="post">
            <div class="input-box">
                <textarea name="query" placeholder="Ask a question based on the legal documents..." rows="1"></textarea>
                <input type="submit" value="Send">
            </div>
        </form>
        <div class="helper-text">Press Enter to submit your question</div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    # Initialize session if it doesn't exist
    if 'conversation' not in request.cookies:
        conversation = []
    else:
        import json
        conversation = json.loads(request.cookies.get('conversation', '[]'))
    
    # Get theme preference
    theme = request.cookies.get('theme', 'light')
    
    if request.method == "POST":
        query = request.form["query"]
        
        # Add user query to conversation
        conversation.append({"role": "user", "content": query})
        
        # Get bot response
        result = qa_chain.run(query)
        
        # Add bot response to conversation
        conversation.append({"role": "bot", "content": result})
    
    # Create response
    response = render_template_string(HTML_TEMPLATE, 
                                     conversation=conversation,
                                     theme=theme)
    
    # Set cookies
    from flask import make_response
    resp = make_response(response)
    
    if request.method == "POST":
        import json
        resp.set_cookie('conversation', json.dumps(conversation))
    
    # Handle theme toggle if present
    if request.args.get('toggle_theme'):
        new_theme = 'dark' if theme == 'light' else 'light'
        resp.set_cookie('theme', new_theme)
    
    return resp

if __name__ == "__main__":
    app.run(debug=True)
