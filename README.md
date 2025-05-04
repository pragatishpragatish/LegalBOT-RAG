
# ⚖️ LegalBOT – RAG-Powered Legal Assistant (Offline)

LegalBOT is an AI-powered legal assistant that uses Retrieval-Augmented Generation (RAG) to answer legal questions based on a curated set of uploaded legal PDFs. Built using Flask, FAISS, Ollama, and LangChain, it runs fully offline using open-source LLMs like **Gemma 3B** via Ollama.

---

## 🚀 Features

- 🧠 **RAG-Based QA**: Answers user queries based on actual legal documents (PDFs).
- 📄 **Multi-PDF Support**: Just drop PDFs into the `/data` folder.
- 🔍 **Semantic Search** with FAISS and HuggingFace embeddings.
- 🗨️ **Chat Interface**: Modern conversational UI with persistent history.
- 🎨 **Dark & Light Mode** toggle via one click.
- 🔐 **Secure Sessions**: No cookies for chat history.
- 📦 **Offline & Private**: Runs locally with no internet needed.

---

## 📁 Folder Structure

```bash
.
├── app.py               # Main Flask app
├── template.html        # Chat UI HTML template
├── data/                # Folder to store your legal PDFs
├── index/               # FAISS vector index (auto-generated)
├── README.md            # This file
```

---

## 🔧 Installation

> **Requirements**:
> - Python 3.8+
> - [Ollama](https://ollama.com) installed and running locally
> - A working model pulled (e.g., `gemma3:1b`)

### 1. 📦 Install Python dependencies

```bash
pip install flask langchain langchain-community langchain-ollama faiss-cpu sentence-transformers pymupdf
```

### 2. 🧠 Pull LLM model via Ollama

```bash
ollama pull gemma3:1b
```

You can change this model later in the code (`app.py`).

### 3. 📄 Add Legal PDFs

Place all your legal documents inside the `data/` folder. These will be processed automatically.

### 4. ▶️ Run the app

```bash
python app.py
```

Then visit: [http://localhost:5000](http://localhost:5000)

---

## 🌟 Usage

- Ask legal questions like:  
  *"What are the grounds for divorce under Indian law?"*  
  *"Explain the rights of an arrested person in India."*

- Click **Toggle Theme** for dark/light mode.
- Use the **Clear Conversation** button to reset the chat.

---

## 🧪 Tech Stack

| Component           | Technology                       |
|--------------------|----------------------------------|
| Backend            | Flask                            |
| LLM                | Gemma via Ollama                 |
| Embeddings         | Sentence-Transformers (`MiniLM`) |
| Vector DB          | FAISS                            |
| Document Parsing   | PyMuPDF                          |
| RAG Framework      | LangChain                        |

---

## 🛡️ Privacy & Security

- ✅ All data is processed **locally**.
- ✅ No external API calls.
- ✅ User chat history is stored in Flask `session`, not cookies.

---

## 📦 TODO / Enhancements

- [ ] File upload support (add PDFs on the fly)
- [ ] Streaming response (typing animation)
- [ ] PDF viewer pane
- [ ] Admin dashboard (monitor usage)
- [ ] SQLite or MongoDB chat history
- [ ] Docker deployment

---

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/e1351c18-ab25-444b-8b40-69b257b70b22)


---

## 🧠 Credits

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## ❤️ Like this project?

Star it, share it, or build your own private assistant on top of it!
