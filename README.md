# StudyMate — AI-Powered PDF Q&A (Hugging Face version)

StudyMate lets students upload **PDFs** (textbooks, notes, papers) and ask **natural-language questions**.  
It performs **semantic search** over the PDFs using **SentenceTransformers + FAISS** and generates **grounded answers with citations** using a **Hugging Face model** (default: `google/flan-t5-small`).

## ✨ Features
- Upload multiple PDFs
- Clean extraction via **PyMuPDF**
- Chunking with overlap for better recall
- Fast embeddings with **all-MiniLM-L6-v2**
- FAISS vector search
- LLM answer generation via **Hugging Face Transformers**
- Inline **citations**: `[Page X, FileName]`
- Streamlit UI with tunable parameters

## 🧱 Folder Structure
```
studymate/
├─ streamlit_app.py
├─ requirements.txt
├─ .env.example
├─ README.md
├─ assets/
│  └─ styles.css
├─ src/
│  ├─ __init__.py
│  ├─ text_utils.py
│  ├─ pdf_ingest.py
│  ├─ vectorstore.py
│  ├─ llm_hf.py
│  └─ rag.py
└─ data/
   └─ samples/
```

## 🚀 Quickstart
1. (Optional) Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Set a Hugging Face token (only if using private models):
   - Copy `.env.example` to `.env`, fill `HF_TOKEN` if needed.
4. Run:
   ```bash
   streamlit run streamlit_app.py
   ```

> **Tip:** First run will download the HF model weights; ensure internet access. For low-RAM systems, switch to `google/flan-t5-small` in the app's Settings.

## 🧠 Models
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generation (default): `google/flan-t5-small` (fast and lightweight)
  - Alternatives: `google/flan-t5-base` (balanced), `google/flan-t5-large` (more powerful), `ibm-granite/granite-3.3-2b-instruct` (powerful but slower)

## 🔒 Environment Variables
Create a `.env` file if you need to pass a **Hugging Face token** (not required for public models):
```
HF_TOKEN=your_hf_token_here
```

## ⚡ Performance Optimizations
- Uses Streamlit caching to avoid reloading models on every request
- Implements fallback models for better reliability
- Includes `hf_xet` package for faster Hugging Face downloads
- Reduced token generation for faster responses

## 🧪 Notes
- Works best with text-based PDFs. Scanned PDFs without OCR won't extract well.
- If answers feel vague, increase Top-K or chunk size, or switch to a stronger embedding model.
- For CPU-only, keep generation model small (FLAN-T5 base/small).

## 🛠️ Troubleshooting
### LLM Setup Errors
If you encounter errors like "does not appear to have a file named pytorch_model.bin", the model has TensorFlow weights only. The application will automatically try to load with `from_tf=True`.

### Disk Space Issues
Models can require several hundred MBs to several GBs of disk space. If you see disk space warnings:
- Free up disk space on your system
- Use smaller models like `google/flan-t5-small` or `sshleifer/tiny-gpt2`
- Clear Hugging Face cache: Run `huggingface-cli delete-cache` in your terminal/command prompt (requires huggingface-hub package)
  - Note: Make sure you don't add extra characters like periods at the end of the command

### Changing Download Location
By default, Hugging Face models are downloaded to your user cache directory. To change this to the D drive:
1. Set the `HF_HOME` environment variable to a path on your D drive:
   - Windows Command Prompt: `set HF_HOME=D:\huggingface_cache`
   - Windows PowerShell: `$env:HF_HOME="D:\huggingface_cache"`
   - Or set it in your system environment variables
2. Restart your terminal/command prompt after setting the environment variable
3. Run the application as normal

### Model Loading Failures
The application includes fallback models that will load if primary models fail:
1. Primary model (user-selected or default)
2. TensorFlow fallback (if primary has TF weights only)
3. Minimal fallback model (`sshleifer/tiny-gpt2`)
4. Dummy model (returns default message if all else fails)

### "index out of range in self" Error
This error occurs when there are issues with model output formats. The application now handles this gracefully by:
- Checking the output format before accessing it
- Providing fallback responses when generation fails
- Displaying helpful error messages in the UI

### Improving Answer Quality
To get more focused answers:
1. Be specific in your questions (e.g., "What are the extra features of the LMS platform?" rather than "Tell me about the LMS")
2. The system will focus on the most relevant context from your PDFs
3. For complex questions, try breaking them into smaller parts

## 🚀 New Features
### Dark Mode
Toggle between light and dark mode for comfortable reading in any lighting condition.

### Question History
Keep track of your previous questions and answers with the built-in history feature.

### Export Answers
Download answers and their sources as text files for offline reference.

### Document Summarization
Get a concise summary of your uploaded documents with the click of a button.

### Search in History
Search through your previous questions and answers to quickly find relevant information.

## 📜 License
MIT (do whatever, just be nice and cite 🙂)
