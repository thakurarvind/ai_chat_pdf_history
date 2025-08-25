AI Chat with PDF + Qdrant + Google Gemini (API key)
--------------------------------------------------
Features:
- Upload multiple PDFs and index their chunks into local Qdrant.
- Chat with the indexed documents using Google Gemini (API key).
- Session-based chat history (shows previous Q&A in the UI).
- Ability to restrict searches to the first uploaded file via a dropdown.
- Optional persistence of chat history into Qdrant (chat_history collection).

Quick start:
1. Start Qdrant locally (default http://localhost:6333).
2. Set GEMINI_API_KEY in .env file.
3. Create and activate venv, then install requirements:
   pip install -r requirements.txt
4. Run app:
   python app.py
5. Open http://127.0.0.1:5000

Notes:
- Embeddings and LLM calls use the google-generativeai package.
- Make sure your API key supports embeddings & Gemini model calls.
