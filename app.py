import os
import time
#from uuid import uuid4
import uuid
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, session
from flask_session import Session
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from io import BytesIO
import secrets


from embeddings import embed_text
from llm import answer_query

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
sec_key= secrets.token_hex(32);
print(sec_key)


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', sec_key)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION', 'pdf_chunks')
CHAT_COLLECTION = os.getenv('CHAT_COLLECTION', 'chat_history')

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure collections exist
existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
if CHAT_COLLECTION not in existing:
    qdrant.create_collection(
        collection_name=CHAT_COLLECTION,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE)
    )

def ensure_session():
    # track uploaded files order and chat history in session
    if 'uploaded_files' not in session:
        session['uploaded_files'] = []  # list of filenames in upload order
    if 'chat_history' not in session:
        session['chat_history'] = []  # list of {"q":.., "a":.., "timestamp":..}

@app.route('/')
def index():
    ensure_session()
    return render_template('index.html', uploaded_files=session['uploaded_files'], chat_history=session['chat_history'])

@app.route('/upload', methods=['POST'])
def upload():
    ensure_session()
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'no file'}), 400
    filename = f.filename
    # Save temporarily and read
    content = f.read()
    #reader = PdfReader(stream=content)
    reader = PdfReader(BytesIO(content))
    #file_id = str(uuid4())
    file_id = str(uuid.uuid4())

    upload_order = len(session['uploaded_files']) + 1
    # Process pages and upsert each page as a chunk
    points = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        if not text.strip():
            continue
        vec = embed_text(text)
        print(vec)
        #pid = f"{file_id}-{i}"
        pid = i
        payload = {'text': text, 'file_name': filename, 'upload_order': upload_order, 'page': i}
        points.append(PointStruct(id=pid, vector=vec, payload=payload))

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    # register file in session
    #session['uploaded_files'].append({'file_name': filename, 'file_id': file_id, 'order': upload_order})
    session['uploaded_files'].append({'file_name': filename, 'file_id': pid, 'order': upload_order})
    session.modified = True
    return jsonify({'status': 'ok', 'file_name': filename, 'chunks': len(points)})

@app.route('/ask', methods=['POST'])
def ask():
    ensure_session()
    data = request.json or {}
    question = data.get('question')
    file_filter = data.get('file_name')  # optional: restrict to a specific file_name
    if not question:
        return jsonify({'error': 'no question'}), 400
    q_vec = embed_text(question)
    # Build filter
    query_filter = None
    if file_filter:
        query_filter = {
            'must': [
                {'key': 'file_name', 'match': {'value': file_filter}}
            ]
        }
    search_result = qdrant.search(

        collection_name=COLLECTION_NAME,
        query_vector=q_vec,
        limit=5,
        with_payload=True,
        query_filter=query_filter
    )
    docs = [hit.payload.get('text', '') for hit in search_result]
    answer = answer_query(question, docs)
    # Save chat in session and optionally in qdrant chat_history
    chat_item = {'question': question, 'answer': answer, 'timestamp': int(time.time()), 'file_filter': file_filter}
    session['chat_history'].append(chat_item)
    session.modified = True
    # also persist to qdrant chat collection (as payload); use a dummy vector [0.0]
    try:
        qdrant.upsert(collection_name=CHAT_COLLECTION, points=[PointStruct(id=str(uuid4()), vector=[0.0], payload=chat_item)])
    except Exception:
        pass
    return jsonify({'answer': answer, 'chat_history': session['chat_history']})

@app.route('/history', methods=['GET'])
def history():
    ensure_session()
    return jsonify({'chat_history': session['chat_history']})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['chat_history'] = []
    session.modified = True
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
