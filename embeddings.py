import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Adjust model name if your key uses a different embeddings model
#EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'models/text-embedding-003')  # change if needed
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'models/embedding-001')  # change if needed

def embed_text(text):
    # Trim or handle long text as needed
    try:
        resp = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        # The return structure may vary; handle both dict/list shapes
        if isinstance(resp, dict) and 'embedding' in resp:
            return resp['embedding']
        if isinstance(resp, list) and len(resp) and 'embedding' in resp[0]:
            return resp[0]['embedding']
        # Some clients return nested data; try common paths
        if isinstance(resp, dict) and 'data' in resp and isinstance(resp['data'], list):
            first = resp['data'][0]
            return first.get('embedding') or first.get('vector') or first.get('values')
        raise ValueError('Unexpected embedding response format')
    except Exception as e:
        print('Embedding error:', e)
        # return zero vector to avoid crash; in prod you should handle retries
        return [0.0]*768
