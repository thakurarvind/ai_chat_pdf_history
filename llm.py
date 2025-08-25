import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

def answer_query(question, docs):
    context_text = "\n\n---\n\n".join(docs) if docs else ''
    prompt = f"""You are a helpful assistant. Use the following context taken from user-uploaded PDFs to answer the question.

Context:
{context_text}

Question: {question}

Answer concisely and cite the context where relevant."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        if isinstance(response, dict) and 'candidates' in response and len(response['candidates']):
            # some responses provide candidates list
            return response['candidates'][0].get('content', '')
        return str(response)
    except Exception as e:
        if docs:
            return f'Fallback (Gemini API error: {str(e)}). Top context:\n\n' + '\n\n'.join(docs[:3])
        return f'No answer available: Gemini API error: {str(e)}'
