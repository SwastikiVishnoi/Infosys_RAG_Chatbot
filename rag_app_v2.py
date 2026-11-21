import json
import os
import re

import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from google import genai
from rank_bm25 import BM25Okapi

MODEL = 'gemini-3.5-flash-lite'  # use the same model name that worked in Colab
K = 10


def tokenize(text):
    return re.findall(r'\w+', text.lower())


@st.cache_resource
def load_everything():
    with open('chunks.json', encoding='utf-8') as f:
        chunks = json.load(f)
    embeddings = np.load('embeddings.npy')
    bm25 = BM25Okapi([tokenize(c['text']) for c in chunks])
    embedder = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
    return chunks, embeddings, bm25, embedder


def get_api_key():
    try:
        return st.secrets['GEMINI_API_KEY']
    except Exception:
        return os.environ.get('GEMINI_API_KEY')


def hybrid_search(question, chunks, embeddings, bm25, embedder, k=K, pool=30):
    q = np.array(list(embedder.embed([question])))[0]
    q = q / np.linalg.norm(q)
    sem = embeddings @ q
    kw = bm25.get_scores(tokenize(question))
    sem_rank = sem.argsort()[::-1][:pool]
    kw_rank = np.argsort(kw)[::-1][:pool]
    fused = {}
    for r, i in enumerate(sem_rank):
        fused[i] = fused.get(i, 0) + 1 / (60 + r)
    for r, i in enumerate(kw_rank):
        fused[i] = fused.get(i, 0) + 1 / (60 + r)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(chunks[i]['page'], chunks[i]['text']) for i in top]


def answer_question(question, results, client):
    context = ''
    for page, text in results:
        context += f'[PDF page {page}]\n{text}\n\n'
    prompt = f"""You answer questions about the Infosys Integrated Annual Report 2025-26.
Use ONLY the context below. Cite the PDF page numbers you used, like (page 72).
If the answer is not in the context, say exactly: I could not find this in the report.

Context:
{context}
Question: {question}
Answer:"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


st.set_page_config(page_title='Infosys Annual Report Q&A')
st.title('Infosys Annual Report Q&A')
st.write('Ask a question about the Infosys Integrated Annual Report 2025-26. '
         'Answers cite PDF page numbers.')

api_key = get_api_key()
if not api_key:
    st.error('Gemini API key not found. Set GEMINI_API_KEY (see README).')
    st.stop()

client = genai.Client(api_key=api_key)
chunks, embeddings, bm25, embedder = load_everything()

question = st.text_input('Your question', placeholder='Who is the CEO of Infosys?')

if st.button('Ask') and question.strip():
    with st.spinner('Searching the report...'):
        results = hybrid_search(question, chunks, embeddings, bm25, embedder)
        try:
            answer = answer_question(question, results, client)
        except Exception as e:
            st.error(f'The language model request failed: {e}')
            st.stop()
    st.subheader('Answer')
    st.write(answer)

    # Pick out the page numbers the answer cites
    cited = set()
    for group in re.findall(r'pages?\s*([\d,\s]+(?:and\s*\d+)?)', answer, flags=re.I):
        cited.update(int(n) for n in re.findall(r'\d+', group))
    cited = sorted(cited)
    if cited:
        st.caption('Pages cited in the answer: ' + ', '.join(str(p) for p in cited))

    # Group the retrieved passages by page, cited pages first
    by_page = {}
    for page, text in results:
        by_page.setdefault(page, []).append(text)
    ordered = sorted(by_page, key=lambda p: (p not in cited, p))

    with st.expander(f'Evidence: passages retrieved from {len(ordered)} pages'):
        st.caption('These are the raw text pieces the model was given. '
                   'Pieces are cut at fixed lengths, so they can start or end mid-sentence.')
        for page in ordered:
            tag = ' (cited)' if page in cited else ''
            st.markdown(f'**PDF page {page}{tag}**')
            st.write(' ... '.join(t[:250] for t in by_page[page][:2]))

st.caption('Answers come from the report text and can be wrong. Check the cited pages.')
