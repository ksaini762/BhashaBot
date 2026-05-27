import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from deep_translator import GoogleTranslator
import tempfile

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="BhashaBot — Ask in Any Language, Answer in Punjabi",
    page_icon="🤖",
    layout="centered"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background-color: #0a1628; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    color: #f0efe9;
    line-height: 1.2;
    margin-bottom: 8px;
}
.hero-title span { color: #c9a84c; }

.hero-sub {
    color: #8892a4;
    font-size: 1rem;
    margin-bottom: 32px;
    line-height: 1.7;
}

.gold-tag {
    background: rgba(201,168,76,0.1);
    border: 1px solid rgba(201,168,76,0.3);
    color: #c9a84c;
    padding: 4px 12px;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-radius: 2px;
    display: inline-block;
    margin-bottom: 16px;
}

.answer-box {
    background: #112240;
    border: 1px solid rgba(201,168,76,0.2);
    border-left: 3px solid #c9a84c;
    padding: 20px 24px;
    border-radius: 4px;
    margin: 12px 0;
}

.answer-label {
    color: #c9a84c;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.answer-text {
    color: #ccd6f6;
    font-size: 0.95rem;
    line-height: 1.8;
}

.punjabi-box {
    background: #0a1628;
    border: 1px solid rgba(201,168,76,0.2);
    border-left: 3px solid #e8c96a;
    padding: 20px 24px;
    border-radius: 4px;
    margin: 12px 0;
}

.step-box {
    background: #112240;
    border: 1px solid rgba(201,168,76,0.15);
    padding: 16px 20px;
    border-radius: 4px;
    margin: 8px 0;
    color: #8892a4;
    font-size: 0.88rem;
}

.divider {
    border: none;
    border-top: 1px solid rgba(201,168,76,0.15);
    margin: 28px 0;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid #c9a84c !important;
    color: #c9a84c !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    padding: 10px 28px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #c9a84c !important;
    color: #0a1628 !important;
}

.stTextInput > div > div > input {
    background: #112240 !important;
    border: 1px solid rgba(201,168,76,0.2) !important;
    color: #f0efe9 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #c9a84c !important;
    box-shadow: none !important;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── HEADER ────────────────────────────────────────────────────
st.markdown('<div class="gold-tag">AI · NLP · Regional Language</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Bhasha<span>Bot</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload any document · Ask in any language · Get answers in <strong style="color:#c9a84c">Punjabi</strong></div>', unsafe_allow_html=True)
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)


# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="gold-tag">Configuration</div>', unsafe_allow_html=True)
    st.markdown("### API Key")
    groq_api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key from console.groq.com"
    )
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown("### How to use")
    st.markdown('<div class="step-box">① Enter your Groq API key above</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-box">② Upload one or more PDF files</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-box">③ Click "Build Knowledge Base"</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-box">④ Ask any question below</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-box">⑤ Get answers in English + Punjabi</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8892a4; font-size:0.78rem;">Built by Kanchan Saini · TIET 2026<br/>Powered by LangChain + Groq LLaMA 3</div>', unsafe_allow_html=True)


# ─── SESSION STATE ─────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─── HELPER FUNCTIONS ──────────────────────────────────────────
def translate_to_punjabi(text):
    try:
        return GoogleTranslator(source="auto", target="pa").translate(text)
    except Exception as e:
        return f"[Translation error: {e}]"


def build_knowledge_base(uploaded_files):
    all_docs = []
    with st.spinner("Reading documents..."):
        for uploaded_file in uploaded_files:
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)
                st.success(f"✅ Loaded: {uploaded_file.name} ({len(docs)} pages)")
            except Exception as e:
                st.error(f"❌ Error loading {uploaded_file.name}: {e}")
            finally:
                os.unlink(tmp_path)

    if not all_docs:
        st.error("No documents loaded. Please upload valid PDF or TXT files.")
        return None

    with st.spinner("Creating embeddings (takes 1-2 mins)..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    st.success(f"✅ Knowledge base ready! ({len(chunks)} chunks indexed)")
    return vectorstore


def build_chain(vectorstore, api_key):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3, api_key=api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the context below to answer the question.
Give a clear and short answer. If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ─── DOCUMENT UPLOAD ───────────────────────────────────────────
st.markdown("### Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files and groq_api_key:
    if st.button("Build Knowledge Base"):
        vs = build_knowledge_base(uploaded_files)
        if vs:
            st.session_state.vectorstore = vs
            st.session_state.chain = build_chain(vs, groq_api_key)
            st.session_state.chat_history = []

elif uploaded_files and not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar first.")

elif not uploaded_files:
    st.markdown('<div class="step-box">👆 Upload a PDF or TXT file above to get started</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider"/>', unsafe_allow_html=True)


# ─── CHAT INTERFACE ────────────────────────────────────────────
if st.session_state.chain:
    st.markdown("### Ask a Question")
    user_question = st.text_input("", placeholder="What is this document about?", label_visibility="collapsed")

    if st.button("Get Answer in Punjabi"):
        if user_question.strip():
            with st.spinner("Searching documents..."):
                english_answer = st.session_state.chain.invoke(user_question)
            with st.spinner("Translating to Punjabi..."):
                punjabi_answer = translate_to_punjabi(english_answer)

            st.session_state.chat_history.append({
                "question": user_question,
                "english": english_answer,
                "punjabi": punjabi_answer
            })
        else:
            st.warning("Please enter a question.")

    # Show chat history
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div style="color:#c9a84c; font-size:0.8rem; letter-spacing:1px; margin-top:20px;">YOU ASKED</div>
        <div style="color:#f0efe9; font-size:0.95rem; margin-bottom:12px;">{chat['question']}</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="answer-box">
            <div class="answer-label">English Answer</div>
            <div class="answer-text">{chat['english']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="punjabi-box">
            <div class="answer-label">ਪੰਜਾਬੀ ਜਵਾਬ · Punjabi Answer</div>
            <div class="answer-text">{chat['punjabi']}</div>
        </div>
        <hr class="divider"/>
        """, unsafe_allow_html=True)

else:
    st.markdown('<div class="step-box">💬 Chat will appear here once you upload documents and build the knowledge base</div>', unsafe_allow_html=True)
