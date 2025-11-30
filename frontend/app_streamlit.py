# frontend/app_streamlit.py
import streamlit as st
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Knowledge Base Agent",
    page_icon="📚",
    layout="wide"
)
# ---- CUSTOM CSS (ChatGPT-like UI) ---- #
st.markdown("""
<style>

.chat-container {
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
    width: 100%;
}

.user-bubble {
    background-color: #0b93f6;
    color: white;
    padding: 12px;
    border-radius: 12px;
    max-width: 70%;
    margin-left: auto;
}

.bot-bubble {
    background-color: #f1f0f0;
    color: black;
    padding: 12px;
    border-radius: 12px;
    max-width: 70%;
    margin-right: auto;
}

.source-box {
    background-color: #e8e8e8;
    padding: 8px;
    border-radius: 8px;
    margin-top: 5px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

st.title("📚 AI Knowledge Base Agent")


with st.sidebar:
    st.header("Upload Documents")
    project = st.text_input("AI Agent", value="KB_Agent")
    uploaded = st.file_uploader("Upload multiple files", accept_multiple_files=True, type=["pdf","txt","docx"])
    if st.button("Upload & Ingest"):
        if uploaded:
            files = uploaded
            files_payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
            with st.spinner("Uploading and ingesting..."):
                resp = requests.post(f"{API_URL}/upload", files=files_payload, data={"project": project})
            # st.success(resp.json())
            try:
                st.success(resp.json())
            except Exception:
                st.error("Backend returned invalid response")
                st.write(resp.text)


st.header("Ask a question")
# project_q = st.text_input("Project to query")
question = st.text_input("Your question")
ask = st.button("Ask")
if ask and question.strip():
    with st.spinner("Querying backend..."):
        resp = requests.post(f"{API_URL}/query", data={"project": project_q, "question": question})
    if resp.status_code==200:
        res = resp.json()
        st.subheader("Answer")
        st.write(res.get("answer"))
        st.subheader("Sources")
        for s in res.get("sources", []):
            st.markdown(f"**{s.get('source')}** — _{s.get('snippet')}..._")
    else:
        st.error(resp.text)

















