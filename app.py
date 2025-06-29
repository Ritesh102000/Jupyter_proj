import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from langdetect import detect
import pandas as pd
from datetime import datetime
import os

# --- Normalize vectors ---
def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# --- Load sentence transformer model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

model = load_model()

# --- Load multilingual FAQ data ---
@st.cache_data
def load_data():
    with open("jupiter_faqs_multilingual.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    en_questions, hi_questions = [], []
    en_answers, hi_answers = [], []
    en_meta, hi_meta = [], []

    for qa in faq_data["FAQs"]:
        en_questions.append(qa["question"])
        en_answers.append(qa["answer"])
        en_meta.append(qa)

        hi_questions.append(qa["question_hi"])
        hi_answers.append(qa.get("answer_hi", qa["answer"]))
        hi_meta.append(qa)

    en_embeddings = normalize(model.encode(en_questions, show_progress_bar=True))
    hi_embeddings = normalize(model.encode(hi_questions, show_progress_bar=True))

    en_index = faiss.IndexFlatIP(en_embeddings.shape[1])
    en_index.add(en_embeddings)

    hi_index = faiss.IndexFlatIP(hi_embeddings.shape[1])
    hi_index.add(hi_embeddings)

    return {
        "en": (en_questions, en_answers, en_meta, en_index, en_embeddings),
        "hi": (hi_questions, hi_answers, hi_meta, hi_index, hi_embeddings)
    }

faq_data = load_data()

# --- Get best match from FAISS ---
def get_best_match(query, top_k=3, threshold=0.6, force_lang=None):
    debug = {}
    try:
        detected = detect(query)
        debug["raw_detected_language"] = detected
        lang = force_lang if force_lang else ('hi' if detected == 'hi' else 'en')
    except:
        lang = force_lang if force_lang else 'en'

    debug["used_language"] = lang
    debug["query_length"] = len(query.strip())

    questions, answers, meta, index, embeddings = faq_data.get(lang, faq_data["en"])

    if len(query.strip()) < 4 or not any(c.isalpha() for c in query):
        debug["reason"] = "Too short or no alphabetic chars"
        return None, lang, debug

    if query.lower() in ["hi", "hello", "ok", "okay", "how are you", "test"]:
        debug["reason"] = "Generic or small talk query"
        return None, lang, debug

    q_embed = normalize(model.encode([query]))
    debug["query_embedding_preview"] = q_embed[0][:5].tolist()

    D, I = index.search(q_embed, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        debug["cosine_similarity"] = float(score)
        if score < threshold:
            debug["reason"] = f"Low similarity: {score:.2f} < threshold {threshold}"
            return None, lang, debug
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "meta": meta[idx],
            "score": float(score)
        })

    debug["reason"] = "Match found"
    return results, lang, debug

# --- Streamlit UI ---
st.set_page_config(page_title="Jupiter FAQ Assistant", page_icon="🤖")
st.title("🤖 Jupiter FAQ Assistant")
st.markdown("Ask your question in **English or Hindi**. I’ll find the closest FAQ and ask for your feedback.")

# Language override
override_lang = st.selectbox("🌐 Force Language (optional)", ["Auto", "English", "Hindi"])
force_lang = {'English': 'en', 'Hindi': 'hi'}.get(override_lang, None)

# User query
user_query = st.text_input("🧠 Your question", "")

# Session state to store feedback across queries
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

if user_query:
    results, lang, debug = get_best_match(user_query, top_k=3, force_lang=force_lang)

    with st.expander("🔍 Debug Info"):
        st.json(debug)

    if results:
        st.markdown("### 🔎 Top 3 Matched FAQs")
        feedback = []

        for i, match in enumerate(results):
            st.markdown(f"#### 🔹 Match #{i + 1}")
            st.markdown(f"**Q:** {match['question']}")
            st.markdown(f"**A:** {match['answer']}")
            st.caption(f"Similarity Score: {round(match['score'] * 100, 2)}%")
            correctness = st.radio(
                f"✅ Is this answer correct for Match #{i + 1}?",
                ["Yes", "No"],
                key=f"feedback_{i}"
            )
            feedback.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": user_query,
                "matched_question": match['question'],
                "answer": match['answer'],
                "score": match['score'],
                "is_correct": correctness,
                "language": lang
            })

        if st.button("📊 Submit Feedback"):
            st.session_state.feedback_log.extend(feedback)

            # Save session feedback to cumulative CSV
            df_all = pd.DataFrame(st.session_state.feedback_log)
            cumulative_file = "faq_feedback_cumulative.csv"
            if os.path.exists(cumulative_file):
                existing_df = pd.read_csv(cumulative_file, encoding="utf-8-sig")
                df_all = pd.concat([existing_df, df_all], ignore_index=True).drop_duplicates()

            df_all.to_csv(cumulative_file, index=False, encoding="utf-8-sig")

            # Also allow download of current session batch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_file = f"faq_feedback_{timestamp}.csv"
            session_df = pd.DataFrame(feedback)

            st.success("✅ Feedback submitted and saved!")
            st.download_button(
                label="⬇️ Download This Feedback Batch",
                data=session_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=session_file,
                mime="text/csv"
            )

            # Show accuracy summary
            st.markdown("### 📈 Summary")
            summary = df_all.groupby("is_correct").size().to_dict()
            total = sum(summary.values())
            st.write(f"Total Feedback Entries: {total}")
            st.write({k: f"{v} ({(v / total) * 100:.2f}%)" for k, v in summary.items()})
    else:
        st.warning("🙁 Sorry, I couldn't confidently answer that. Try rephrasing your question.")
