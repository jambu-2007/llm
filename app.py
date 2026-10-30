"""Streamlit interface for exploring sentence embeddings and similarity."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from embedder import MODEL_NAMES, embed
from similarity import cosine_similarity, rank

PROJECT_DIR = Path(__file__).resolve().parent
SENTENCES_PATH = PROJECT_DIR / "data" / "sentences.json"


def load_sentences() -> list[dict[str, str]]:
    """Read the built-in sentence collection from JSON."""
    with SENTENCES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def cached_embeddings(
    sentences: tuple[str, ...], model_name: str
) -> np.ndarray:
    """Cache corpus embeddings between Streamlit reruns."""
    return embed(sentences, model_name)


st.set_page_config(page_title="Word Embedding and Semantic Similarity Explorer", layout="wide")
st.title("Word Embedding and Semantic Similarity Explorer")
st.write(
    "Explore how sentence embeddings capture meaning, then compare their cosine "
    "similarity across the sample collection."
)

with st.sidebar:
    st.header("Explorer settings")
    model_name = st.selectbox("Embedding model", list(MODEL_NAMES))
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5)
    custom_text = st.text_area(
        "Add custom sentences",
        placeholder="Enter one sentence per line",
        help="Custom sentences are included in search and PCA for this session.",
    )

sentences = load_sentences()
for index, text in enumerate(line.strip() for line in custom_text.splitlines()):
    if text:
        sentences.append(
            {"id": f"C{index + 1:02d}", "topic": "Custom", "sentence": text}
        )

sentence_texts = tuple(item["sentence"] for item in sentences)
with st.spinner("Loading the model and preparing sentence embeddings..."):
    corpus_vectors = cached_embeddings(sentence_texts, model_name)

search_tab, compare_tab, pca_tab = st.tabs(
    ["Search", "Compare Two Sentences", "PCA Visualization"]
)

with search_tab:
    st.subheader("Find sentences by meaning")
    query = st.text_input("Enter a query", placeholder="For example: students studying for an exam")
    if query.strip():
        query_vector = embed([query.strip()], model_name)[0]
        indices, scores = rank(query_vector, corpus_vectors)
        result_rows = [
            {
                "Rank": position,
                "Sentence": sentences[index]["sentence"],
                "Topic": sentences[index]["topic"],
                "Cosine similarity": round(scores[position - 1], 4),
            }
            for position, index in enumerate(indices[:top_k], start=1)
        ]
        results = pd.DataFrame(result_rows)
        st.dataframe(results, use_container_width=True, hide_index=True)
        chart_data = results.set_index("Rank")["Cosine similarity"]
        st.bar_chart(chart_data)
    else:
        st.info("Enter a query to see the closest sentences.")

with compare_tab:
    st.subheader("Compare two sentences")
    sentence_a = st.text_input("First sentence", key="sentence_a")
    sentence_b = st.text_input("Second sentence", key="sentence_b")
    if sentence_a.strip() and sentence_b.strip():
        pair_vectors = embed([sentence_a.strip(), sentence_b.strip()], model_name)
        score = cosine_similarity(pair_vectors[0], pair_vectors[1])
        st.metric("Cosine similarity", f"{score:.4f}")
        st.caption("A higher score indicates more similar embedding directions.")
    else:
        st.info("Enter both sentences to compare them.")

with pca_tab:
    st.subheader("Sentence embeddings in two dimensions")
    if len(sentences) < 2:
        st.info("Add at least two sentences to create a PCA chart.")
    else:
        points = PCA(n_components=2).fit_transform(corpus_vectors)
        chart = pd.DataFrame(
            {
                "PC 1": points[:, 0],
                "PC 2": points[:, 1],
                "Topic": [item["topic"] for item in sentences],
                "Sentence": sentence_texts,
            }
        )
        figure, axis = plt.subplots(figsize=(9, 6))
        for topic, topic_rows in chart.groupby("Topic", sort=True):
            axis.scatter(topic_rows["PC 1"], topic_rows["PC 2"], label=topic, alpha=0.8)
        axis.set_xlabel("Principal component 1")
        axis.set_ylabel("Principal component 2")
        axis.legend(title="Topic")
        axis.grid(alpha=0.2)
        st.pyplot(figure)
        plt.close(figure)
