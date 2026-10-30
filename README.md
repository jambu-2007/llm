# Word Embedding and Semantic Similarity Explorer

This local, CPU-only Python project turns sentences into numerical vectors and
lets you explore how closely their meanings match. The Streamlit app includes
semantic search, two-sentence comparison, and a PCA visualization. Two
Sentence Transformers models are available: `all-MiniLM-L6-v2` and
`paraphrase-multilingual-MiniLM-L12-v2`.

## Requirements and installation

Use Python 3.12 on Windows. From the `TERMINAL` directory, activate the
existing virtual environment; this project does not create another one:

```powershell
.\venv\Scripts\Activate.ps1
cd .\semantic_explorer
python -m pip install -r requirements.txt
```

The first model use downloads the selected model from Hugging Face and needs
internet access. The model is cached locally after download. Inference is
explicitly configured for CPU; no GPU is required.

## How embeddings work

Sentence Transformers encode each input sentence as a dense vector. The
embedding captures semantic patterns learned from language examples, so
sentences with similar meanings often have vectors pointing in similar
directions even when they use different words. This is not a guarantee: a
model can miss context, ambiguity, or domain-specific meaning.

## Cosine similarity

For vectors **A** and **B**, cosine similarity is:

```text
similarity(A, B) = dot(A, B) / (norm(A) * norm(B))
```

The score compares the angle between vectors: values nearer to `1` indicate
similar directions, values near `0` indicate little directional similarity,
and negative values indicate opposing directions. This project returns `0`
when either vector has zero length to avoid division by zero.

## Run the Streamlit app

From `semantic_explorer`:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Choose a model and Top-K value in the
sidebar. Enter one custom sentence per line to include it in search and PCA.
The tabs provide semantic search, pairwise comparison, and a two-dimensional
PCA view colored by topic.

## Run the evaluation

```powershell
python evaluate.py
```

The script evaluates all query labels and unrelated sentence pairs for both
models. It prints per-model averages, Top-1 and Top-3 accuracy, and any query
whose expected sentence is missing from the Top 3. It writes
`results/evaluation_details.csv` and `results/model_comparison.csv`.

## Run the tests

```powershell
python -m pytest -v
```

The similarity tests compare the NumPy calculation with scikit-learn and
check identical, orthogonal, and zero-vector behavior.

## Project workflow

1. Load the sentence collection from `data/sentences.json`.
2. Encode sentences using the selected, cached CPU model.
3. Compare a query or sentence pair with the manual cosine similarity helper.
4. Sort search results by score; use PCA to project corpus vectors into two
   dimensions for display.
5. Evaluate labeled queries and unrelated pairs, then save the CSV summaries.

## Limitations

- The included dataset and evaluation labels are small demonstrations, not a
  representative benchmark.
- Similarity is a model score, not proof that two statements are equivalent
  or factually consistent.
- Ambiguous words (such as “bank”) can still be difficult for a sentence
  embedding model, especially without surrounding context.
- PCA compresses high-dimensional vectors and can hide meaningful differences.
- The multilingual model may be slower and larger than the English-focused
  model, and either model requires an initial internet download.
