# Word Embedding and Semantic Similarity Explorer

## Mentor notes for project review and viva

These notes describe the current project implementation in this folder. They
are written to help explain what the program does, why its components were
chosen, and what its evaluation does and does not prove.

---

## 1. Project overview

### What did I develop?

I developed a small local web application that searches a collection of
sentences by **meaning**. A user types a query, the program converts it into a
numerical sentence embedding, and it ranks the dataset sentences by cosine
similarity to that embedding. The app also compares two sentences directly
and plots the dataset embeddings in two dimensions.

### Which subject area does it belong to?

It is a project in **Natural Language Processing (NLP)**, a branch of
Artificial Intelligence (AI). It uses **Machine Learning (ML)** and
**Deep Learning** because the sentence encoders are neural-network models
trained on language data. In this project I use pretrained models; I do not
train or fine-tune a model myself.

### Problem statement

Traditional search can depend heavily on the exact words a user types. A
sentence collection may contain the right information but express it with
different words. The problem is to retrieve sentences that are semantically
related to a query, even when their wording is different.

### Objective

The objective is to demonstrate a complete semantic-search workflow:

1. Encode sentences using a pretrained Sentence Transformer.
2. Compare the query embedding against dataset embeddings with cosine
   similarity.
3. Rank and display the most similar sentences.
4. Compare two sentences, visualize embeddings, and measure retrieval quality
   on a small labeled test set.

### Why is it needed, and where is it useful?

People do not always ask a question using the exact vocabulary present in a
document. A person might search for “a fix for a phone security weakness,”
while a document says “a patch repaired a vulnerability in the phone app.”
Semantic retrieval can help connect those expressions. The same general idea
can support document search, FAQ retrieval, recommendation, and other
language applications.

### Difference from keyword search

Keyword search mainly looks for overlapping terms or matches. This project
compares learned sentence vectors, so different but related words can still
produce a high score. A keyword search might not connect “football” with
“soccer”; a pretrained encoder can learn that these terms occur in related
contexts. It is not perfect: the word “bank” can mean a financial institution
or the land beside a river, and a model can sometimes confuse those senses.

---

## 2. Technology stack

The pinned direct dependencies are listed in `requirements.txt`. Some
packages used internally by Sentence Transformers, such as PyTorch and
Transformers, are installed as transitive dependencies rather than listed as
direct project requirements.

| Technology | What it is | Why we use it | Role in this project |
|---|---|---|---|
| Python | General-purpose programming language | Clear syntax and a strong machine-learning ecosystem | Runs the app, embedding helpers, evaluator, and tests |
| Streamlit (`streamlit==1.42.0`) | Python framework for data apps | Builds an interactive web UI without a separate frontend | Sidebar, tabs, text inputs, result table, chart, and metrics |
| Sentence Transformers (`sentence-transformers==3.4.1`) | Library for encoding text as sentence vectors | Provides pretrained sentence-embedding models through a simple API | Loads the two selectable encoders and runs `model.encode(...)` |
| PyTorch (`torch`, installed transitively) | Deep-learning tensor and model runtime | Sentence Transformer models use it for neural inference | Executes model inference; the code requests `device="cpu"` |
| Transformers (`transformers`, installed transitively) | Library of transformer architectures and pretrained model utilities | Supports model loading and tokenizer/model components | Used internally by Sentence Transformers |
| NumPy (`numpy==1.26.4`) | Library for arrays and numerical operations | Efficiently represents embeddings and performs vector math | Stores vectors and computes dot products, norms, and ranking scores |
| Pandas (`pandas==2.2.3`) | Library for tabular data | Conveniently builds tables and CSV files | Formats app results and evaluation output |
| Matplotlib (`matplotlib==3.10.0`) | Plotting library | Provides customizable plots | Draws the PCA scatter plot in the app |
| Scikit-learn (`scikit-learn==1.6.1`) | Machine-learning utilities | Includes PCA and a trusted similarity implementation for testing | PCA in the app; reference cosine calculation in tests |
| Pytest (`pytest==8.3.4`) | Python testing framework | Runs repeatable automated checks | Executes the four tests in `tests/test_similarity.py` |
| JSON (`json`, Python standard library) | Text format and parser for structured data | Keeps the small dataset and test labels simple to inspect | Loads `sentences.json` and `test_cases.json` |
| `pathlib` (Python standard library) | Cross-platform path utilities | Avoids fragile paths relative to the shell's current folder | Locates data and results relative to each Python file |
| `functools.lru_cache` (Python standard library) | Function-result cache | Avoids repeatedly loading the same large model | Caches supported Sentence Transformer model objects |

Other dependencies such as SciPy, Hugging Face Hub, Tokenizers, and
Safetensors may be installed as dependencies of the model stack. They are not
direct imports in the project's Python files.

---

## 3. Actual application interface

The interface is implemented in `app.py`. It loads the JSON sentences,
optionally appends custom text, creates or retrieves the corpus embeddings,
and then presents three tabs.

### Sidebar

- **Embedding model selector:** `st.selectbox` lets the user choose either
  `all-MiniLM-L6-v2` or `paraphrase-multilingual-MiniLM-L12-v2`. The chosen
  name is passed to the embedding helper for both corpus and query.
- **Top-K slider:** `st.slider` selects how many of the highest-ranked
  results to show. It ranges from 1 to 10 and starts at 5.
- **Add custom sentences:** `st.text_area` accepts one sentence per line.
  Non-empty lines are appended to this session's search and PCA corpus with
  the topic `Custom`. The built-in JSON file itself is not changed.

Streamlit reruns the script when controls change. To avoid repeating work,
`cached_embeddings` uses `st.cache_data` for corpus vectors, and
`embedder._load_model` uses `lru_cache` for the model object.

### Search tab

1. The user enters text in the query input.
2. `embed([query.strip()], model_name)[0]` encodes it into a vector.
3. `rank(query_vector, corpus_vectors)` computes a score for every corpus
   vector and returns indices in descending score order.
4. The app displays the selected Top-K sentences in a Pandas table. It rounds
   displayed scores to four decimal places.
5. `st.bar_chart` plots those displayed scores against their ranks.

The table includes rank, sentence, topic, and cosine similarity. The chart
visualizes the same displayed similarity values; it does not calculate a
separate score.

### Compare Two Sentences tab

The user enters two sentences. The app encodes them together by calling
`embed([sentence_a.strip(), sentence_b.strip()], model_name)`. It then takes
the two returned rows and calls `cosine_similarity` from `similarity.py`.
That helper returns one numeric score shown by `st.metric`.

Example pair that should be related:

- “A forward scored late in the soccer game.”
- “The striker found the net near the end of the match.”

Example pair that should be unrelated:

- “A forward scored late in the soccer game.”
- “The bakery sells warm bread and flaky pastries each morning.”

These are illustrative examples, not fixed scores. Actual output depends on
the selected model and the sentences.

### PCA Visualization tab

PCA means **Principal Component Analysis**. It is a method that takes data
with many dimensions and projects it into fewer dimensions while retaining
as much overall variation as possible in the selected components. A sentence
embedding can have hundreds of values, which cannot be directly drawn on a
flat chart. The app uses `PCA(n_components=2)` to create two coordinates per
sentence.

The scatter plot shows one point per corpus sentence. The point's color
indicates its topic (Sports, Food, Technology, Weather, Education, or Custom).
Points near one another have similar coordinates in this two-dimensional
projection, but closeness in the plot is only a visual aid. PCA loses
information, so the chart is not as exact as calculating the original
high-dimensional cosine score.

---

## 4. Explaining the supplied search example

The demonstration query is:

> A football player scored a goal

The displayed results are:

| Rank | Sentence | Score |
|---:|---|---:|
| 1 | The striker scored a decisive goal in the final minutes of the match. | 0.5871 |
| 2 | A forward found the net just before the soccer game ended. | 0.4179 |
| 3 | The coach changed the team's defense after halftime. | 0.1781 |
| 4 | Thousands of fans cheered as the home team lifted the trophy. | 0.1429 |
| 5 | A thick blanket of fog reduced visibility on the road. | 0.0994 |

### Why the first sentence ranks highest

It describes a player scoring a goal in a match, which is very close to the
query's meaning. It even shares the important words “scored” and “goal.” The
model's embedding for this sentence therefore points more in the same
direction as the query embedding than the other examples do.

### Why the second is also relevant

It uses different wording: “forward” instead of “football player,” “found
the net” instead of “scored a goal,” and “soccer game” instead of
“football.” These expressions describe a similar event. The model learned
semantic relationships from its pretraining data, so it can give this
paraphrase a higher score than unrelated sentences even without identical
wording.

### Why the weather sentence is low

The fog sentence is about visibility and weather. It has no meaningful
connection to a football player scoring. Its vector points in a different
direction, so its score is low compared with the relevant sports results.

### Why the scores differ

Cosine similarity measures how closely two vectors point in the same
direction. The top result is especially close in meaning and shares key
concepts with the query; the second expresses a related event in different
words; the remaining sentences are less relevant or unrelated. The exact
score is not a probability, and a score such as `0.5871` does not mean “58.71%
certain.” Scores depend on the embedding model, the query, and the text.

The model does not use a hand-written mapping that says “football means
soccer” or “net means goal.” It converts the whole sentence using patterns
learned during pretraining. In context, “football player,” “striker,”
“forward,” “soccer,” and “goal” occur in related language contexts, which
can place their sentence vectors near one another.

### Natural demonstration explanation

> “I entered ‘A football player scored a goal.’ The first result is the
> striker sentence because it describes almost the same event. The second
> result uses different wording—‘forward found the net’—but it means nearly
> the same thing, so it is also ranked near the top. The coach sentence is
> still about sports, but it describes defense rather than scoring, so its
> score is lower. The fog sentence is about weather and is unrelated, so it
> receives the lowest score shown. These values are cosine-similarity scores,
> not percentages or confidence probabilities.”

---

## 5. Complete internal workflow

```text
Built-in JSON sentences (+ optional custom sentences)
                         |
                         v
             Streamlit application (app.py)
                         |
             corpus sentence strings
                         v
       Sentence Transformer CPU model (embedder.py)
                         |
              numerical embeddings
                         |
User query -> embedder.py -> query embedding
                         |
                         v
     NumPy cosine similarity and sorting (similarity.py)
                         |
                         v
             Top-K ranked sentence indices
                         |
                         v
         Pandas result table and Streamlit bar chart
```

In the Compare tab, the app sends two user sentences to `embedder.py` and
compares their vectors using the same cosine helper. In the PCA tab, it gives
the corpus matrix to Scikit-learn's PCA and uses Matplotlib to plot the
projected points.

### How the Python files communicate

- `app.py` imports `MODEL_NAMES` and `embed` from `embedder.py`.
- `app.py` imports `cosine_similarity` and `rank` from `similarity.py`.
- `evaluate.py` imports `MODEL_NAMES` and `embed` from `embedder.py`, and
  imports `cosine_similarity` and `rank` from `similarity.py`.
- The app and evaluator each read the JSON data files directly.
- The test module imports the similarity helpers and checks them against
  expected behavior.
- `requirements.txt` tells pip which direct external packages to install.

The model does not get trained when I run the app. It is loaded from the
pretrained model repository (or its local Hugging Face cache), then used for
inference.

---

## 6. Explanation of every project source/data file

### `app.py`

**Purpose:** Defines and runs the Streamlit user interface.

Important elements:

- `load_sentences()` reads `data/sentences.json`.
- `cached_embeddings(...)` calls `embed(...)` and caches the corpus vectors
  with Streamlit.
- The sidebar handles model choice, Top-K, and custom sentences.
- The Search tab embeds a query, calls `rank`, and displays a table and bar
  chart.
- The Compare tab embeds two inputs and calls `cosine_similarity`.
- The PCA tab projects corpus embeddings using Scikit-learn PCA, then plots
  topic-colored points with Matplotlib.

Small excerpt from the actual search implementation:

```python
query_vector = embed([query.strip()], model_name)[0]
indices, scores = rank(query_vector, corpus_vectors)
```

### `embedder.py`

**Purpose:** Defines supported model names, caches model objects, and turns
sentences into NumPy vectors.

- `MODEL_NAMES` maps the UI's short model names to Hugging Face model IDs.
- `_load_model(model_name)` validates the choice and caches the model using
  `@lru_cache`. It explicitly passes `device="cpu"`.
- `embed(texts, model_name)` calls the model's `encode` method and returns a
  NumPy array.
- Empty input returns an empty two-dimensional array with the model's
  embedding width.

Excerpt:

```python
return SentenceTransformer(MODEL_NAMES[model_name], device="cpu")
```

This is model loading for inference; there is no training loop or
fine-tuning code in the project.

### `similarity.py`

**Purpose:** Implements cosine similarity and sorts corpus sentences by
similarity, using NumPy rather than a black-box ranking service.

The core score is calculated as:

```python
denominator = np.linalg.norm(a) * np.linalg.norm(b)
if denominator == 0:
    return 0.0
return float(np.dot(a, b) / denominator)
```

`cosine_similarity(...)` checks that vector lengths match and safely returns
zero if either vector has zero norm. `rank(query_vec, corpus_vecs)` checks
input dimensions, calculates all scores, and uses stable descending sorting.
It returns a list of original corpus indices and a parallel list of scores.

### `evaluate.py`

**Purpose:** Checks both models against the project's small labeled test set
and writes CSV results.

For each model it:

1. Embeds all 25 dataset sentences and all evaluation queries.
2. Calculates query-to-expected-sentence scores for all relevant IDs.
3. Calculates sentence-to-sentence scores for the listed unrelated pairs.
4. Ranks each query against the dataset and counts Top-1 and Top-3 hits.
5. Prints a failure message if none of the expected IDs appear in the Top 3.
6. Writes `results/evaluation_details.csv` and
   `results/model_comparison.csv`.

The summary's average related score is averaged over all query/relevant-ID
combinations. The unrelated average is over the unrelated pairs. Top-1
accuracy means that at least one expected relevant ID was ranked first.
Top-3 accuracy means that at least one expected relevant ID appeared within
the first three.

### `data/sentences.json`

**Purpose:** Stores the searchable corpus. It contains 25 original sentences
across five topics: Sports, Food, Technology, Weather, and Education. Every
entry has an ID, topic, and sentence. It includes paraphrase-like examples,
the ambiguous river-bank and financial-bank usage, and sentences that share
some words but differ in meaning.

### `data/test_cases.json`

**Purpose:** Stores evaluation labels. It contains 12 query cases, each with
a query ID, query text, relevant sentence IDs, and a short explanation. It
also contains 10 unrelated sentence pairs with explanations. These labels
are the reference used to compute the reported evaluation metrics.

### `tests/test_similarity.py`

**Purpose:** Tests the correctness and edge-case behavior of the custom
similarity helpers.

The four tests check:

1. Custom cosine similarity agrees with Scikit-learn's
   `cosine_similarity` on sample vectors.
2. Identical nonzero vectors return `1.0`.
3. Orthogonal vectors return `0.0`.
4. Zero vectors are safe, including through `rank`.

Run them with:

```powershell
python -m pytest -v
```

### `requirements.txt`

**Purpose:** Pins the versions of the project's direct third-party
dependencies so an installation uses the specified versions:

```text
sentence-transformers==3.4.1
numpy==1.26.4
streamlit==1.42.0
pandas==2.2.3
matplotlib==3.10.0
scikit-learn==1.6.1
pytest==8.3.4
```

Packages such as PyTorch and Transformers are required underneath
Sentence Transformers and are installed as dependencies. They are not
separately pinned in this project's requirements file.

---

## 7. Important technical concepts

### Natural Language Processing (NLP)

NLP is the area of computing that works with human language. Tokenization,
translation, text classification, and semantic search are examples. This
project uses NLP to represent and compare sentence meaning.

### Word embeddings and sentence embeddings

A **word embedding** represents a word as a list of numbers. A **sentence
embedding** represents a whole sentence as one vector. The project's
Sentence Transformer returns sentence embeddings, not an individual vector
for every word shown in the interface.

### Vector representation

A vector is an ordered list of numbers. For example, `[0.2, -0.5, 0.8]` is a
small illustrative vector. Real model embeddings have many more dimensions.
The numerical values are not human-readable labels; together they encode
patterns that help compare text.

### Pretrained model

A pretrained model has already learned language patterns from a large
training process performed before this project. The project downloads or
loads that model and uses it to encode text. I do **not** train or fine-tune
the model as part of this application.

### Semantic similarity

Semantic similarity is how close two texts are in meaning, rather than how
many exact words they share. “The car is very fast” and “The vehicle moves
quickly” use different words but express similar ideas.

### Cosine similarity, step by step

For vectors **A** and **B**:

```text
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

1. `A · B` is the dot product, which combines corresponding vector values.
2. `||A||` and `||B||` are the vector lengths (Euclidean norms).
3. Dividing by the product of lengths normalizes the dot product, making the
   score reflect direction rather than just vector size.
4. Similar directions tend toward `1`; perpendicular directions give `0`.
   Negative values can occur for general vectors.
5. The code returns `0.0` if either vector has zero length, avoiding division
   by zero.

For a simple example, `[1, 0]` and `[1, 0]` point in the same direction and
have cosine similarity `1`. `[1, 0]` and `[0, 1]` are perpendicular and have
similarity `0`.

### Top-K ranking

Top-K means returning the K best-scoring results. If K is 5, the app shows
the five corpus sentences with the largest cosine similarity scores. The
slider controls K; it does not change how embeddings are generated.

### Top-1 and Top-3 accuracy

- **Top-1 accuracy:** the proportion of evaluation queries whose expected
  relevant sentence appears at rank 1.
- **Top-3 accuracy:** the proportion of queries whose expected relevant
  sentence appears somewhere in the first three results.

For a query with more than one expected relevant ID, this evaluator counts a
hit when any expected ID appears in the respective cutoff. These metrics
describe only the supplied small test set.

### PCA

PCA reduces the original embedding dimensions to two coordinates for
visualization. It is useful for drawing and exploring groups, but it
compresses the data and may not preserve every pairwise distance exactly.

---

## 8. Testing and evaluation

### Unit tests

The test suite focuses on the handwritten cosine and ranking utilities. It
compares a sample calculation to Scikit-learn and tests identical,
orthogonal, and zero-vector inputs. The project test run completed with **4
tests passed**. These tests do not train models or test the semantic quality
of every possible query.

### Dataset evaluation

`test_cases.json` contains 12 query cases and 10 unrelated sentence pairs.
The evaluator runs both pretrained models against the same 25-sentence
corpus. It calculates:

- Average query-to-relevant-sentence cosine score.
- Average cosine score for the listed unrelated sentence pairs.
- Top-1 retrieval accuracy.
- Top-3 retrieval accuracy.

Current results from `results/model_comparison.csv`:

| Model | Average related cosine | Average unrelated cosine | Top-1 accuracy | Top-3 accuracy |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 0.654328 | 0.064937 | 91.67% (11/12) | 100% (12/12) |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0.745026 | 0.041274 | 100% (12/12) | 100% (12/12) |

On this particular test set, the multilingual model has a higher average
related score, a lower average unrelated-pair score, and higher Top-1
accuracy. Both models found at least one expected result in the Top 3 for all
12 queries. The first model's only Top-1 miss is Q12: it ranked S03 before
the expected S05, but S05 still appeared within its Top 3. The multilingual
model ranked S05 first for that query.

These results do not prove one model is universally better. The evaluation
set is small and manually labeled; a larger, more representative dataset
could produce different results. CSVs record the individual query outcomes
and the summary metrics.

---

## 9. Real-world usefulness and extensions

The current app demonstrates the core retrieval method on a small local
corpus. With appropriate data, validation, and user-interface work, the same
idea can be extended:

1. **Resume and job matching:** Compare a job description with candidate
   resume sections. For example, “Python data analysis experience” could
   retrieve a resume paragraph that says “built reports and cleaned datasets
   using Python.”
2. **Semantic document search:** Encode paragraphs from company documents
   and search them with natural-language questions. For example, “How many
   vacation days do new employees receive?” could retrieve the relevant
   benefits paragraph.
3. **Chatbots and FAQ systems:** Find the FAQ question most similar to a
   user's wording. For example, “I cannot access my account” could retrieve
   an FAQ about resetting a password.
4. **Recommendation systems:** Recommend items whose descriptions are
   similar to a user's interests or saved items. For example, someone who
   likes beginner gardening guides could be shown other beginner plant-care
   content.
5. **Educational content retrieval:** Find a lesson or explanation by
   concept rather than exact title. For example, “show me how fractions
   represent part of a whole” could retrieve a lesson titled “Understanding
   fractions with visual models.”

Production systems may need larger datasets, access controls, privacy
protections, relevance feedback, indexing, and human evaluation.

---

## 10. Demonstration script for the actual app

### 1. Open the application

**Action:** Start the app with `python -m streamlit run app.py` and show the
page title.

**Say:** “This is my Word Embedding and Semantic Similarity Explorer. It
demonstrates how a computer can search sentences by meaning using pretrained
sentence embeddings.”

### 2. Introduce the technology

**Action:** Point to the model selector in the sidebar.

**Say:** “The interface is built with Streamlit. Sentence Transformers
converts text into vectors, and NumPy calculates cosine similarity. The
model is pretrained, so I use it for inference rather than training it
myself. The implementation explicitly runs the model on CPU.”

### 3. Set up and search

**Action:** Choose a model, leave Top-K at 5, and enter:
`A football player scored a goal`

**Say:** “The query is converted into an embedding and compared with all
sentence embeddings in the dataset. The application sorts the scores and
shows the five closest results.”

### 4. Explain the ranking

**Action:** Point to the first three results and then the bar chart.

**Say:** “The striker sentence is first because it describes a very similar
scoring event. The forward finding the net is also relevant, even though its
wording differs. The coach sentence is about sports but a different event,
so it scores lower. The fog sentence is about weather and is unrelated. The
bars visualize the same scores shown in the table; these scores are
similarity values, not percentages.”

### 5. Demonstrate sentence comparison

**Action:** Open **Compare Two Sentences** and enter:

- First: `The striker scored a goal near the end of the match.`
- Second: `A forward found the net just before the soccer game ended.`

Then replace the second sentence with:
`A thick blanket of fog reduced visibility on the road.`

**Say:** “The app embeds both inputs and calculates their cosine similarity.
The paraphrase should be more similar than the weather sentence. The displayed
number depends on the selected model and is not a probability.”

### 6. Demonstrate PCA

**Action:** Open **PCA Visualization**.

**Say:** “Each point represents one sentence embedding. PCA reduces the
high-dimensional vectors to two coordinates for display, and colors show
topics. It helps visualize possible clusters, but the projection is not a
perfect map of every original similarity.”

### 7. Present evaluation

**Action:** Show the evaluation summary from
`results/model_comparison.csv`, or state the results.

**Say:** “I evaluated 12 labeled queries and 10 unrelated sentence pairs
with both models. In this small dataset, all-MiniLM-L6-v2 achieved 91.67
percent Top-1 and 100 percent Top-3 accuracy. The multilingual model achieved
100 percent for both. These values apply to these test cases only; they do
not guarantee the same performance on a larger real-world dataset.”

### 8. Conclude

**Say:** “This project shows a complete semantic-search pipeline: encode
sentences, calculate cosine similarity, rank the results, visualize them,
and evaluate retrieval. A natural next step would be testing on a larger
domain-specific dataset.”

---

## 11. Thirty likely viva questions and simple answers

1. **What is the title of your project?**  
   Word Embedding and Semantic Similarity Explorer.

2. **What problem does it address?**  
   It retrieves sentences with related meanings even when they use different
   words.

3. **Which domain does the project belong to?**  
   It is an NLP application within AI, using machine-learning and
   deep-learning models.

4. **What is the main objective?**  
   To encode sentences as vectors, compare them, and rank semantic matches.

5. **What is NLP?**  
   NLP is the area of computing that enables programs to work with human
   language.

6. **What is an embedding?**  
   It is a list of numbers that represents text in a form a computer can
   process.

7. **What is the difference between a word and sentence embedding?**  
   A word embedding represents one word; a sentence embedding represents the
   meaning of a whole sentence.

8. **Does this project train its own model?**  
   No. It loads pretrained models and uses them for inference; it does not
   train or fine-tune them.

9. **Which models can the user choose?**  
   `all-MiniLM-L6-v2` and `paraphrase-multilingual-MiniLM-L12-v2`.

10. **Why use Sentence Transformers?**  
    It provides models and a simple API designed to create useful sentence
    embeddings.

11. **Why use CPU?**  
    The app explicitly requests CPU execution, so a GPU is not required.

12. **What is cosine similarity?**  
    It measures the angle-based similarity between two vectors after
    normalizing by their lengths.

13. **What does a cosine score near 1 mean?**  
    The vectors point in similar directions, which often indicates related
    embeddings.

14. **Is a score of 0.8 an 80 percent probability?**  
    No. It is a cosine similarity value, not a probability or confidence
    percentage.

15. **How does the code handle a zero vector?**  
    It returns `0.0` when the product of vector norms is zero, avoiding
    division by zero.

16. **What does Top-K mean?**  
    It means the K highest-ranked results; the user selects K using the
    sidebar slider.

17. **What is the role of NumPy?**  
    It stores embedding arrays and performs vector operations such as dot
    products, norms, and sorting.

18. **What does `rank` return?**  
    It returns corpus indices sorted from highest to lowest similarity and
    their corresponding scores.

19. **Why is caching used?**  
    Model loading and embedding the corpus take time; caches avoid repeating
    that work unnecessarily during app reruns.

20. **What is Streamlit used for?**  
    It builds the interactive web interface directly in Python.

21. **What is PCA?**  
    Principal Component Analysis reduces data dimensions while retaining
    as much variation as possible in the chosen components.

22. **Why do you use PCA?**  
    The embeddings have many dimensions, so PCA makes two coordinates that
    can be plotted on a 2D chart.

23. **What does a point represent in the PCA chart?**  
    One sentence embedding, colored according to its topic.

24. **What is Top-1 accuracy in this evaluator?**  
    The proportion of queries for which an expected relevant sentence is
    ranked first.

25. **What is Top-3 accuracy?**  
    The proportion of queries for which an expected relevant sentence
    appears among the first three results.

26. **Which model performed better in your recorded evaluation?**  
    On this small test set, the multilingual MiniLM model had 100% Top-1 and
    Top-3 accuracy, versus 91.67% Top-1 and 100% Top-3 for all-MiniLM-L6-v2.

27. **What does the unrelated-pair score measure?**  
    It is the average cosine similarity between the sentence pairs labeled
    unrelated in the test JSON.

28. **Why do you use pytest and Scikit-learn in the tests?**  
    Pytest runs the checks; Scikit-learn provides a reference cosine result
    against which the manual NumPy implementation is compared.

29. **What is one limitation?**  
    The dataset is small, so the evaluation results may not represent
    performance on a large real-world collection.

30. **How could you extend this project?**  
    I could use a larger domain-specific corpus and apply semantic retrieval
    to documents, FAQs, education materials, or job descriptions.

---

## 12. Presentation versions

### 30-second introduction

> “My project is called Word Embedding and Semantic Similarity Explorer. It
> is an NLP application that searches sentences by meaning instead of only
> matching exact words. I use pretrained Sentence Transformer models to
> convert text into vectors, then NumPy calculates cosine similarity and
> ranks the closest sentences. The Streamlit app also compares two sentences
> and visualizes embeddings with PCA. The models run on CPU, and I evaluated
> them on a small labeled dataset.”

### 2-minute explanation

> “The problem I address is that a user and a document may express the same
> idea with different words. A keyword search may miss that connection. My
> application converts each sentence into a numerical sentence embedding
> using one of two pretrained Sentence Transformer models. These models are
> already trained; my project does not train or fine-tune them.
>
> When a user enters a query, the application embeds it and compares that
> vector with the dataset vectors using cosine similarity. The similarity
> helper uses a dot product divided by the product of the vector norms. It
> handles zero vectors safely. The app sorts the scores and displays the
> Top-K sentences in a table and bar chart.
>
> The other tabs compare two sentences directly and use PCA to project the
> sentence vectors into two dimensions for visualization. I use Streamlit
> for the interface, NumPy for vector operations, Pandas for tables and CSVs,
> Matplotlib for the scatter plot, and Scikit-learn for PCA. The models run
> on CPU.
>
> I evaluated 12 queries and 10 unrelated pairs. In the saved results,
> all-MiniLM-L6-v2 achieved 91.67 percent Top-1 and 100 percent Top-3
> accuracy. The multilingual model achieved 100 percent for both. This is a
> small demonstration dataset, so those results should not be treated as a
> broad benchmark.”

### Complete 5-minute presentation

> “Good morning. My project is titled Word Embedding and Semantic Similarity
> Explorer. It is an NLP project in the AI domain. Its purpose is to find
> sentences that are similar in meaning, even if they do not contain exactly
> the same words.
>
> A normal keyword search depends on matching words. For example, a user
> might say ‘a football player scored a goal,’ while the stored sentence
> says ‘a striker scored a decisive goal.’ A semantic-search system can
> connect these expressions by representing the whole sentence as a vector.
>
> My application uses Sentence Transformers and provides two pretrained
> models: all-MiniLM-L6-v2 and paraphrase-multilingual-MiniLM-L12-v2. These
> models have already learned language patterns before I use them. My
> application only loads them and performs inference; it does not train or
> fine-tune them. The code explicitly uses the CPU, so no GPU is required.
>
> I will demonstrate the search tab. I enter ‘A football player scored a
> goal.’ The app converts this query into an embedding. It compares that
> embedding to the dataset sentence embeddings using cosine similarity. The
> first result describes a striker scoring a goal, so it ranks highest. The
> second result says a forward found the net in a soccer game; the wording
> differs, but the meaning is similar. The coach's defense sentence is still
> about sports but is less related to scoring. The fog sentence is about
> weather and receives a low score.
>
> The formula is the dot product of two vectors divided by the product of
> their lengths. It compares direction. A score closer to one generally
> means the embedding directions are more similar, but it is not a
> probability or a percentage. My NumPy implementation also checks for
> zero vectors to avoid division by zero. The Top-K slider controls how many
> results are shown.
>
> In the Compare Two Sentences tab, I can enter a paraphrase and observe
> their similarity, then replace one sentence with an unrelated example. In
> the PCA tab, each point is a sentence embedding. PCA reduces the
> high-dimensional vectors to two coordinates so they can be plotted, and
> colors show topics. This is useful for exploration, but the projection
> loses information.
>
> The project has 25 dataset sentences from five topics, 12 labeled search
> queries, and 10 unrelated pairs. The evaluator writes detailed results
> and a model comparison to CSV. In my saved results, all-MiniLM-L6-v2 has
> an average related cosine score of 0.654328, an average unrelated-pair
> score of 0.064937, 91.67 percent Top-1 accuracy, and 100 percent Top-3
> accuracy. The multilingual model has 0.745026 average related score,
> 0.041274 average unrelated score, and 100 percent for both Top-1 and
> Top-3. These figures are for this small dataset only.
>
> I use pytest to check that my manual cosine implementation matches
> Scikit-learn and handles identical, orthogonal, and zero vectors. All four
> tests passed. Important limitations are the small dataset, possible
> ambiguity in language, and information loss in PCA. This technology can
> also be used for document search, FAQ retrieval, recommendation, education,
> and resume matching when supplied with suitable data.
>
> To conclude, my project demonstrates the full semantic retrieval process:
> text is encoded, vectors are compared, matches are ranked, results are
> visualized, and retrieval is evaluated. Thank you.”

### Short conclusion

> “This project demonstrates that pretrained sentence embeddings can help a
> program retrieve semantically related text even when the exact words
> differ. It combines a simple Streamlit interface with CPU-based inference,
> NumPy cosine similarity, visualization, testing, and a small evaluation.
> Its results are promising for the sample dataset, while larger real-world
> datasets would be needed for stronger conclusions.”
