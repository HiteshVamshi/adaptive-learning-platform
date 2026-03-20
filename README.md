# Adaptive Learning Platform

Locally runnable adaptive learning platform for CBSE Class 10 Mathematics. The system combines official curriculum structure, hybrid retrieval, RAG, mastery modeling, recommendations, modular agents, lightweight adaptation, and a Streamlit UI.

## Official Sources

The curriculum bootstrap in this repo is grounded in these official sources:

- CBSE Class X Mathematics syllabus 2024-25:
  `https://cbseacademic.nic.in/web_material/CurriculumMain25/Sec/Maths_Sec_2024-25.pdf`
- NCERT Class X Mathematics textbook index page:
  `https://ncert.nic.in/textbook.php?jemh1=ps-14`
- NCERT Class X Mathematics contents PDF:
  `https://ncert.nic.in/textbook/pdf/jemh1ps.pdf`

Those sources are used to define:

- official chapters and syllabus topics
- textbook chapter and section decomposition
- concept graph seed structure
- theory notes tied to syllabus wording and textbook sections
- textbook-backed bootstrap questions for uncovered chapters

## What The Platform Includes

- Data layer with official syllabus topics, textbook sections, theory notes, concepts, questions, and solutions
- Knowledge graph built with `networkx`
- Hybrid search with BM25 and FAISS vector retrieval
- RAG question answering with retrieved context inspection
- Explainable concept mastery engine
- Adaptive recommendation engine
- Content generation module
- Tool-using Tutor, Practice, and Query agents
- Lightweight difficulty adaptation demo
- Streamlit UI with search, practice, test, analysis, tutor, generation, curriculum/data, and debug views

## Environment Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

## Reproducible Build Order

Run the full build from a clean checkout in this order.

### 1. Generate official-source curriculum data and SQLite mirror

```powershell
.\.venv\Scripts\python.exe scripts\generate_data.py --sqlite-path artifacts\bootstrap_data\adaptive_learning.db
```

This writes:

- `artifacts/bootstrap_data/concepts.csv`
- `artifacts/bootstrap_data/questions.csv`
- `artifacts/bootstrap_data/solutions.csv`
- `artifacts/bootstrap_data/concept_relationships.csv`
- `artifacts/bootstrap_data/concept_graph.json`
- `artifacts/bootstrap_data/syllabus_topics.csv`
- `artifacts/bootstrap_data/textbook_sections.csv`
- `artifacts/bootstrap_data/theory_content.csv`
- `artifacts/bootstrap_data/dataset_summary.json`
- `artifacts/bootstrap_data/adaptive_learning.db`

### 2. Build the hybrid search index

```powershell
.\.venv\Scripts\python.exe scripts\build_search_index.py --embedding-backend hash
```

### 3. Build the RAG index

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_index.py --embedding-backend hash
```

### 4. Build mastery artifacts

```powershell
.\.venv\Scripts\python.exe scripts\build_mastery.py
```

### 5. Build recommendations

```powershell
.\.venv\Scripts\python.exe scripts\build_recommendations.py --top-k 10
```

### 6. Build generated content

```powershell
.\.venv\Scripts\python.exe scripts\build_generated_content.py --backend grounded
```

### 7. Build the lightweight fine-tuning / adaptation demo

```powershell
.\.venv\Scripts\python.exe scripts\build_fine_tuning.py
```

### 8. Launch Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Deployment

### Recommended Target

Use Streamlit Community Cloud for this app. It is a native fit for the current architecture because the repository already ships a Streamlit entrypoint, Python dependencies, and build artifacts.

Netlify is not the right target for the full application in its current form. Netlify's official deployment model is centered on static sites and serverless functions, not a long-running Python Streamlit runtime. If you want Netlify later, the practical route would be to split this project into:

- a separate Python API service for search, RAG, mastery, and recommendations
- a separate static frontend deployed on Netlify

### Streamlit Community Cloud Checklist

This repo is now prepared for Streamlit Community Cloud with:

- `requirements.txt` for deployment dependencies
- `.streamlit/config.toml` for headless cloud-friendly runtime settings
- `.streamlit/secrets.toml.example` as a template for future secret management
- `scripts/build_all.py` for one-command artifact generation
- automatic artifact bootstrap in `streamlit_app.py` if a fresh container starts without prebuilt artifacts

### Streamlit Cloud Steps

1. Push the repo to GitHub.
2. In Streamlit Community Cloud, create a new app from this repository.
3. Set the main file path to `streamlit_app.py`.
4. In the app settings, keep Python 3.11 if prompted.
5. Optionally set these environment variables:
   - `ADAPTIVE_LEARNING_EMBEDDING_BACKEND=hash`
   - `ADAPTIVE_LEARNING_AUTO_BUILD=true`
6. Deploy.

### Predeploy Build Command

For a clean rebuild before pushing:

```powershell
.\.venv\Scripts\python.exe scripts\build_all.py --force --embedding-backend hash
```

### Local Production-like Run

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\build_all.py --force --embedding-backend hash
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Optional Smoke Checks

```powershell
.\.venv\Scripts\python.exe scripts\search_demo.py "probability simple event question" --top-k 3
.\.venv\Scripts\python.exe scripts\rag_demo.py "What is the classical definition of probability?" --top-k 4 --generator-backend grounded
.\.venv\Scripts\python.exe scripts\mastery_demo.py --top-k 5
.\.venv\Scripts\python.exe scripts\recommendation_demo.py --top-k 5
.\.venv\Scripts\python.exe scripts\content_generation_demo.py --task question --concept-id c_classical_probability --difficulty easy
.\.venv\Scripts\python.exe scripts\fine_tuning_demo.py --concept-id c_hcf_lcm --difficulty hard
```

## Storage Model

This repo uses a hybrid local storage model:

- CSV and JSON artifacts in `artifacts/` are reproducible build outputs and easy to inspect in Git.
- SQLite in `artifacts/bootstrap_data/adaptive_learning.db` mirrors the curriculum dataset for local querying and UI status inspection.

For a laptop-scale portfolio system, this is the right tradeoff. If you later add persistent learner state, store mutable runtime data in SQLite as well.

## Streamlit Views

- `Curriculum & Data`: official-source provenance, syllabus topics, textbook sections, theory notes, question coverage, and SQLite status
- `Search`: BM25 vs vector vs hybrid retrieval signals
- `Practice`: mastery-driven recommendations and generated practice
- `Test`: manual attempts with live mastery recomputation
- `Analysis Dashboard`: mastery trends and knowledge graph view
- `AI Tutor`: tool-using Tutor, Practice, and Query agents
- `Content Generation Demo`: generated questions, explanations, summaries, and adaptation comparison
- `System Debug View`: raw retrieval, RAG context, traces, and summary artifacts

## Current Data Coverage

After running the build, inspect:

- `artifacts/bootstrap_data/dataset_summary.json`
- the `Curriculum & Data` page in Streamlit
- the SQLite `dataset_metadata` table

Those surfaces show:

- official chapter coverage
- question counts by chapter
- question counts by source
- theory note coverage
- SQLite table row counts

## Notes

- The default `hash` embedding backend is the safest fully local fallback for reproducible offline runs.
- The code is structured so `sentence-transformers` can be used when available.
- Some question bank content is still deterministic bootstrap content derived from official syllabus and textbook section structure. That is deliberate and traceable through the `source` column.
