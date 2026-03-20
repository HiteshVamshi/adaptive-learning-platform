# Adaptive Learning Platform

This repository contains a locally runnable adaptive learning platform for CBSE Class 10 Mathematics. The build is being created in stages.

## Step 1

Step 1 creates the structured data layer used by the later search, RAG, mastery, recommendation, and agent modules:

- `concepts.csv`: chapters and concepts with hierarchy and metadata
- `questions.csv`: curated practice questions with difficulty, tags, and concept mapping
- `solutions.csv`: worked solutions and final answers
- `concept_relationships.csv`: prerequisite and related-concept edges
- `concept_graph.json`: serialized `networkx` graph for downstream modules
- `dataset_summary.json`: row counts and chapter-level coverage stats

## Run

```powershell
.\.venv\Scripts\python.exe scripts\generate_data.py
```

Artifacts are written to `artifacts/bootstrap_data`.

## Step 2

Step 2 adds hybrid retrieval over the generated question corpus:

- BM25 keyword retrieval
- FAISS-backed vector retrieval
- query understanding for difficulty and concept detection
- knowledge-graph-based concept expansion
- score fusion for hybrid ranking

Build the search index:

```powershell
.\.venv\Scripts\python.exe scripts\build_search_index.py --embedding-backend hash
```

Run a local search demo:

```powershell
.\.venv\Scripts\python.exe scripts\search_demo.py "hard irrational numbers proof" --top-k 3
```

## Step 3

Step 3 adds grounded RAG question answering:

- theory, question, and solution chunking
- FAISS retrieval over chunked context
- query expansion via the knowledge graph
- grounded answer generation with a local fallback backend
- explicit visibility into retrieved context and the synthesis prompt

Build the RAG index:

```powershell
.\.venv\Scripts\python.exe scripts\build_rag_index.py --embedding-backend hash
```

Run a RAG demo:

```powershell
.\.venv\Scripts\python.exe scripts\rag_demo.py "Why is 5 + sqrt(3) irrational?" --top-k 4 --generator-backend grounded
```

## Step 4

Step 4 adds the explainable concept mastery engine:

- simulated student attempts with correctness and response time
- deterministic concept scoring from accuracy, completion, speed, and challenge
- graph-aware mastery propagation from prerequisite and related concepts
- per-concept mastery snapshot plus mastery-over-time history
- artifact generation for later recommendations and dashboard views

Build mastery artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\build_mastery.py
```

Inspect the mastery outputs:

```powershell
.\.venv\Scripts\python.exe scripts\mastery_demo.py --top-k 5
```

## Step 5

Step 5 adds adaptive practice recommendations:

- ranks questions from learner mastery, graph structure, novelty, and difficulty alignment
- explains each recommendation with weakness and progression rationale
- limits over-concentration on a single concept
- writes recommendation artifacts for the future dashboard and practice UI

Build recommendations:

```powershell
.\.venv\Scripts\python.exe scripts\build_recommendations.py --top-k 10
```

Inspect recommendation outputs:

```powershell
.\.venv\Scripts\python.exe scripts\recommendation_demo.py --top-k 5
```

## Step 6

Step 6 adds modular content generation:

- generates new practice questions with answers and worked explanations
- generates concept explanations for a student audience
- generates summaries for concepts or chapters
- keeps prompt templates and generation backends separate from callers
- writes generation artifacts for later agent and UI integration

Build a sample content bundle:

```powershell
.\.venv\Scripts\python.exe scripts\build_generated_content.py --backend grounded
```

Run a generation demo:

```powershell
.\.venv\Scripts\python.exe scripts\content_generation_demo.py --task question --concept-id c_hcf_lcm --difficulty hard
```

## Step 7

Step 7 adds modular agents with tool traces:

- `TutorAgent` uses retrieval, mastery, and the knowledge graph to explain concepts
- `PracticeAgent` uses mastery, recommendations, and content generation to assign practice
- `QueryAgent` rewrites user queries using search-oriented concept and difficulty detection
- all agents call reusable tool adapters instead of embedding system logic directly
- each agent can write a structured trace for the future debug view

Run agent demos:

```powershell
.\.venv\Scripts\python.exe scripts\agent_demo.py --agent tutor --query "Explain why 5 + sqrt(3) is irrational"
.\.venv\Scripts\python.exe scripts\agent_demo.py --agent practice --query "Give me practice for my weak concepts"
.\.venv\Scripts\python.exe scripts\agent_demo.py --agent query --query "hard trigo proof question"
```

## Step 8

Step 8 adds a lightweight fine-tuning and adaptation demo:

- builds a local difficulty-calibration dataset from the curated bank and generated examples
- trains a small TF-IDF plus logistic regression model to classify difficulty style
- uses that calibrator to steer generated prompts toward a target difficulty
- compares baseline and adapted generations with explicit target-alignment probabilities
- writes artifacts for the future debug view and generation dashboard

Build the adaptation artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\build_fine_tuning.py
```

Run a fine-tuning demo:

```powershell
.\.venv\Scripts\python.exe scripts\fine_tuning_demo.py --concept-id c_hcf_lcm --difficulty hard
```

## Step 9

Step 9 adds the integrated Streamlit application:

- Search view showing BM25, vector, and hybrid retrieval signals
- Practice view showing weak concepts, recommendations, and generated questions
- Test view for recording manual attempts and previewing mastery updates
- Analysis dashboard for mastery trends and knowledge-graph inspection
- AI Tutor chat backed by modular agents and visible tool traces
- Content generation view for questions, explanations, summaries, and adaptation
- System debug view exposing retrieval context, agent traces, and raw artifact summaries

Run the app locally:

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```
