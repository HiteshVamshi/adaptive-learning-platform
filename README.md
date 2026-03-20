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
