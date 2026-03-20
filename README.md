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
