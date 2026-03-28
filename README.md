# HELIOS-Med

**Hybrid Engine for Language-driven Intelligent Orchestration of Semantic Medical Knowledge**

HELIOS-Med is a local, agentic Retrieval-Augmented Generation (RAG) system for healthcare and medical-policy document analysis. It ingests PDF documents, indexes them in a Chroma vector store, retrieves relevant passages for a user question, and generates grounded answers with source citations through a Chainlit chat interface.

## What It Does

- Ingests PDF-based medical or healthcare-policy documents
- Cleans and chunks document text for retrieval
- Stores embeddings in a local Chroma database
- Retrieves relevant context with Max Marginal Relevance (MMR) search
- Uses a LangGraph workflow to orchestrate retrieval, relevance validation, and answer generation
- Generates cited answers with a local Ollama model
- Exposes the system through a simple Chainlit UI

## Architecture Overview

The current pipeline is organized as:

1. **Document ingestion**  
   `modules/ingestion.py` loads PDFs from `data/`, preprocesses the text, chunks it, and stores embeddings in `chroma_db/`.

2. **Preprocessing**  
   `modules/preprocess.py` removes repeated page markers, normalizes whitespace, and cleans OCR-style formatting artifacts.

3. **Retrieval + orchestration**  
   `modules/engine.py` defines the HELIOS state graph using LangGraph:
   - `retrieve`
   - `format_sources`
   - `grade_relevance`
   - `generate`

4. **User interface**  
   `main.py` runs a Chainlit chatbot that displays:
   - final grounded answer
   - confidence score
   - reasoning trace
   - verified source list
   - retrieved document chunks in the side panel

## Tech Stack

- Python 3.10+
- LangChain
- LangGraph
- Chainlit
- ChromaDB
- Hugging Face embeddings (`all-MiniLM-L6-v2`)
- Ollama for local LLM inference
- PyPDF for PDF loading

## Repository Structure

```text
HELIOS-MED/
|-- data/                  # Input PDF documents
|-- chroma_db/             # Persisted Chroma vector database
|-- modules/
|   |-- engine.py          # LangGraph orchestration engine
|   |-- ingestion.py       # PDF ingestion and vector indexing
|   |-- preprocess.py      # Text cleanup utilities
|   |-- prompts.py         # Prompt helpers / prompt-related code
|   |-- test.py            # Simple CLI smoke test
|-- main.py                # Chainlit application entrypoint
|-- requirements.txt
|-- pyproject.toml
|-- chainlit.md            # Chainlit welcome content
```

## Prerequisites

Before running HELIOS-Med, make sure you have:

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running locally
- A supported Ollama model pulled locally, such as:

```powershell
ollama pull qwen2.5:3b
```

## Installation

You can install dependencies with either `uv` or `pip`.

### Option 1: Using `uv`

```powershell
uv sync
```

### Option 2: Using `pip`

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

HELIOS-Med reads runtime settings from environment variables. If not set, it falls back to sensible defaults.

| Variable | Default | Purpose |
|---|---|---|
| `HELIOS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model used for indexing and retrieval |
| `HELIOS_CHROMA_DIR` | `./chroma_db` | Path to the persistent vector store |
| `HELIOS_LLM_MODEL` | `qwen2.5:3b` | Ollama model used for answer generation |
| `HELIOS_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server endpoint |
| `HELIOS_CHUNK_K` | `6` | Number of retrieved chunks used for answering |

Example PowerShell session:

```powershell
$env:HELIOS_LLM_MODEL="qwen2.5:3b"
$env:HELIOS_OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

## How To Run

### 1. Add your documents

Place one or more PDF files inside the `data/` directory.

The repo already includes a demo file:

- `data/helios_healthcare_policy_demo.pdf`

### 2. Build the vector database

```powershell
python modules/ingestion.py
```

This will:

- load PDFs from `data/`
- preprocess and chunk the text
- generate embeddings
- persist the vector store to `chroma_db/`

### 3. Run a quick CLI smoke test

```powershell
python modules/test.py
```

This runs a sample question against the LangGraph workflow and writes a structured result to `test_output.json`.

### 4. Launch the chat interface

```powershell
chainlit run main.py
```

Then open the local Chainlit URL shown in the terminal and start asking questions.

## Example Questions

- What is the aim of this document?
- What is the purpose of the policy document?
- What steps are involved in policy development?
- How are healthcare policies approved?
- How is policy compliance monitored?
- What future trends are described in healthcare policy governance?

## Current Workflow Logic

At runtime, HELIOS-Med:

1. receives a user question
2. retrieves the top document chunks from Chroma
3. performs a basic relevance check
4. formats source references
5. prompts the LLM to answer using only the retrieved sources
6. returns the answer with citations and a lightweight reasoning trace

## Output Features

The Chainlit app currently shows:

- a grounded answer
- source references with page numbers
- retrieved chunk previews in the side panel
- a simple confidence heuristic based on retrieved document count
- a trace of retrieval and validation steps

## Notes and Limitations

- The current relevance grading step is lightweight and mainly checks whether documents were retrieved.
- Confidence scoring is heuristic, not a calibrated evaluation metric.
- Answer quality depends heavily on document quality, chunking, and the selected Ollama model.
- The system is currently optimized for local experimentation and demos rather than production deployment.

## Future Improvement Ideas

- stronger relevance grading using an LLM or classifier
- citation verification against exact supporting spans
- multi-document comparison and summarization
- better evaluation pipelines for factuality and retrieval quality
- support for domain-specific medical embedding models
- Dockerized deployment and reproducible setup

## Why HELIOS-Med?

HELIOS-Med is designed as a compact research and prototyping framework for trustworthy medical-document question answering. Its focus is not just generation, but orchestration: combining retrieval, validation, explainability, and local inference into a single workflow that can be inspected and extended.


