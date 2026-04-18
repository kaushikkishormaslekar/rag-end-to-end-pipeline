# RAG Pipeline: Local, Fast, GPU-Ready

<p align="center">
	<strong>Retrieval-Augmented Generation for PDF Q&A on your own machine.</strong><br/>
	Build embeddings on GPU, retrieve context in milliseconds, and answer questions with a local model.
</p>

<p align="center">
	<img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python"/>
	<img alt="CUDA" src="https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia"/>
	<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-GPU-red?style=for-the-badge&logo=pytorch"/>
	<img alt="Notebook" src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter"/>
</p>

---

## Why This Repo Exists

Most RAG tutorials are either too abstract or rely heavily on cloud services.
This project is practical: you give it a PDF, it builds an embedding index locally, and answers questions using retrieved context.

### What makes it interesting

- Runs fully local (documents stay on your machine)
- Uses GPU acceleration for faster embedding generation
- Keeps architecture simple enough to inspect end-to-end in one notebook
- Easy to extend into API or UI later

---

## Architecture At A Glance

```mermaid
flowchart LR
		A[PDF Document] --> B[Text Extraction]
		B --> C[Chunking]
		C --> D[Embedding Model\nall-mpnet-base-v2]
		D --> E[Vector Store\nCSV + Tensor embeddings]

		Q[User Query] --> Q1[Query Embedding]
		Q1 --> S[Similarity Search\nTop-k Context]
		E --> S
		S --> G[LLM Generation\nflan-t5-small]
		G --> OUT[Grounded Answer]
```

---

## Tech Stack

- Python 3.12
- PyTorch + CUDA
- sentence-transformers
- transformers + accelerate + bitsandbytes
- PyMuPDF
- pandas / numpy / nltk
- Jupyter Notebook workflow

Core files in this repo:

- `pipeline.ipynb`: end-to-end RAG pipeline notebook
- `requirements.txt`: dependency list
- `install_requirements.ps1`: setup helper script
- `text_chunks_and_embeddings_df.csv`: generated embeddings/chunks output

---

## Quick Start

### 1) Clone and enter the project

```bash
git clone <your-repo-url>
cd retrieve-augmented-generation/rag-pipeline
```

### 2) Create and activate virtual environment

```powershell
python -m venv venv_py312
venv_py312\Scripts\Activate.ps1
```

### 3) Install GPU-enabled PyTorch (CUDA 12.1 example)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4) Install project dependencies

```powershell
pip install -r requirements.txt
```

### 5) Launch notebook

```powershell
jupyter notebook pipeline.ipynb
```

---

## Run Flow

1. Put your PDF in the repo folder.
2. Open `pipeline.ipynb`.
3. Set the PDF path variable.
4. Run cells in order.
5. Ask questions and inspect retrieved context.

Pipeline steps:

1. Extract text from PDF
2. Split text into chunks
3. Generate chunk embeddings on GPU
4. Compute query embedding
5. Retrieve top-k relevant chunks
6. Generate an answer using retrieved context

---

## Performance Snapshot

Observed on a 1,313-page PDF with RTX 3050 (4 GB VRAM):

- Text extraction: ~30s
- Chunk creation: 6,429 chunks
- Embedding generation: ~1m 15s (GPU)
- Retrieval latency: ~0.0001s per query

Approximate memory footprint:

- Embeddings: ~50 MB
- LLM weights: ~163 MB
- Total GPU memory used: typically under 1 GB

---

## Project Structure

```text
rag-pipeline/
|- install_requirements.ps1
|- pipeline.ipynb
|- requirements.txt
|- text_chunks_and_embeddings_df.csv
|- Hands-On-LLM.pdf
```

---

## Customization Ideas

- Swap embedding model for speed vs quality experiments
- Add reranking step after initial retrieval
- Replace CSV persistence with FAISS or a vector DB
- Add a simple API (FastAPI) or chat UI (Gradio)
- Add evaluation notebook for retrieval quality and answer faithfulness

---

## Notes

- 4 GB VRAM works, but model size is constrained
- Better retrieval quality usually gives better answers than bigger generation models
- Keep credentials out of source control; pass tokens via environment variables when required

---

## License

MIT

## Credits

- Hugging Face ecosystem
- Sentence Transformers
- PyTorch
