# Local RAG Implementation with GPU Acceleration

Retrieval-Augmented Generation (RAG) system for document Q&A, running entirely on local hardware with CUDA GPU support.

## Overview

This implementation processes PDF documents, generates embeddings, and uses similarity search to retrieve relevant context for question answering. All inference runs locally on NVIDIA GPUs.

**Key Features:**

- GPU-accelerated embedding generation (~1 min for 6,400+ chunks)
- Fast semantic search (~0.0001s per query)
- Local inference with quantized models
- Works with any PDF document

## Technical Stack

- **Python**: 3.12.10
- **CUDA**: 12.1 / 13.1
- **PyTorch**: 2.5.1+cu121
- **Embedding Model**: sentence-transformers/all-mpnet-base-v2 (768-dim)
- **LLM**: google/flan-t5-small (163 MB)
- **GPU**: NVIDIA GeForce RTX 3050 (4GB VRAM)

## Requirements

- NVIDIA GPU with CUDA support (4GB+ VRAM recommended)
- Python 3.12
- CUDA drivers installed

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create Python 3.12 virtual environment:

```bash
python -m venv venv_py312
venv_py312\Scripts\activate  # Windows
# source venv_py312/bin/activate  # Linux/Mac
```

3. Install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install dependencies:

```bash
pip install sentence-transformers transformers accelerate bitsandbytes pandas nltk pymupdf jupyter ipykernel ipywidgets huggingface-hub
```

5. Register Jupyter kernel:

```bash
python -m ipykernel install --user --name=rag_py312 --display-name="Python 3.12 (RAG GPU)"
```

## Usage

1. Place your PDF document in the project directory
2. Update the `pdf_path` variable in the notebook
3. Run all cells in `Local RAG.ipynb`

The notebook will:

- Extract text from PDF
- Chunk into sentences
- Generate embeddings on GPU
- Set up similarity search
- Load language model
- Enable interactive Q&A

## Performance

**Document Processing (1,313-page PDF):**

- Text extraction: ~30 seconds
- Chunking: 6,429 sentence chunks created
- Embedding generation: 1 min 15 sec (GPU)
- Query search: 0.0001 sec per query (GPU)

**Memory Usage:**

- Embeddings: ~50 MB
- Language model: ~163 MB
- Total GPU usage: <1 GB

## How It Works

1. **Document Preprocessing**: PDF → text → sentence chunks
2. **Embedding Generation**: Convert chunks to 768-dim vectors using SentenceTransformer
3. **Similarity Search**: Query embeddings compared via dot product, return top-k matches
4. **Context Retrieval**: Extract relevant text chunks based on similarity scores
5. **Answer Generation**: LLM generates response using retrieved context + query

## Limitations

- 4GB VRAM limits model size (using small models with quantization)
- Response quality depends on document content and retrieval accuracy
- Currently configured for single PDF processing

## License

MIT

## Acknowledgments

- Sentence Transformers library
- Hugging Face Transformers
- PyTorch team

Hugging Face token removed from the repository. Use an environment variable or notebook prompt at runtime instead.
