# Install all required packages for RAG model with GPU support
# Run this after PyTorch installation is complete

Write-Host "Installing Python packages for RAG model..." -ForegroundColor Green

# Activate virtual environment
$venvPath = ".\venv_py312\Scripts\python.exe"

# Install core packages
& $venvPath -m pip install --upgrade pip

# Install ML/AI packages
& $venvPath -m pip install sentence-transformers
& $venvPath -m pip install transformers
& $venvPath -m pip install accelerate
& $venvPath -m pip install bitsandbytes

# Install data processing packages
& $venvPath -m pip install pandas
& $venvPath -m pip install numpy
& $venvPath -m pip install scipy

# Install NLP packages
& $venvPath -m pip install nltk
& $venvPath -m pip install tqdm

# Install PDF processing
& $venvPath -m pip install pymupdf

# Install Jupyter and related
& $venvPath -m pip install jupyter
& $venvPath -m pip install ipykernel
& $venvPath -m pip install ipywidgets

# Install Hugging Face
& $venvPath -m pip install huggingface-hub

# Register the kernel with Jupyter
& $venvPath -m ipykernel install --user --name=rag_py312 --display-name="Python 3.12 (RAG GPU)"

Write-Host "`n✅ All packages installed successfully!" -ForegroundColor Green
Write-Host "🚀 You can now use the 'Python 3.12 (RAG GPU)' kernel in your notebook" -ForegroundColor Cyan


#FET3BlbkFJdBV_7qSAOL3Ao6BbR5ORtprcRZVgJecsWrCA5eY_KTMg_DuGqMK8Wn7bs1QB09CMlgmzNOM10A