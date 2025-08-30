# Environment Setup Guide - LoRA Persona Project

## Overview
This project uses `uv` as a modern, fast alternative to `pip` and `venv` for Python package and environment management.

## Why `uv`?
- **10-100x faster** than pip
- Better dependency resolution
- Built-in virtual environment management
- Compatible with existing requirements.txt
- Written in Rust for maximum performance

## Quick Start

### 1. Environment Activation
```powershell
# Activate the virtual environment
.venv\Scripts\activate

# Verify you're in the right environment
python setup_env.py check
```

### 2. Start Development
```powershell
# Start Jupyter Lab
jupyter lab

# In Jupyter, select the "LoRA Persona (uv)" kernel
```

## Environment Management Commands

### Basic Commands
```powershell
# Check environment status
python setup_env.py check

# Complete environment setup (first time)
python setup_env.py setup

# Install/update packages
python setup_env.py install
```

### uv Commands
```powershell
# Install new packages
uv pip install package_name

# Install from requirements.txt
uv pip install -r requirements.txt

# Create new virtual environment
uv venv --python 3.11

# List installed packages
uv pip list

# Update all packages
uv pip install --upgrade -r requirements.txt
```

## Installed Packages

### Core ML/AI Libraries
- `torch` (CUDA-enabled for RTX A4000)
- `transformers` (HuggingFace)
- `datasets` (Data loading)
- `tokenizers` (Text processing)

### LoRA and Fine-tuning
- `peft` (Parameter-Efficient Fine-Tuning)
- `accelerate` (Training acceleration)
- `bitsandbytes` (Quantization)
- `trl` (Transformer Reinforcement Learning)

### Data Science
- `pandas`, `numpy` (Data manipulation)
- `matplotlib`, `seaborn` (Visualization)
- `scikit-learn` (ML utilities)

### Development
- `jupyter`, `ipykernel` (Notebook environment)
- `ipywidgets` (Interactive widgets)

## GPU Setup

### For RTX A4000
1. **CUDA Version**: The environment includes PyTorch with CUDA 12.1 support
2. **Memory**: Optimized for 16GB VRAM
3. **Verification**: Run `python setup_env.py check` to verify GPU detection

### Troubleshooting GPU Issues
If GPU is not detected:
1. Update NVIDIA drivers
2. Restart system after PyTorch installation
3. Check CUDA installation: `nvidia-smi`

## Project Structure
```
Lora-persona/
├── .venv/                          # Virtual environment (uv managed)
├── .gitignore                      # Git ignore (protects personal data)
├── requirements.txt                # Package dependencies
├── setup_env.py                    # Environment management script
├── slack_data_processor.py         # Slack data processing
├── LoRA-slack-persona-local.ipynb  # Main training notebook
├── sampledata/                     # Your Slack exports (gitignored)
├── processed_datasets/             # Processed training data (gitignored)
└── lora-persona-model-*/           # Trained models (gitignored)
```

## Development Workflow

### 1. Data Processing
```python
# In notebook, run the Slack Data Processing cell
process_slack_data_interactive()
```

### 2. Training
```python
# Use the processed dataset automatically
# or load custom data
```

### 3. Model Management
```python
# Built-in functions for model management
list_saved_models()
cleanup_old_models()
show_gpu_memory()
```

## Environment Updates

### Adding New Packages
```powershell
# Add to requirements.txt, then:
uv pip install package_name
# or
uv pip install -r requirements.txt
```

### Updating Packages
```powershell
# Update specific package
uv pip install --upgrade package_name

# Update all packages
uv pip install --upgrade -r requirements.txt
```

## Benefits Over Traditional venv

| Feature | venv + pip | uv |
|---------|------------|-----|
| Package installation | Slow | 10-100x faster |
| Dependency resolution | Basic | Advanced |
| Parallel downloads | No | Yes |
| Python version management | Manual | Built-in |
| Cache efficiency | Poor | Excellent |
| Cross-platform | Good | Excellent |

## Tips and Best Practices

1. **Always activate the environment** before working on the project
2. **Use `python setup_env.py check`** to verify your setup
3. **Update packages regularly** with `uv pip install --upgrade -r requirements.txt`
4. **The environment is gitignored** - safe to delete and recreate if needed
5. **Use the Jupyter kernel** "LoRA Persona (uv)" for consistent environment

## Troubleshooting

### Common Issues
1. **Environment not activated**: Run `.venv\Scripts\activate`
2. **Packages not found**: Run `uv pip install -r requirements.txt`
3. **GPU not detected**: Restart system, update drivers
4. **Kernel not available**: Run `python -m ipykernel install --user --name=lora-persona --display-name="LoRA Persona (uv)"`

### Getting Help
```powershell
# Check environment status
python setup_env.py

# Reinstall everything
rm -rf .venv
python setup_env.py setup
```