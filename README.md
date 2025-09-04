# LoRA Persona Training Project

## 🎯 Project Overview

This project creates a personalized AI model using LoRA (Low-Rank Adaptation) fine-tuning based on your Slack conversation data. The model learns your communication style and can generate responses that match your persona.

### Key Features
- **Slack Data Processing**: Converts Slack exports into training datasets
- **LoRA Fine-tuning**: Efficient model training with minimal resource requirements
- **Local Training**: Optimized for RTX A4000 GPU on workstation
- **Development Flexibility**: Works on laptop (CPU) and workstation (GPU)
- **Modern Environment**: Uses `uv` for fast package management

## 🏗️ Project Structure

```
Lora-persona/
├── 📋 Documentation & Setup
│   ├── README.md                       # This file - project overview
│   ├── ENVIRONMENT.md                  # Detailed environment setup guide
│   ├── requirements.txt                # Python dependencies
│   ├── setup_env.py                    # Environment management script
│   └── start.bat                       # Windows quick start script
│
├── 🔧 Core Scripts
│   ├── slack_data_processor.py         # Slack export → training data converter
│   └── LoRA-slack-persona-local.ipynb  # Main training notebook (RTX A4000 optimized)
│
├── 📁 Data Directories (gitignored)
│   ├── sampledata/                     # Your Slack export files
│   │   └── slack/                      # Place Slack exports here
│   ├── processed_datasets/             # Generated training datasets
│   ├── training_data/                  # Processed datasets cache
│   ├── lora-output-*/                  # Training checkpoints
│   └── lora-persona-model-*/           # Final trained models
│
├── 🐍 Python Environment (gitignored)
│   └── .venv/                          # Virtual environment (uv managed)
│
└── 📊 Notebooks (reference)
    └── colab/
        └── LoRA-slack-persona.ipynb    # Original Colab version
```

## 🔄 Development Workflow

### Development Environment (Laptop - CPU)
- **Purpose**: Data processing, code development, testing
- **Capabilities**: 
  - Process Slack data
  - Code development and debugging
  - Small-scale testing
  - Environment setup and package management

### Training Environment (Workstation - GPU)
- **Purpose**: Model training and inference
- **Requirements**: RTX A4000 or similar GPU
- **Capabilities**:
  - LoRA fine-tuning
  - Model inference and testing
  - Large-scale data processing

## 🚀 Quick Start Guide

### 1. Environment Setup (Both Laptop & Workstation)

# LoRA Persona Training Project

## 🎯 Project Overview

This project creates a personalized AI model using LoRA (Low-Rank Adaptation) fine-tuning based on your Slack conversation data. The model learns your communication style and can generate responses that match your persona.

### Key Features
- **Slack Data Processing**: Converts Slack exports into training datasets
- **LoRA Fine-tuning**: Efficient model training with minimal resource requirements
- **Multi-Platform Training**: Optimized for both Nvidia RTX GPUs and Apple Silicon
- **Development Flexibility**: Works on laptop (CPU) and workstation (GPU)
- **Modern Environment**: Uses `uv` for fast package management

## 🏗️ Project Structure

```
Lora-persona/
├── 📋 Documentation & Setup
│   ├── README.md                       # This file - project overview
│   ├── ENVIRONMENT.md                  # Detailed environment setup guide
│   ├── requirements.txt                # Python dependencies
│   ├── setup_env.py                    # Environment management script
│   └── start.bat                       # Windows quick start script
│
├── 🔧 Core Scripts
│   ├── slack_data_processor.py         # Slack export → training data converter
│   ├── LoRA-slack-persona-local.ipynb  # Main training notebook (RTX A4000 optimized)
│   └── LoRA-slack-persona-apple-silicon.ipynb # Main training notebook (Apple Silicon optimized)
│
├── 📁 Data Directories (gitignored)
│   ├── sampledata/                     # Your Slack export files
│   │   └── slack/                      # Place Slack exports here
│   ├── processed_datasets/             # Generated training datasets
│   ├── training_data/                  # Processed datasets cache
│   ├── lora-output-*/                  # Training checkpoints
│   └── lora-persona-model-*/           # Final trained models
│
├── 🐍 Python Environment (gitignored)
│   └── .venv/                          # Virtual environment (uv managed)
│
└── 📊 Notebooks (reference)
    └── colab/
        └── LoRA-slack-persona.ipynb    # Original Colab version
```

## 🔄 Development Workflow

### Development Environment (Laptop - CPU)
- **Purpose**: Data processing, code development, testing
- **Capabilities**: 
  - Process Slack data
  - Code development and debugging
  - Small-scale testing
  - Environment setup and package management

### Training Environment (Workstation - Nvidia GPU)
- **Purpose**: Model training and inference
- **Requirements**: RTX A4000 or similar GPU
- **Capabilities**:
  - LoRA fine-tuning
  - Model inference and testing
  - Large-scale data processing

### Training Environment (Apple Silicon - M1/M2/M3 GPU)
- **Purpose**: Model training and inference on Apple hardware
- **Requirements**: Apple Silicon Mac with macOS 12.3+
- **Capabilities**:
  - LoRA fine-tuning using Metal Performance Shaders (MPS)
  - Model inference and testing

## 🚀 Quick Start Guide

### 1. Environment Setup (All Platforms)

```powershell
# Clone the repository
git clone https://github.com/yuzuponikemi/Lora-persona.git
cd Lora-persona

# Quick start (Windows)
.\start.bat

# Manual setup (macOS/Linux)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Verify setup
python setup_env.py check
```

### 2. Data Preparation (Laptop or Workstation)

```powershell
# 1. Export your Slack data and place in sampledata/slack/
# 2. Start Jupyter
jupyter lab

# 3. Open the appropriate notebook for your hardware:
#    - LoRA-slack-persona-local.ipynb (for Nvidia GPUs)
#    - LoRA-slack-persona-apple-silicon.ipynb (for Apple Silicon)
# 4. Run the Slack Data Processing cell
# 5. Follow interactive prompts to select user and process data
```

### 3. Model Training (Workstation with GPU)

```powershell
# Ensure GPU environment is set up
python setup_env.py check  # Should show GPU detected

# Open the appropriate notebook and run training cells
# The notebook automatically detects and uses processed datasets
```

## 📊 Data Flow

```
Slack Export → Data Processor → Training Dataset → LoRA Training → Persona Model
     ↓              ↓                ↓               ↓             ↓
[Raw JSON]    [slack_data_     [instruction/   [LoRA fine-    [Your digital
              processor.py]    input/output    tuning]        twin model]
                               format]
```

## 🛠️ Environment Management

### Package Manager: `uv` (Modern Alternative to pip)
- **10-100x faster** than pip
- Better dependency resolution
- Built-in virtual environment management
- Cross-platform compatibility

### Key Commands
```powershell
# Environment status
python setup_env.py check

# Install new packages
uv pip install package_name

# Update all packages
uv pip install --upgrade -r requirements.txt

# Create new environment
uv venv --python 3.11
```

### Environment Detection
The project automatically adapts to your hardware:
- **CPU Mode**: Data processing, development, testing
- **GPU Mode (CUDA)**: For Nvidia GPUs
- **GPU Mode (MPS)**: For Apple Silicon

## 📝 Key Scripts and Their Purposes

### `slack_data_processor.py`
**Purpose**: Convert Slack exports to LLM training format
- Loads Slack metadata (users, channels)
- Analyzes user activity and statistics
- Creates conversation threads with context
- Exports in industry-standard formats (Alpaca, Chat, JSONL)

### `LoRA-slack-persona-local.ipynb`
**Purpose**: Main training notebook optimized for local Nvidia GPU
- Environment verification and setup
- Model loading with RTX A4000 optimizations (4-bit quantization)
- Data loading with automatic dataset detection
- LoRA fine-tuning configuration
- Model testing and inference

### `LoRA-slack-persona-apple-silicon.ipynb`
**Purpose**: Main training notebook optimized for Apple Silicon
- Environment verification for MPS
- Model loading with `float16` for compatibility
- Data loading with automatic dataset detection
- LoRA fine-tuning configuration for MPS
- Model testing and inference

### `setup_env.py`
**Purpose**: Environment management and verification
- Check `uv` installation
- Verify virtual environment status
- Test GPU availability (CUDA and MPS)
- Validate package installations
- Guided setup process

## 🎯 Training Configuration

### LoRA Parameters (Optimized for RTX A4000)
```python
LoraConfig(
    r=16,                    # Rank (balance between capability and memory)
    lora_alpha=32,          # Scaling factor
    target_modules=[...],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### Training Arguments (Memory Optimized for RTX A4000)
```python
TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,                         # Use bfloat16 for RTX A4000
    optim="paged_adamw_8bit"          # Memory-efficient optimizer
)
```

### LoRA Parameters (Optimized for Apple Silicon)
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[...],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### Training Arguments (Memory Optimized for Apple Silicon)
```python
TrainingArguments(
    per_device_train_batch_size=1,    # Smaller batch size for MPS
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,                        # Use float16 for MPS
    optim="adamw_torch"               # Standard optimizer
)
```

## 📊 Dataset Format

### Industry-Standard Alpaca Format
```json
{
  "instruction": "以下の対話の文脈に続いて、あなたらしく返信を生成してください。",
  "input": "ユーザーA: 「プロジェクトの進捗はいかがですか？」",
  "output": "順調に進んでいます！来週までには主要機能の実装が完了予定です。"
}
```

## 🔍 Troubleshooting Guide

### Common Issues and Solutions

#### Environment Issues
```powershell
# Environment not activated (macOS/Linux)
source .venv/bin/activate

# Environment not activated (Windows)
.venv\Scripts\activate

# Packages missing
uv pip install -r requirements.txt
```

#### GPU Issues (Nvidia CUDA)
```powershell
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### GPU Issues (Apple Silicon MPS)
```powershell
# Verify PyTorch MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# If MPS is not available, ensure you are on macOS 12.3+ and have a compatible PyTorch version.
```

---

**Note**: This project is designed to work seamlessly across development (laptop) and training (workstation) environments, with automatic adaptation to available hardware capabilities.


### 2. Data Preparation (Laptop or Workstation)

```powershell
# 1. Export your Slack data and place in sampledata/slack/
# 2. Start Jupyter
jupyter lab

# 3. Open LoRA-slack-persona-local.ipynb
# 4. Run the Slack Data Processing cell
# 5. Follow interactive prompts to select user and process data
```

### 3. Model Training (Workstation with GPU)

```powershell
# Ensure GPU environment is set up
python setup_env.py check  # Should show GPU detected

# Open the notebook and run training cells
# The notebook automatically detects and uses processed datasets
```

## 📊 Data Flow

```
Slack Export → Data Processor → Training Dataset → LoRA Training → Persona Model
     ↓              ↓                ↓               ↓             ↓
[Raw JSON]    [slack_data_     [instruction/   [LoRA fine-    [Your digital
              processor.py]    input/output    tuning]        twin model]
                               format]
```

## 🛠️ Environment Management

### Package Manager: `uv` (Modern Alternative to pip)
- **10-100x faster** than pip
- Better dependency resolution
- Built-in virtual environment management
- Cross-platform compatibility

### Key Commands
```powershell
# Environment status
python setup_env.py check

# Install new packages
uv pip install package_name

# Update all packages
uv pip install --upgrade -r requirements.txt

# Create new environment
uv venv --python 3.11
```

### Environment Detection
The project automatically adapts to your hardware:
- **CPU Mode**: Data processing, development, testing
- **GPU Mode**: Model training, inference optimization

## 📝 Key Scripts and Their Purposes

### `slack_data_processor.py`
**Purpose**: Convert Slack exports to LLM training format
- Loads Slack metadata (users, channels)
- Analyzes user activity and statistics
- Creates conversation threads with context
- Exports in industry-standard formats (Alpaca, Chat, JSONL)

**Usage**:
```python
# Interactive mode
process_slack_data_interactive()

# Custom path
process_custom_slack_path("path/to/slack/export")
```

### `LoRA-slack-persona-local.ipynb`
**Purpose**: Main training notebook optimized for local GPU
- Environment verification and setup
- Model loading with RTX A4000 optimizations
- Data loading with automatic dataset detection
- LoRA fine-tuning configuration
- Model testing and inference
- Utility functions for model management

### `setup_env.py`
**Purpose**: Environment management and verification
- Check `uv` installation
- Verify virtual environment status
- Test GPU availability
- Validate package installations
- Guided setup process

## 🎯 Training Configuration

### LoRA Parameters (Optimized for RTX A4000)
```python
LoraConfig(
    r=16,                    # Rank (balance between capability and memory)
    lora_alpha=32,          # Scaling factor
    target_modules=[        # Attention layers to target
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,      # Regularization
    task_type="CAUSAL_LM"   # Language modeling task
)
```

### Training Arguments (Memory Optimized)
```python
TrainingArguments(
    per_device_train_batch_size=4,    # RTX A4000 optimized
    gradient_accumulation_steps=2,     # Effective batch size = 8
    num_train_epochs=3,                # Start with 3 epochs
    learning_rate=2e-4,                # LoRA learning rate
    bf16=True,                         # Use bfloat16 for RTX A4000
    optim="paged_adamw_8bit"          # Memory-efficient optimizer
)
```

## 📊 Dataset Format

### Industry-Standard Alpaca Format
```json
{
  "instruction": "以下の対話の文脈に続いて、あなたらしく返信を生成してください。",
  "input": "ユーザーA: 「プロジェクトの進捗はいかがですか？」",
  "output": "順調に進んでいます！来週までには主要機能の実装が完了予定です。"
}
```

### Quality Metrics
- **Context Length**: 5 previous messages for context
- **Response Quality**: Minimum 10 characters, meaningful content
- **Thread Continuity**: Maintains conversation flow
- **User Focus**: Filters for target user's responses only

## 🔧 Development Best Practices

### Code Organization
- **Modular Design**: Separate data processing, training, and utilities
- **Error Handling**: Comprehensive error checking and user guidance
- **Memory Management**: GPU cache management and cleanup utilities
- **Logging**: Detailed progress tracking and debugging info

### Data Privacy
- **Gitignore Protection**: All personal data excluded from version control
- **Local Processing**: Data never leaves your local environment
- **Configurable Paths**: Easy to specify custom data locations

### Performance Optimization
- **Adaptive Batch Sizing**: Automatically adjusts to available memory
- **Efficient Loading**: Lazy loading and caching strategies
- **Progress Monitoring**: Real-time training metrics and GPU usage

## 🔍 Troubleshooting Guide

### Common Issues and Solutions

#### Environment Issues
```powershell
# Environment not activated
.venv\Scripts\activate

# Packages missing
uv pip install -r requirements.txt

# Wrong Python version
uv venv --python 3.11
```

#### GPU Issues
```powershell
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall CUDA PyTorch
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Data Processing Issues
- **No Slack data**: Place exports in `sampledata/slack/`
- **Processing fails**: Check JSON format and file permissions
- **Insufficient data**: Minimum 50+ examples recommended

### Getting Help
1. **Check environment**: `python setup_env.py check`
2. **Review logs**: Training outputs saved in `lora-output-*/logs/`
3. **Clean restart**: Delete `.venv` and run `python setup_env.py setup`

## 🎯 Project Goals

### Primary Objectives
1. **Learn Your Communication Style**: Capture tone, vocabulary, and response patterns
2. **Generate Authentic Responses**: Create replies that sound like you
3. **Maintain Context Awareness**: Understand conversation flow and context
4. **Efficient Training**: Minimize computational requirements while maximizing quality

### Success Metrics
- **Response Quality**: Coherent, contextually appropriate replies
- **Style Consistency**: Matches your communication patterns
- **Training Efficiency**: Completes training within reasonable time/memory constraints
- **Inference Speed**: Fast response generation for interactive use

## 🔄 Version Control Strategy

### Tracked Files
- Source code (`.py`, `.ipynb`)
- Documentation (`.md`)
- Configuration (`requirements.txt`, `setup_env.py`)
- Environment setup scripts

### Ignored Files
- Personal data (`sampledata/`, `processed_datasets/`)
- Trained models (`lora-persona-model-*`)
- Training outputs (`lora-output-*`)
- Python environment (`.venv/`)
- Temporary files and caches

This structure ensures you can safely share code while protecting personal data and avoiding repository bloat from large model files.

---

**Note**: This project is designed to work seamlessly across development (laptop) and training (workstation) environments, with automatic adaptation to available hardware capabilities.