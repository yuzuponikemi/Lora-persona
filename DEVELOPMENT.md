# Development Workflow Guide

## 🖥️ Multi-Environment Development Strategy

This project is designed to work seamlessly across two environments:
- **Laptop (Development)**: Code development, data processing, testing
- **Workstation (Training)**: GPU-accelerated model training and inference

## 📋 Environment Comparison

| Feature | Laptop (CPU) | Workstation (GPU) |
|---------|--------------|-------------------|
| **Hardware** | CPU only | RTX A4000 (16GB VRAM) |
| **Purpose** | Development & Data Processing | Training & Inference |
| **Capabilities** | ✅ Code development<br>✅ Data processing<br>✅ Small model testing<br>❌ LoRA training | ✅ All laptop features<br>✅ LoRA fine-tuning<br>✅ Large model inference<br>✅ GPU-accelerated processing |
| **Limitations** | No GPU training | Higher power consumption |

## 🔄 Typical Development Workflow

### Phase 1: Development & Data Preparation (Laptop)

#### 1.1 Environment Setup
```powershell
# Clone and setup on laptop
git clone https://github.com/yuzuponikemi/Lora-persona.git
cd Lora-persona
.\start.bat  # or manual setup with uv

# Verify environment (will show no GPU - this is expected)
python setup_env.py check
```

#### 1.2 Code Development
- Edit scripts (`slack_data_processor.py`, notebook cells)
- Test data processing functions
- Debug and refine code logic
- Update documentation

#### 1.3 Data Processing
```python
# Process Slack data (CPU-intensive but doable)
process_slack_data_interactive()

# This creates:
# - processed_datasets/slack_persona_*.csv
# - processed_datasets/slack_persona_*.json
# - processed_datasets/slack_persona_*.jsonl
```

#### 1.4 Commit Changes
```powershell
git add .
git commit -m "Update data processing logic"
git push origin main
```

### Phase 2: Training & Inference (Workstation)

#### 2.1 Sync Changes
```powershell
# On workstation
git pull origin main

# Setup environment (if first time)
.\start.bat
python setup_env.py check  # Should show GPU detected
```

#### 2.2 Training Execution
```powershell
# Start Jupyter with GPU environment
jupyter lab

# Open LoRA-slack-persona-local.ipynb
# Select "LoRA Persona (uv)" kernel
# Run training cells
```

#### 2.3 Model Testing & Validation
```python
# Interactive testing
interactive_test()

# Model management
list_saved_models()
show_gpu_memory()
```

## 🔧 Environment Synchronization

### Keeping Environments in Sync

#### Dependencies
Both environments use the same `requirements.txt`:
```powershell
# Update packages on both machines
uv pip install --upgrade -r requirements.txt
```

#### Code Changes
```powershell
# Laptop: develop and test
git add . && git commit -m "Feature update" && git push

# Workstation: sync and train
git pull origin main
```

#### Data Transfer
Since data is gitignored, you may need to transfer datasets:

**Option 1: Manual Copy**
```powershell
# Copy processed datasets between machines
# From laptop to workstation:
copy processed_datasets/* \\workstation\shared\Lora-persona\processed_datasets\
```

**Option 2: Re-process on Workstation**
```python
# Process data again on workstation (faster with GPU)
process_slack_data_interactive()
```

## 🎯 Development Best Practices

### 1. Code Development Strategy

#### On Laptop (Development Environment)
- **Focus**: Logic development, data processing, testing
- **Tools**: VS Code, Jupyter (CPU mode), debugging tools
- **Testing**: Small datasets, sample processing

```python
# Test with small dataset
sample_data = my_slack_data[:10]  # Test with 10 examples
```

#### On Workstation (Training Environment)
- **Focus**: Model training, full-scale testing, inference
- **Tools**: Jupyter (GPU mode), tensorboard, monitoring tools
- **Training**: Full datasets, production training

### 2. Performance Optimization

#### Laptop Optimizations
```python
# Reduce memory usage for development
training_args = TrainingArguments(
    per_device_train_batch_size=1,    # Smaller batch for CPU
    gradient_accumulation_steps=1,     # Less accumulation
    num_train_epochs=1,                # Quick test runs
    eval_strategy="no",                # Disable evaluation
    save_steps=1000,                   # Less frequent saves
)
```

#### Workstation Optimizations
```python
# Full performance for training
training_args = TrainingArguments(
    per_device_train_batch_size=4,     # Full batch size
    gradient_accumulation_steps=2,      # Effective batch = 8
    num_train_epochs=3,                 # Full training
    eval_strategy="epoch",              # Full evaluation
    save_steps=50,                      # Frequent saves
)
```

### 3. Environment Detection

The project automatically detects the environment:

```python
def detect_environment():
    """Detect current environment capabilities"""
    has_gpu = torch.cuda.is_available()
    gpu_memory = 0
    
    if has_gpu:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 Training Environment: GPU detected ({gpu_memory:.1f} GB)")
        return "training"
    else:
        print("💻 Development Environment: CPU only")
        return "development"

# Auto-configure based on environment
env_type = detect_environment()
if env_type == "training":
    # GPU-optimized settings
    batch_size = 4
    use_cache = True
else:
    # CPU-friendly settings
    batch_size = 1
    use_cache = False
```

## 📊 Development Workflow Examples

### Example 1: Adding a New Feature

#### Step 1: Develop on Laptop
```python
# In slack_data_processor.py
def new_analysis_feature(data):
    """New feature for better data analysis"""
    # Develop and test logic here
    return processed_data

# Test with small dataset
test_data = load_sample_data()
result = new_analysis_feature(test_data)
```

#### Step 2: Test and Commit
```powershell
# Test the feature
python -c "from slack_data_processor import new_analysis_feature; print('Feature works!')"

# Commit changes
git add . && git commit -m "Add new analysis feature" && git push
```

#### Step 3: Deploy to Workstation
```powershell
# On workstation
git pull origin main

# Test with full GPU environment
python setup_env.py check
jupyter lab  # Use the new feature in training
```

### Example 2: Training Optimization

#### Step 1: Research on Laptop
```python
# Research optimal parameters (CPU testing)
test_configs = [
    {"r": 8, "alpha": 16},
    {"r": 16, "alpha": 32},
    {"r": 32, "alpha": 64}
]

# Test configuration logic (no actual training)
for config in test_configs:
    lora_config = LoraConfig(**config)
    print(f"Config {config}: trainable params = {count_parameters(lora_config)}")
```

#### Step 2: Full Training on Workstation
```python
# Apply best configuration for full training
best_config = {"r": 16, "alpha": 32}  # From laptop testing
lora_config = LoraConfig(**best_config)

# Run full training with GPU
trainer.train()
```

## 🔍 Troubleshooting Multi-Environment Issues

### Common Problems and Solutions

#### Problem: Environment Sync Issues
```powershell
# Solution: Reset environment
rm -rf .venv
python setup_env.py setup
```

#### Problem: Package Version Conflicts
```powershell
# Solution: Use exact versions
uv pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

#### Problem: Data Not Available on Workstation
```python
# Solution: Check data paths
processed_datasets_dir = Path("./processed_datasets")
if not processed_datasets_dir.exists():
    print("⚠️  No processed datasets found. Run data processing first.")
    process_slack_data_interactive()
```

#### Problem: GPU Not Detected on Workstation
```powershell
# Check NVIDIA drivers
nvidia-smi

# Reinstall CUDA PyTorch
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Restart system if needed
```

## 📝 Development Checklist

### Before Switching Environments

#### Laptop → Workstation
- [ ] Commit all code changes
- [ ] Push to repository
- [ ] Verify data processing output
- [ ] Document any manual steps needed

#### Workstation → Laptop
- [ ] Save any trained models (if needed for analysis)
- [ ] Commit any configuration changes
- [ ] Document training results
- [ ] Clean up large temporary files

### Environment Verification
- [ ] `python setup_env.py check` passes
- [ ] Correct packages installed (`uv pip list`)
- [ ] Git repository is clean (`git status`)
- [ ] Required data files exist

## 🎯 Development Tips

### 1. Use Environment Variables
```python
import os

# Detect environment automatically
IS_TRAINING_ENV = torch.cuda.is_available()
DATA_PATH = os.getenv("SLACK_DATA_PATH", "./sampledata/slack")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "./models")
```

### 2. Conditional Configuration
```python
# Configure based on environment
if IS_TRAINING_ENV:
    # GPU settings
    config = {
        "batch_size": 4,
        "num_epochs": 3,
        "save_steps": 50
    }
else:
    # CPU settings
    config = {
        "batch_size": 1,
        "num_epochs": 1,
        "save_steps": 1000
    }
```

### 3. Logging and Monitoring
```python
import logging

# Environment-aware logging
log_level = logging.DEBUG if not IS_TRAINING_ENV else logging.INFO
logging.basicConfig(level=log_level)

logger = logging.getLogger(__name__)
logger.info(f"Running in {'training' if IS_TRAINING_ENV else 'development'} environment")
```

This workflow ensures efficient development across both environments while maintaining consistency and avoiding common pitfalls.