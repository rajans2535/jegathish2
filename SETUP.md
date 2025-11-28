# Setup and Installation Guide

## Environment Requirements

### System Requirements
- **OS:** Windows, macOS, or Linux
- **Python:** 3.8, 3.9, 3.10, or 3.11
- **GPU:** Optional (NVIDIA GPU with CUDA 11.x for faster training)
- **Memory:** 8GB RAM minimum (16GB recommended)
- **Disk:** 5GB for datasets and models

### Python Dependencies

```
tensorflow==2.13.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
yfinance==0.2.28
pyyaml==6.0
statsmodels==0.14.0
jupyter==1.0.0
```

## Installation Steps

### 1. Clone or Setup Project Structure

```bash
# Create project directory
mkdir Advanced_TimeSeries_Forecasting
cd Advanced_TimeSeries_Forecasting

# Create folder structure
mkdir -p 1_Data/{data_splits}
mkdir -p 2_Code/utils
mkdir -p 3_Models
mkdir -p 4_Results/visualizations
mkdir -p 5_Report
mkdir -p 6_Documentation
mkdir -p 7_Config
```

### 2. Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r 6_Documentation/PROJECT_REQUIREMENTS.txt

# Or install manually
pip install tensorflow==2.13.0 numpy==1.24.3 pandas==2.0.3 \
    scikit-learn==1.3.0 matplotlib==3.7.2 seaborn==0.12.2 \
    yfinance==0.2.28 pyyaml==6.0 statsmodels==0.14.0 jupyter==1.0.0
```

### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"  # If needed
python -c "import pandas as pd; print(f'Pandas {pd.__version__}')"
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU Setup

1. **Install CUDA Toolkit:**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Select your OS and architecture
   - Follow installation instructions

2. **Install cuDNN:**
   - Download from: https://developer.nvidia.com/cudnn
   - Extract and copy to CUDA installation directory

3. **Verify GPU Detection:**
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### Alternative: Use Google Colab (Free GPU)

```python
# In Google Colab:
!pip install -r requirements.txt

# Upload project files or clone from GitHub
!git clone <repository-url>
```

## Configuration

### 1. Create config.yaml

Copy the provided `7_Config/config.yaml` and modify as needed:

```yaml
data:
  source: 'stock'
  ticker: 'AAPL'
  periods: 1000
  lookback: 30
  forecast_horizon: 7

model:
  lstm_units: 128
  dropout_rate: 0.2
  attention_units: 128

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

### 2. Set Environment Variables (Optional)

```bash
# Windows
set TF_CPP_MIN_LOG_LEVEL=2
set CUDA_VISIBLE_DEVICES=0

# macOS/Linux
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
```

## Testing Installation

```python
# test_installation.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import yaml

print("✓ NumPy:", np.__version__)
print("✓ Pandas:", pd.__version__)
print("✓ TensorFlow:", tf.__version__)
print("✓ Scikit-learn: OK")
print("✓ yfinance: OK")
print("✓ PyYAML: OK")

# Test GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"✓ GPUs Detected: {len(gpus)}")
if gpus:
    for gpu in gpus:
        print(f"  - {gpu}")

print("\n✓ Installation successful!")
```

```bash
python test_installation.py
```

## Troubleshooting

### Issue: TensorFlow not found

**Solution:**
```bash
pip install --upgrade tensorflow
```

### Issue: GPU not detected

**Solution:**
```bash
# Check CUDA/cuDNN installation
nvidia-smi

# Reinstall tensorflow with GPU support
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

### Issue: Memory error during training

**Solution:**
```python
# Reduce batch size in config.yaml
batch_size: 16  # Instead of 32

# Or reduce model size
lstm_units: 64  # Instead of 128
```

### Issue: Module not found error

**Solution:**
```bash
# Ensure virtual environment is activated
# Reinstall all requirements
pip install -r 6_Documentation/PROJECT_REQUIREMENTS.txt --force-reinstall
```

### Issue: yfinance data fetch fails

**Solution:**
```python
# Use simulated data instead in config.yaml
data:
  source: 'energy'  # Switch to simulated energy data

# Or update yfinance
pip install --upgrade yfinance
```

## Project Structure After Setup

```
Advanced_TimeSeries_Forecasting/
├── venv/                          # Virtual environment
├── 1_Data/                        # Data storage
├── 2_Code/                        # Source code
│   ├── *.py files
│   └── utils/
├── 3_Models/                      # Trained models
├── 4_Results/                     # Results and visualizations
├── 5_Report/                      # Reports
├── 6_Documentation/               # Documentation files
├── 7_Config/                      # Configuration files
│   └── config.yaml
├── PROJECT_REQUIREMENTS.txt       # Dependencies
└── project.log                    # Execution log
```

## First Run

```python
# 1. Test data loading
python -c "from 2_Code.data_loading import DataLoader; loader = DataLoader(); print('Data loader OK')"

# 2. Run full pipeline
python 2_Code/main_pipeline.py

# 3. Check results
ls 4_Results/
```

## Next Steps

1. Review `README.md` for project overview
2. Check `USAGE.md` for code examples
3. Read `PROJECT_REPORT.md` for technical details
4. Run `main_pipeline.py` to execute complete project

## Support & Help

For issues:
1. Check error messages carefully
2. Verify all dependencies installed: `pip list | grep tensorflow`
3. Check Python version: `python --version`
4. Review logs in `project.log`
5. Consult troubleshooting section above