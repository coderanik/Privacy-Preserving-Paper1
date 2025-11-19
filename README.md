# Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models

This repository contains implementations of **Differential Privacy Federated Prompt Learning (DP-FPL)** for CLIP models. The codebase includes 5 different notebooks, each designed for different use cases and computational requirements.

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Notebooks Overview](#notebooks-overview)
- [How to Run](#how-to-run)
- [Detailed Differences](#detailed-differences)
- [Key Features](#key-features)

## 🎯 Overview

This project implements federated learning with differential privacy for prompt tuning of CLIP (Contrastive Language-Image Pre-training) models. The approach combines:
- **Federated Learning**: Multiple clients train locally without sharing raw data
- **Differential Privacy**: Adds calibrated noise to protect individual data privacy
- **Prompt Learning**: Efficient fine-tuning using learnable prompts instead of full model weights
- **Low-Rank Factorization**: Reduces communication costs and improves efficiency

## 📦 Requirements

### Core Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install numpy tqdm
```

### Additional Dependencies (for OpenAI CLIP notebooks)

```bash
pip install git+https://github.com/openai/CLIP.git
```

### System Requirements

- **CPU-only implementation** (all notebooks run on CPU)
- Python 3.7+
- ~2-4 GB RAM recommended
- Internet connection for downloading CLIP models and datasets

## 📚 Notebooks Overview

| Notebook | CLIP Source | DP Type | Dataset | Use Case |
|----------|------------|---------|---------|----------|
| `simple-cpu-code.ipynb` | Simulated | Simulated | Synthetic | Quick testing, learning |
| `DP-FPL_HF_CLIP_Simulated.ipynb` | HuggingFace | Simulated | Flowers102 | Fast experimentation |
| `DP-FPL_HF_CLIP_TrueDP.ipynb` | HuggingFace | True DP | Flowers102 | Research, publication |
| `DP-FPL_OpenAI_CLIP_Simulated.ipynb` | OpenAI | Simulated | Flowers102 | Fast experimentation |
| `DP-FPL_OpenAI_CLIP_TrueDP.ipynb` | OpenAI | True DP | Flowers102 | Research, publication |

## 🚀 How to Run

### 1. Simple CPU Code (Lightweight)

**File**: `simple-cpu-code.ipynb`

**Best for**: Understanding the algorithm, quick testing, educational purposes

**Steps**:
1. Open the notebook in Jupyter/Colab
2. Run all cells sequentially
3. No additional setup required - uses synthetic data

**Expected Runtime**: < 1 minute

**Output**: Training loss per round for 5 federated rounds

---

### 2. HuggingFace CLIP - Simulated DP

**File**: `DP-FPL_HF_CLIP_Simulated.ipynb`

**Best for**: Fast experimentation with real CLIP models and datasets

**Steps**:
1. Install dependencies (first cell installs packages)
2. Run cells sequentially
3. Dataset (Flowers102) downloads automatically on first run
4. Training runs with fixed noise levels (SIGMA_L=0.3, SIGMA_G=0.1)

**Expected Runtime**: 10-30 minutes (depending on CPU)

**Output**: 
- Training loss per round
- Test accuracy for each client

**Key Parameters**:
- `NUM_CLIENTS = 10`
- `ROUNDS = 5`
- `LOCAL_EPOCHS = 1`
- `BATCH_SIZE = 16`

---

### 3. HuggingFace CLIP - True DP

**File**: `DP-FPL_HF_CLIP_TrueDP.ipynb`

**Best for**: Research experiments requiring formal privacy guarantees

**Steps**:
1. Install dependencies (first cell)
2. Run cells sequentially
3. Dataset downloads automatically
4. Training runs for multiple epsilon values: `[0.4, 0.2, 0.1, 0.05, 0.01]`

**Expected Runtime**: 30-60 minutes (runs 5 different epsilon values)

**Output**:
- Training loss per round for each epsilon
- Test accuracy for each client
- Computed noise levels (sigma_L, sigma_G) based on privacy budget

**Key Features**:
- Proper Gaussian mechanism with epsilon-delta privacy
- Gradient clipping threshold: 10
- Delta (δ): 1e-5
- Noise computed as: `σ = sqrt(2*log(1.25/δ)) * sensitivity / ε`

---

### 4. OpenAI CLIP - Simulated DP

**File**: `DP-FPL_OpenAI_CLIP_Simulated.ipynb`

**Best for**: Fast experimentation with OpenAI's official CLIP implementation

**Steps**:
1. Install OpenAI CLIP (first cell): `pip install git+https://github.com/openai/CLIP.git`
2. Run cells sequentially
3. Dataset downloads automatically
4. Uses CLIP ViT-B/32 model

**Expected Runtime**: 10-30 minutes

**Output**:
- Training loss per round
- Personalization accuracy for each client

**Key Differences from HF version**:
- Uses OpenAI's official CLIP library
- Different preprocessing pipeline
- ViT-B/32 architecture (vs ViT-B/16 in HF)

---

### 5. OpenAI CLIP - True DP

**File**: `DP-FPL_OpenAI_CLIP_TrueDP.ipynb`

**Best for**: Research with OpenAI CLIP and formal privacy guarantees

**Status**: ⚠️ **Incomplete** - Currently contains only header comments

**Note**: This notebook appears to be a placeholder. For True DP with OpenAI CLIP, you may need to adapt the HuggingFace True DP implementation.

---

## 🔍 Detailed Differences

### CLIP Implementation Differences

#### HuggingFace CLIP (`transformers` library)
- **Model**: `CLIPModel.from_pretrained("openai/clip-vit-base-patch16")`
- **Processor**: `CLIPProcessor` for image/text preprocessing
- **Architecture**: ViT-B/16 (Vision Transformer Base, 16x16 patches)
- **Embedding Dimension**: 512
- **Advantages**: 
  - Easy integration with HuggingFace ecosystem
  - Standardized preprocessing
  - Well-documented API

#### OpenAI CLIP (GitHub repository)
- **Model**: `clip.load("ViT-B/32")`
- **Preprocessing**: Custom `preprocess` function
- **Architecture**: ViT-B/32 (Vision Transformer Base, 32x32 patches)
- **Embedding Dimension**: 512
- **Advantages**:
  - Official OpenAI implementation
  - Direct access to original model weights
  - Slightly different architecture (32x32 patches)

### Differential Privacy Differences

#### Simulated DP
- **Noise Levels**: Fixed values (e.g., `SIGMA_L = 0.3`, `SIGMA_G = 0.1`)
- **Privacy Guarantee**: ❌ No formal privacy guarantee
- **Use Case**: Fast prototyping, understanding algorithm behavior
- **Advantages**: Fast, simple, predictable
- **Disadvantages**: Not suitable for production or privacy-sensitive applications

#### True DP
- **Noise Calculation**: Computed from privacy budget (ε, δ)
- **Formula**: `σ = sqrt(2*log(1.25/δ)) * sensitivity / ε`
- **Privacy Guarantee**: ✅ Formal (ε, δ)-differential privacy
- **Use Case**: Research, production, privacy-sensitive applications
- **Advantages**: Provable privacy guarantees
- **Disadvantages**: Slower, requires careful parameter tuning

**Privacy Parameters in True DP**:
- **Epsilon (ε)**: Privacy budget (lower = more private) - tested: [0.4, 0.2, 0.1, 0.05, 0.01]
- **Delta (δ)**: Failure probability - set to 1e-5
- **Sensitivity**: Gradient clipping threshold / batch size
- **Noise**: Gaussian noise with computed standard deviation

### Dataset Differences

#### Synthetic Data (`simple-cpu-code.ipynb`)
- **Type**: Random tensors generated on-the-fly
- **Size**: Configurable per client (default: 1000 samples)
- **Classes**: PROMPT_LEN (16) classes
- **Use Case**: Algorithm testing, no real-world data needed

#### Flowers102 Dataset (Other notebooks)
- **Type**: Real image classification dataset
- **Size**: ~8,189 training images, 1,633 test images
- **Classes**: 102 flower categories
- **Distribution**: Non-IID (each client gets different classes)
- **Download**: Automatic on first run (saved to `./data/`)

### Architecture Differences

#### Simple Implementation
- **CLIP**: Simulated with linear layers
- **Embedding Dim**: 128 (reduced for speed)
- **Prompt Length**: 16
- **Low-Rank Rank**: 4
- **Clients**: 5
- **Rounds**: 5

#### Full Implementations
- **CLIP**: Real pre-trained models (frozen)
- **Embedding Dim**: 512 (CLIP standard)
- **Prompt Length**: 16
- **Low-Rank Rank**: 4-8 (varies by notebook)
- **Clients**: 10
- **Rounds**: 5

## ✨ Key Features

### Federated Learning
- **10 clients** (5 in simple version)
- **Non-IID data distribution** (each client has different classes)
- **Local training** with multiple epochs
- **Global aggregation** of prompts

### Differential Privacy
- **Local DP (LDP)**: Noise added to local prompt gradients
- **Global DP (GDP)**: Noise added to aggregated global gradients
- **Gradient clipping**: Prevents large gradients from breaking privacy
- **Low-rank factorization**: Reduces communication and improves privacy

### Prompt Learning
- **Global prompt**: Shared across all clients
- **Local prompt**: Personalized per client
- **Full prompt**: Global + Local (used for inference)
- **Low-rank decomposition**: `prompt = U @ V + residual`

### Evaluation
- **Training loss**: Monitored per round
- **Test accuracy**: Evaluated on Flowers102 test set
- **Personalization**: Each client tested on their local classes

## 📊 Expected Results

### Simple CPU Code
```
Round 1: mean loss = 3.4621
Round 2: mean loss = 3.4364
Round 3: mean loss = 3.4776
Round 4: mean loss = 3.2800
Round 5: mean loss = 3.3831
```

### Full Implementations
- **Training Loss**: Should decrease over rounds (typically 2.5-4.0 range)
- **Test Accuracy**: Varies by epsilon (privacy-accuracy tradeoff)
  - Higher epsilon (less private) → Higher accuracy
  - Lower epsilon (more private) → Lower accuracy

## 🔧 Customization

### Adjusting Privacy Budget (True DP)
```python
EPSILONS = [0.4, 0.2, 0.1, 0.05, 0.01]  # Modify this list
```

### Changing Number of Clients
```python
NUM_CLIENTS = 10  # Adjust based on your needs
```

### Modifying Noise Levels (Simulated DP)
```python
SIGMA_L = 0.3  # Local DP noise
SIGMA_G = 0.1  # Global DP noise
```

### Changing Model Architecture
- **HuggingFace**: Change model name in `CLIPModel.from_pretrained()`
- **OpenAI**: Change model name in `clip.load()` (e.g., "ViT-B/16", "ViT-L/14")

## 📝 Notes

1. **CPU-Only**: All implementations are CPU-only for accessibility
2. **First Run**: Dataset download may take a few minutes
3. **Memory**: Flowers102 dataset requires ~500MB disk space
4. **Reproducibility**: Set random seeds for reproducible results
5. **Privacy**: True DP implementations provide formal guarantees; simulated versions do not

## 🤝 Citation

If you use this code in your research, please cite the original paper on Privacy-Preserving Personalized Federated Prompt Learning.

## 📄 License

[Add your license information here]

---

**Last Updated**: [Current Date]
**Maintained by**: [Your Name/Organization]

