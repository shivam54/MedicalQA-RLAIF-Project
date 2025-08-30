# Medical Q&A Fine-tuning Project

## Overview

This project implements a comprehensive pipeline for fine-tuning Large Language Models (LLMs) for medical question-answering tasks. The approach combines Supervised Fine-Tuning (SFT) with Reinforcement Learning from AI Feedback (RLAIF) using Direct Preference Optimization (DPO) to create highly accurate medical Q&A models.

## Project Structure

```
MedicalQA Project/
├── MedicalQA.ipynb          # Main implementation notebook
├── README.md               # This file
├── requirements.txt        # Python dependencies
```

## Methodology

The project follows a sophisticated 4-step approach:

### 1. Supervised Fine-Tuning (SFT)
- **Base Model**: Qwen2.5-1.5B-Instruct-GPTQ-Int4
- **Training Data**: Medical literature Q&As from UltraMedical Dataset
- **Output**: Fine-tuned SFT model optimized for medical domain

### 2. Candidate Response Generation
- The fine-tuned SFT model generates 2 candidate responses for each question
- Uses open-ended Q&A dataset from UltraMedical Preference Dataset
- Implements controlled generation with temperature and top-p sampling

### 3. LLM-based Evaluation
- **Evaluator Model**: Qwen2.5-3B-Instruct
- Scores and ranks candidate outputs based on:
  - Factual accuracy
  - Completeness
  - Key point coverage
- Outputs numerical scores (0-1 scale) for each candidate

### 4. Policy Optimization using DPO
- **Dataset Preparation**: Higher-scored outputs become "chosen" answers; lower-scored become "rejected"
- **Training**: DPO model trained on preference pairs
- **Final Output**: RLAIF (Reinforcement Learning from AI Feedback) model

## Key Features

- **Quantized Models**: Uses GPTQ quantization for efficient inference
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning approach
- **Automated Evaluation**: LLM-based scoring system
- **Scalable Pipeline**: Designed to handle large medical datasets
- **Reproducible Results**: Comprehensive evaluation metrics

## Requirements

### Python Dependencies
```
torch
datasets
transformers
peft
optimum
auto-gptq
tqdm
huggingface_hub
```

### Hardware Requirements
- CUDA-compatible GPU (recommended)
- Minimum 8GB GPU memory
- 16GB+ system RAM

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd MedicalQA-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face authentication:
```python
from huggingface_hub import login
login(token="your_hf_token")
```

## Usage

### Running the Notebook

1. Open `MedicalQA.ipynb` in Jupyter or Google Colab
2. Ensure you have access to the required models and datasets
3. Run cells sequentially to execute the full pipeline

### Key Functions

- `load_medical_dataset()`: Loads UltraMedical Preference dataset
- `generate_answers_finetuned()`: Generates candidate responses
- `rank_answers()`: Evaluates and scores responses
- `evaluate_finetuned_model()`: Main evaluation pipeline

## Dataset

The project uses the **UltraMedical-Preference** dataset from Hugging Face:
- **Source**: `TsinghuaC3I/UltraMedical-Preference`
- **Size**: 109,353 training examples
- **Format**: Question-answer pairs with preference annotations

## Models Used

### Base Models
- **Qwen2.5-1.5B-Instruct-GPTQ-Int4**: Main model for fine-tuning
- **Qwen2.5-3B-Instruct**: Evaluation model

### Fine-tuned Models
- **SFT Model**: Supervised fine-tuned version
- **DPO Model**: Direct Preference Optimization model
- **RLAIF Model**: Final reinforcement learning model

## Results

The pipeline evaluates model performance on:
- **Accuracy**: Factual correctness of responses
- **Completeness**: Coverage of required information
- **Preference Alignment**: Agreement with human preferences

## Technical Details

### Model Configuration
- **Quantization**: GPTQ Int4 for memory efficiency
- **Fine-tuning**: LoRA adapters for parameter efficiency
- **Generation**: Temperature=0.7, top_p=0.95
- **Evaluation**: Custom scoring prompt with 0-1 scale

### Performance Optimizations
- GPU acceleration with CUDA
- Batch processing for efficiency
- Memory-efficient quantization
- Parallel evaluation pipeline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## Acknowledgments

- Hugging Face for model hosting and datasets
- Alibaba Cloud for Qwen models
- UltraMedical dataset contributors


## Contact

For questions or issues, please open an issue on the GitHub repository.
