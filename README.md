# GPT-2-like Model from Scratch

This project implements a simplified version of the GPT-2 model from scratch using PyTorch. It includes training the model on a small subset of the WikiText-2 dataset, fine-tuning it on the IMDB dataset for sentiment analysis, applying Reinforcement Learning with Human Feedback (RLHF), and generating text for evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Steps](#steps)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

The goal of this project is to build, train, and fine-tune a GPT-2-like model from scratch. The model is trained on a small subset of the WikiText-2 dataset for language modeling, fine-tuned on the IMDB dataset for sentiment analysis, and improved using Reinforcement Learning with Human Feedback (RLHF). Finally, the model is evaluated by generating text and computing performance metrics.

## Steps

The project is divided into five steps:

1. **Step 1**: Build a GPT-2-like model from scratch.
   - Implement token and positional embeddings, multi-head self-attention, feedforward layers, and layer normalization.

2. **Step 2**: Train the model on a small subset of WikiText-2.
   - Train the model to predict the next token in a sequence using the WikiText-2 dataset.

3. **Step 3**: Fine-tune the model on a small subset of IMDB.
   - Adapt the model for sentiment analysis by fine-tuning it on the IMDB dataset.

4. **Step 4**: Apply Reinforcement Learning with Human Feedback (RLHF).
   - Improve the model's text generation by aligning it with human preferences using RLHF.

5. **Step 5**: Generate text and evaluate the model's performance.
   - Generate text using the fine-tuned model and evaluate its performance on a test dataset.

## Requirements

To run this project, you need the following:

- Python 3.x
- `torch` (PyTorch)
- `transformers` (Hugging Face Transformers library)
- `datasets` (Hugging Face Datasets library)
- `numpy`
- `tqdm` (for progress bars)
- `stable-baselines3` (for RLHF)
- `gymnasium` (for RLHF environment)
- A CUDA-capable GPU (optional but recommended for faster training)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/gpt2-from-scratch.git
   cd gpt2-from-scratch
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install torch transformers datasets numpy tqdm stable-baselines3 gymnasium
   ```

## Usage

1. **Run the script:**
   ```bash
   python gpt2_from_scratch.py
   ```

2. **Steps performed by the script:**
   - Build a GPT-2-like model from scratch.
   - Train the model on a small subset of WikiText-2.
   - Fine-tune the model on a small subset of IMDB.
   - Apply RLHF to align the model with human preferences.
   - Generate text and evaluate the model's performance.

3. **Output:**
   - Training and fine-tuning loss/accuracy metrics.
   - Generated text samples.
   - Evaluation metrics (e.g., loss, accuracy).

## Results

After running the script, you will see output similar to the following:
```
Training Loss: 1.2345, Accuracy: 0.5678
Fine-tuning Loss: 0.3456, Accuracy: 0.7890
Generated Text: "I really enjoyed this movie because..."
Evaluation Metrics: Loss: 0.1234, Accuracy: 0.9012
```

This demonstrates the model's ability to generate coherent and relevant text and its performance on the test dataset.


## License

MIT License
