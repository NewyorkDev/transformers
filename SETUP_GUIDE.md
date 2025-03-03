# Mistral Fine-tuning Setup Guide

This guide will help you set up and run the Mistral fine-tuning project from the transformers repository.

## Prerequisites

- Python 3.8 or higher
- Git
- CUDA-compatible GPU (recommended for faster training)
- pip package manager

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/NewyorkDev/transformers.git
cd transformers
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install the package in development mode
pip install -e ".[quality]"

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install transformers datasets accelerate bitsandbytes wandb python-dotenv
```

### 4. Environment Setup

1. Create a `.env` file in the project root
2. Add your Hugging Face token:
```
HUGGING_FACE_TOKEN=your_token_here
```

You can obtain a token from [Hugging Face](https://huggingface.co/settings/tokens).

### 5. Available Scripts

The project contains several important scripts:

1. **`mistral_finetune.py`** - For fine-tuning Mistral models on your custom data
2. **`lambda_setup.sh`** - Setup script for Lambda Labs environment (for cloud-based training)
3. **`mistral_text_generation.py`** - For text generation with Mistral models

### 6. Preparing Training Data

Prepare your training data in one of these formats:
- Text files (.txt) with paragraphs separated by blank lines
- CSV files with a 'text' column containing training examples

Place your data in the `learning/` directory. Sample data is already provided in this directory.

pip install -r requirements.txt


### 7. Running Fine-tuning

```bash
python mistral_finetune.py
```

The script will:
- Load the Mistral model
- Ask for your dataset path (press Enter to use sample data)
- Configure training parameters (epochs, batch size)
- Start the fine-tuning process
- Save the model and generate test outputs

### 8. For Cloud-based Training (Lambda Labs)

If you have access to Lambda Labs or similar cloud GPU services:

```bash
# Make the script executable (on Linux/Mac)
chmod +x lambda_setup.sh

# Run the setup script
./lambda_setup.sh
```

This will set up the environment and provide instructions for transferring your data and running the training.

## Troubleshooting

- **CUDA errors**: Ensure you have the correct PyTorch version installed for your GPU
- **Memory issues**: Try reducing the batch size in the training parameters
- **Authentication errors**: Make sure your Hugging Face token has the necessary permissions
- **Import errors**: Verify all dependencies are installed correctly

## Additional Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Mistral AI Models](https://huggingface.co/mistralai)

## Support

For issues and questions, please open an issue in the GitHub repository.