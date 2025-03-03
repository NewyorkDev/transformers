#!/bin/bash

# Lambda Labs H100 Setup Script for Mistral Fine-tuning
# Error handling
set -e

# Ensure /workspace exists and is writable
echo "Ensuring /workspace exists and has correct permissions..."
sudo mkdir -p /workspace
sudo chown -R $(whoami):$(whoami) /workspace

# Configuration
EXP_NAME="mistral-finetune-$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/workspace/model_output/${EXP_NAME}"
DATASET_DIR="/workspace/learning_data"

# System updates and basic dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git wget curl python3-pip

# Create workspace directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATASET_DIR}"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate bitsandbytes wandb

# Clone transformers repository (if needed)
if [ ! -d "transformers" ]; then
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install -e .
    cd ..
fi

# Setup data transfer instructions
echo "Please transfer your learning data to ${DATASET_DIR}"
echo "You can use: scp -r ./learning/ ubuntu@<lambda-ip>:${DATASET_DIR}"

# Training script creation
cat > train.py << 'EOL'
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch
import os

def main():
    # Model configuration
    model_name = "mistralai/Mistral-7B-v0.1"
    output_dir = os.getenv("OUTPUT_DIR", "./model_output")
    dataset_dir = os.getenv("DATASET_DIR", "./learning_data")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare dataset
    dataset = load_dataset("text", data_files=f"{dataset_dir}/*.txt")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Training configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,
        save_steps=100,
        logging_steps=10,
        save_total_limit=2,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
EOL

# Instructions for running
echo """
Setup complete! Follow these steps to run the training:

1. Transfer your data:
   scp -r ./learning/ ubuntu@<lambda-ip>:${DATASET_DIR}

2. Start training:
   python3 train.py

3. Monitor training:
   tail -f ${OUTPUT_DIR}/trainer_log.txt

4. After completion, download the model:
   scp -r ubuntu@<lambda-ip>:${OUTPUT_DIR} ./

Make sure to replace <lambda-ip> with your actual Lambda Labs instance IP.
"""
