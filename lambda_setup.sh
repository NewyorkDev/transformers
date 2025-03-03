#!/bin/bash
# Mistral Fine-tuning Setup Script (User-level, no sudo required)
# This script sets up everything in your home directory and uses the model version 7B-Instruct-v0.3

# Exit on any error
set -e

# -----------------------------
# Configuration
# -----------------------------
EXP_NAME="mistral-finetune-$(date +%Y%m%d_%H%M%S)"
BASE_DIR="$HOME/mistral_finetune"
OUTPUT_DIR="${BASE_DIR}/model_output/${EXP_NAME}"
DATASET_DIR="${BASE_DIR}/learning_data"
VENV_DIR="${BASE_DIR}/venv"

# -----------------------------
# 1. Create Local Directories
# -----------------------------
echo "Creating local directories under ${BASE_DIR}..."
mkdir -p "${BASE_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATASET_DIR}"
cd "${BASE_DIR}"

# -----------------------------
# 2. Create & Activate Virtual Environment
# -----------------------------
echo "Creating and activating virtual environment in ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# -----------------------------
# 3. Upgrade pip and Install Python Dependencies
# -----------------------------
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
# Adjust the torch version if needed for your CUDA version.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate bitsandbytes wandb

# -----------------------------
# 4. (Optional) Clone the Transformers Repository
# -----------------------------
if [ ! -d "transformers" ]; then
    echo "Cloning the Hugging Face Transformers repository..."
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install -e .
    cd ..
fi

# -----------------------------
# 5. Create the Training Script (train.py)
# -----------------------------
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
    # Model configuration: Using the Mistral-7B-Instruct-v0.3 model
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    output_dir = os.getenv("OUTPUT_DIR", "./model_output")
    dataset_dir = os.getenv("DATASET_DIR", "./learning_data")

    # Load model and tokenizer with appropriate device mapping and data type
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare dataset: Assumes text files (*.txt) in the dataset directory
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
    
    # Save the fine-tuned model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
EOL

# -----------------------------
# 6. Wrap-Up Instructions
# -----------------------------
echo "-----------------------------------------------------------"
echo "Setup complete!"
echo ""
echo "Your directories are as follows:"
echo "  Base directory:   ${BASE_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Dataset directory: ${DATASET_DIR}"
echo ""
echo "Next steps:"
echo "1. Place your training data (plain text files, e.g. .txt) in:"
echo "   ${DATASET_DIR}"
echo "   For example, copy your text files using:"
echo "      cp /path/to/your/files/*.txt ${DATASET_DIR}/"
echo ""
echo "2. Activate your virtual environment (if not already activated):"
echo "      source ${VENV_DIR}/bin/activate"
echo ""
echo "3. Run the training script by executing:"
echo "      OUTPUT_DIR=${OUTPUT_DIR} DATASET_DIR=${DATASET_DIR} python train.py"
echo ""
echo "After training completes, your fine-tuned model will be available in the output directory:"
echo "      ${OUTPUT_DIR}"
echo "-----------------------------------------------------------"
