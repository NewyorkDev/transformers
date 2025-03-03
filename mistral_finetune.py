from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import os
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd
import numpy as np
from datetime import datetime

def check_gpu_or_raise():
    """
    Check if CUDA is available at all. If not, raise an error immediately.
    Print basic GPU info if available.
    """
    print("=== Checking for GPU availability ===")
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "CUDA is NOT available! This script requires a GPU with CUDA support."
        )
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs detected by PyTorch: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("======================================\n")

def confirm_model_on_gpu(model):
    """
    Confirm that at least some model parameters are on the GPU.
    If everything is on CPU, raise an error.
    """
    print("=== Verifying model device placement ===")
    unique_devices = {param.device for param in model.parameters()}
    print(f"Model parameters are on these devices: {unique_devices}")

    # Check if any parameter is actually on a CUDA device
    if not any(dev.type == 'cuda' for dev in unique_devices):
        raise EnvironmentError(
            "All model parameters appear to be on CPU! "
            "device_map='auto' may have failed or fallen back to CPU."
        )
    print("Model is confirmed to have parameters on GPU.\n")

def quick_forward_pass_check(model, tokenizer):
    """
    Run a small forward pass on the GPU to ensure no hidden errors.
    """
    print("=== Running a quick forward pass test ===")
    test_prompt = "GPU check"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        try:
            outputs = model(**inputs)
            print("Forward pass succeeded. Output keys:", outputs.keys())
        except Exception as e:
            raise RuntimeError(
                "Forward pass on the GPU failed! Check your CUDA/PyTorch setup."
            ) from e
    print("Forward pass test completed successfully.\n")

def setup_mistral():
    """
    Load environment variables for Hugging Face token,
    then load Mistral-7B-Instruct-v0.3 model and tokenizer with GPU checks.
    """
    # 1. Check GPU availability before loading anything
    check_gpu_or_raise()

    # 2. Load environment variables
    load_dotenv()
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    token = os.getenv('HUGGING_FACE_TOKEN')
    if not token:
        raise ValueError("Please set HUGGING_FACE_TOKEN in your .env file.")
    
    # 3. Load tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    
    # 4. Load model with fp16 precision, auto GPU mapping, and disabled cache for training
    print("Loading model with device_map='auto' and fp16 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        token=token,
        device_map="auto",
        use_cache=False
    )

    # 5. Confirm model is on GPU
    confirm_model_on_gpu(model)

    # 6. Quick forward pass test
    quick_forward_pass_check(model, tokenizer)

    # 7. Ensure we have a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to {tokenizer.eos_token}")

    return model, tokenizer

def load_human_text_dataset(file_path=None):
    """
    Load a dataset of human-written text for fine-tuning.

    Args:
        file_path (str): Path to a CSV or TXT file. If None or invalid,
                         loads .txt files from the 'learning/' folder.

    Returns:
        Dataset: A HuggingFace Dataset object containing text.
    """
    if file_path and os.path.exists(file_path):
        # If it's a CSV file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'text' not in df.columns:
                raise ValueError("CSV file must contain a 'text' column.")
            print(f"Loaded {len(df)} lines from CSV: {file_path}")
            return Dataset.from_pandas(df)
        
        # If it's a text file
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Split by double-newline or handle however you like
            texts = content.split('\n\n')
            print(f"Loaded {len(texts)} paragraphs from TXT: {file_path}")
            return Dataset.from_dict({'text': texts})
    
    # If no file or invalid file, load from "learning/" folder
    print("No valid file path provided. Loading all .txt files from 'learning/' folder...")
    learning_dir = os.path.join(os.path.dirname(__file__), "learning")
    texts = []
    
    if not os.path.exists(learning_dir):
        raise ValueError("'learning/' folder does not exist. Please create it and add .txt files.")

    for filename in os.listdir(learning_dir):
        if filename.endswith('.txt'):
            fpath = os.path.join(learning_dir, filename)
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            # Keep each entire file as one entry
            texts.append(content)
            print(f"Loaded file: {filename} with {len(content)} characters.")
    
    if not texts:
        raise ValueError("No text files found in 'learning/' folder. Please add some .txt files.")
    
    print(f"Total loaded text files: {len(texts)}")
    return Dataset.from_dict({'text': texts})

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize text examples for training.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

def train_model(model, tokenizer, dataset, output_dir="./finetuned-mistral", epochs=3, batch_size=4):
    """
    Fine-tune the model on the provided dataset.
    """
    print("=== Tokenizing dataset ===")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    print("Dataset tokenization complete.\n")
    
    # Create a data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We are not doing masked language modeling
    )
    
    # Configure training arguments
    print("=== Setting up TrainingArguments ===")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,  # Mixed precision
        gradient_accumulation_steps=4,
        logging_dir='./logs',
        logging_steps=1,               # Log as often as possible
        log_level="debug",             # Verbose logging
        learning_rate=5e-5,
        warmup_steps=100,
        report_to=["none"],            # Disable WandB or other reporters
        run_name="finetuning-run",     # Avoid run_name = output_dir warnings
    )
    print("TrainingArguments configured.\n")
    
    # Initialize the Trainer
    print("=== Initializing Trainer ===")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset
    )
    print("Trainer initialized.\n")
    
    # Train the model
    print("=== Starting training ===")
    trainer.train()
    print("=== Training complete ===\n")
    
    # Save the final model and tokenizer
    print(f"Saving model and tokenizer to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Save complete.\n")
    
    return model, tokenizer

def generate_text_from_finetuned(prompt, model, tokenizer):
    """
    Generate text using the fine-tuned model. No artificial truncation.
    """
    print("=== Generating text from the fine-tuned model ===")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs["input_ids"],
        max_length=2048,        # Large context window
        temperature=0.85,       # Controls randomness
        top_p=0.92,             # Nucleus sampling
        top_k=60,               # Another sampling parameter
        do_sample=True,
        repetition_penalty=1.2, # Penalize repeated phrases
        no_repeat_ngram_size=3, # Avoid repeating 3-grams
        pad_token_id=tokenizer.eos_token_id
    )
    
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== Generation complete ===\n")
    return text_output

def main():
    """
    Main entry point for fine-tuning and testing the Mistral model.
    """
    # 1. Setup model & tokenizer (includes GPU checks)
    model, tokenizer = setup_mistral()
    
    # 2. Ask user for dataset path (optional)
    print("\nFine-tuning Mistral with human-written text")
    print("-" * 50)
    dataset_path = input("Enter path to your text dataset (press Enter to use 'learning/' folder): ").strip()
    
    # 3. Load dataset
    if dataset_path and os.path.exists(dataset_path):
        dataset = load_human_text_dataset(dataset_path)
    else:
        dataset = load_human_text_dataset()  # loads from 'learning/' folder
    
    print(f"\nLoaded dataset with {len(dataset)} text entries.")
    
    # 4. Ask for training parameters
    epochs = int(input("Enter number of training epochs (default: 3): ") or 3)
    batch_size = int(input("Enter batch size (default: 4): ") or 4)
    
    # 5. Create a unique output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./finetuned-mistral-{timestamp}"
    
    # 6. Train the model
    print("\nStarting fine-tuning process...")
    model, tokenizer = train_model(
        model,
        tokenizer,
        dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size
    )
    
    print(f"Fine-tuning complete! Model saved to: {output_dir}")
    
    # 7. Test the fine-tuned model
    print("\nTesting the fine-tuned model:")
    test_prompt = input("Enter a prompt to test the model: ")
    generated_text = generate_text_from_finetuned(test_prompt, model, tokenizer)
    
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    # 8. Save the generated text
    test_output_dir = os.path.join(os.path.dirname(__file__), "finetuned-output")
    os.makedirs(test_output_dir, exist_ok=True)
    
    filename = f"finetuned_output_{timestamp}.txt"
    file_path = os.path.join(test_output_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {test_prompt}\n\n{generated_text}")
    
    print(f"Generated text saved to: {file_path}")

if __name__ == "__main__":
    main()
