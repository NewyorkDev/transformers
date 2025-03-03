from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import os
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd
import numpy as np
from datetime import datetime

def setup_mistral():
    """
    Load environment variables for Hugging Face token,
    then load Mistral-7B-Instruct-v0.3 model and tokenizer.
    """
    load_dotenv()
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    token = os.getenv('HUGGING_FACE_TOKEN')
    if not token:
        raise ValueError("Please set HUGGING_FACE_TOKEN in your .env file")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    
    # Load model with fp16 precision, auto GPU mapping, and disabled cache for training
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        token=token,
        device_map="auto",
        use_cache=False
    )
    
    # Ensure we have a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
            return Dataset.from_pandas(df)
        
        # If it's a text file
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = f.read().split('\n\n')  # Simple split by double-newline
            return Dataset.from_dict({'text': texts})
    
    # If no file or invalid file, load from "learning/" folder
    print("No valid file path provided. Loading all .txt files from 'learning/' folder...")
    learning_dir = os.path.join(os.path.dirname(__file__), "learning")
    texts = []
    
    if os.path.exists(learning_dir):
        for filename in os.listdir(learning_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(learning_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # You can customize how you parse or split content here:
                    # For simplicity, let's just keep the entire file as one entry.
                    texts.append(content)
    else:
        raise ValueError("'learning/' folder does not exist. Please create it and add .txt files.")

    if not texts:
        raise ValueError("No text files found in 'learning/' folder. Please add some .txt files.")
    
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
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Create a data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We are not doing masked language modeling
    )
    
    # Configure training arguments
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
        learning_rate=5e-5,
        warmup_steps=100,
        report_to=["none"],            # Disable WandB or any other reporters
        run_name="finetuning-run",     # Avoid run_name = output_dir warnings
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def generate_text_from_finetuned(prompt, model, tokenizer):
    """
    Generate text using the fine-tuned model. No artificial truncation.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text with parameters tuned for more human-like output
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
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    """
    Main entry point for fine-tuning and testing the Mistral model.
    """
    # 1. Setup model & tokenizer
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
    
    print(f"Loaded dataset with {len(dataset)} text entries.")
    
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
    
    print(f"\nFine-tuning complete! Model saved to: {output_dir}")
    
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
