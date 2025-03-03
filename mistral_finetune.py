from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import os
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from datetime import datetime

def setup_mistral():
    # Load environment variables
    load_dotenv()
    
    # Initialize model and tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # Get token from environment variable
    token = os.getenv('HUGGING_FACE_TOKEN')
    if not token:
        raise ValueError("Please set HUGGING_FACE_TOKEN in your .env file")
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        token=token,
        device_map="auto",  # Automatically distribute across available GPUs
        use_cache=False  # Important for training
    )
    
    # Set padding token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_human_text_dataset(file_path=None):
    """
    Load a dataset of human-written text for fine-tuning.
    
    Args:
        file_path: Path to a text file or CSV containing human-written text
                  If None, will create a sample dataset from writing-output
    
    Returns:
        A HuggingFace dataset object
    """
    if file_path and os.path.exists(file_path):
        # If it's a CSV file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Assuming the CSV has a 'text' column
            if 'text' in df.columns:
                return Dataset.from_pandas(df)
            else:
                raise ValueError("CSV file must contain a 'text' column")
        
        # If it's a text file
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = f.read().split('\n\n')  # Split by paragraphs
            return Dataset.from_dict({'text': texts})
    
    # If no file provided or file doesn't exist, create sample from writing-output
    else:
        print("No valid file path provided. Creating sample dataset from writing-output directory...")
        output_dir = os.path.join(os.path.dirname(__file__), "writing-output")
        texts = []
        
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Remove the "Topic:" line and keep just the human-like text
                        if "\n\n" in content:
                            texts.append(content.split("\n\n", 1)[1])
        
        if not texts:
            raise ValueError("No text files found in writing-output directory. Please provide a file path.")
            
        return Dataset.from_dict({'text': texts})

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize the text examples for training.
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
    
    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not doing masked language modeling
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,  # Use mixed precision training
        gradient_accumulation_steps=4,  # Accumulate gradients over multiple steps
        logging_dir='./logs',
        logging_steps=100,
        learning_rate=5e-5,
        warmup_steps=100,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def generate_text_from_finetuned(prompt, model, tokenizer, target_length=800):
    """
    Generate text using the fine-tuned model.
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text with parameters tuned for more human-like output
    outputs = model.generate(
        inputs["input_ids"],
        max_length=2048,  # Increased to allow for longer generations
        temperature=0.85,  # Higher temperature for more randomness
        top_p=0.92,  # Slightly higher top_p for more diversity
        top_k=60,  # Add top_k sampling
        do_sample=True,
        repetition_penalty=1.2,  # Penalize repetition
        no_repeat_ngram_size=3,  # Avoid repeating 3-grams
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Adjust length to be close to target (800 characters)
    words = generated_text.split()
    adjusted_text = ""
    char_count = 0
    
    for word in words:
        if char_count + len(word) + 1 <= target_length + 100:  # Allow slight overflow
            adjusted_text += word + " "
            char_count += len(word) + 1
        else:
            break
    
    return adjusted_text.strip()

def main():
    # Setup model and tokenizer
    model, tokenizer = setup_mistral()
    
    # Ask user for dataset path
    print("\nFine-tuning Mistral with human-written text")
    print("-" * 50)
    dataset_path = input("Enter path to your human text dataset (or press Enter to use sample data): ").strip()
    
    # Load dataset
    if dataset_path and os.path.exists(dataset_path):
        dataset = load_human_text_dataset(dataset_path)
    else:
        dataset = load_human_text_dataset()
    
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # Ask for training parameters
    epochs = int(input("Enter number of training epochs (default: 3): ") or 3)
    batch_size = int(input("Enter batch size (default: 4): ") or 4)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./finetuned-mistral-{timestamp}"
    
    # Train the model
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
    
    # Test the fine-tuned model
    print("\nTesting the fine-tuned model:")
    test_prompt = input("Enter a prompt to test the model: ")
    generated_text = generate_text_from_finetuned(test_prompt, model, tokenizer)
    
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    # Save the generated text
    test_output_dir = os.path.join(os.path.dirname(__file__), "finetuned-output")
    os.makedirs(test_output_dir, exist_ok=True)
    
    filename = f"finetuned_output_{timestamp}.txt"
    file_path = os.path.join(test_output_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {test_prompt}\n\n{generated_text}")
    
    print(f"Generated text saved to: {file_path}")

if __name__ == "__main__":
    main()