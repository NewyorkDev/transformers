from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

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
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=token)
    
    return model, tokenizer

def generate_text(prompt, model, tokenizer, target_length=800):
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
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
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "writing-output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup model
    model, tokenizer = setup_mistral()
    
    # More natural writing prompts
    prompts = [
        "10 Critical Reasons Why Regular House Cleaning is Important:",
        "how to clean shower head:",
        "how to clean glass shower doors:"
    ]
    
    # Generate and save text for each prompt
    for i, prompt in enumerate(prompts, 1):
        # Generate text
        generated_text = generate_text(prompt, model, tokenizer)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"writing_{i}_{timestamp}.txt"
        file_path = os.path.join(output_dir, filename)
        
        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Topic: {prompt}\n\n{generated_text}")
        
        print(f"\nGenerated text saved to: {filename}")
        print("-" * 50)

if __name__ == "__main__":
    main()