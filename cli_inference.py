import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
import os

# Set CUDA device before any torch operations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load environment variables
load_dotenv()

# Configuration
BASE_MODEL_NAME = "google/gemma-2-9b"
LORA_WEIGHTS_PATH = "./gemma2-9b-qlora"

def load_and_merge_model():
    print("Loading base model in 4-bit...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading and merging LoRA weights...")
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
    # Keep the model with LoRA adapter for quantized inference

    print(f"Model device: {model.device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_response(message, model, tokenizer, max_length=512, temperature=0.7):
    prompt = f"Instruction: {message}\nResponse:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,  # Increased for longer responses
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Response:")[-1].strip()
    return response

def main():
    # Set Hugging Face token from .env
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    model, tokenizer = load_and_merge_model()
    
    print("CLI Chat Interface for Gemma-2-9B QLoRA")
    print("Type 'quit' or 'exit' to end the conversation.")
    
    while True:
        try:
            message = input("You: ").strip()
            if not message:
                continue
            if message.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            response = generate_response(message, model, tokenizer)
            print(f"Bot: {response}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()