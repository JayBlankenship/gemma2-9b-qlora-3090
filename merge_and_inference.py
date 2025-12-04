import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr
import subprocess
import shutil

# Configuration
BASE_MODEL_NAME = "google/gemma-2-9b"
LORA_WEIGHTS_PATH = "./gemma2-9b-qlora"
MERGED_MODEL_PATH = "./gemma2-9b-merged"
GGUF_PATH = "./gemma2-9b-merged.gguf"
LLAMA_CPP_PATH = "/path/to/llama.cpp"  # Update this path

def merge_lora_to_base():
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)

    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_MODEL_PATH}...")
    merged_model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    return merged_model, tokenizer

def convert_to_gguf():
    print("Converting to GGUF format...")
    # Assuming llama.cpp is installed and built
    # First, convert to GGML format, then to GGUF
    convert_cmd = [
        "python", f"{LLAMA_CPP_PATH}/convert.py",
        "--outfile", GGUF_PATH,
        "--outtype", "f16",  # or q4_0 for 4-bit
        MERGED_MODEL_PATH,
    ]
    subprocess.run(convert_cmd, check=True)
    print(f"GGUF model saved to {GGUF_PATH}")

def load_model_for_inference():
    # For demo, use the merged model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    return model, tokenizer

def generate_response(message, history, model, tokenizer, max_length=512, temperature=0.7):
    # Format the input
    prompt = f"Instruction: {message}\nResponse:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = response.split("Response:")[-1].strip()
    return response

def create_gradio_interface(model, tokenizer):
    def chat(message, history):
        response = generate_response(message, history, model, tokenizer)
        history = history + [[message, response]]
        return history, history

    with gr.Blocks() as demo:
        gr.Markdown("# Gemma-2-9B QLoRA Fine-tuned Chat")
        gr.Markdown("Chat with the fine-tuned Gemma-2-9B model.")
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Enter your message here...")
        clear = gr.Button("Clear")

        msg.submit(chat, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ([], []), None, chatbot, queue=False)

    return demo

def main():
    # Step 1: Merge LoRA to base model
    merged_model, tokenizer = merge_lora_to_base()

    # Step 2: Convert to GGUF (optional, requires llama.cpp)
    try:
        convert_to_gguf()
    except Exception as e:
        print(f"GGUF conversion failed: {e}. Skipping...")

    # Step 3: Load model for inference
    model, tokenizer = load_model_for_inference()

    # Step 4: Create and launch Gradio demo
    demo = create_gradio_interface(model, tokenizer)
    demo.launch(share=True)

if __name__ == "__main__":
    main()