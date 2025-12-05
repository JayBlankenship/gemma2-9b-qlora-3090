import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
BASE_MODEL_NAME = "google/gemma-2-9b"
LORA_WEIGHTS_PATH = "./gemma2-9b-qlora"

def load_and_merge_model():
    print("Loading base model in 4-bit...")
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
    merged_model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return merged_model, tokenizer

def generate_response(message, model, tokenizer, max_length=512, temperature=0.7):
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
    response = response.split("Response:")[-1].strip()
    return response

def create_gradio_interface(model, tokenizer):
    def chat(message, history):
        response = generate_response(message, model, tokenizer)
        history.append([message, response])
        return "", history

    with gr.Blocks() as demo:
        gr.Markdown("# Gemma-2-9B QLoRA Chat")
        gr.Markdown("Simple chat interface for the fine-tuned model.")
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Enter your message...")
        clear = gr.Button("Clear")

        msg.submit(chat, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

    return demo

def main():
    # Set Hugging Face token from .env
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    model, tokenizer = load_and_merge_model()
    
    demo = create_gradio_interface(model, tokenizer)
    demo.launch(share=True)

if __name__ == "__main__":
    main()