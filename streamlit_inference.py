import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
import streamlit as st
import os

# Set CUDA device before any torch operations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load environment variables
load_dotenv()

# Configuration
BASE_MODEL_NAME = "google/gemma-2-9b"
LORA_WEIGHTS_PATH = "./gemma2-9b-qlora"

@st.cache_resource
def load_model():
    st.write("Loading base model in 4-bit...")
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

    st.write("Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_response(message, model, tokenizer, temperature=0.7):
    prompt = f"Instruction: {message}\nResponse:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
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
    st.title("Gemma-2-9B QLoRA Chat")
    st.write("Simple web interface for the fine-tuned model.")

    # Set Hugging Face token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    model, tokenizer = load_model()

    message = st.text_input("Enter your message:", key="input")
    if st.button("Send") and message:
        with st.spinner("Generating response..."):
            response = generate_response(message, model, tokenizer)
        st.text_area("Response:", value=response, height=200)

if __name__ == "__main__":
    main()