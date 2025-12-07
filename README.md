# Gemma-2-9B QLoRA fine-tune on single RTX 3090

Fine-tune Google's Gemma-2-9B model using QLoRA on Alpaca dataset subset for efficient instruction tuning.

## Hardware & Runtime
- **GPU**: Single RTX 3090 (24GB VRAM)
- **Expected runtime**: ~3 hours
- **Dataset**: ~5k Alpaca examples

## Quick Start
```bash
# Request access to Gemma-2-9B at https://huggingface.co/google/gemma-2-9b (agree to terms)

# Create .env file with your Hugging Face token
echo "HF_TOKEN=hf_..." > .env

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py
```

## Target Performance
- **Final eval loss**: <1.6

## Expected VRAM Usage
- Peak VRAM on RTX 3090: ~22–23 GB

## Loss Curve
![Training Loss Curve](loss_curve_placeholder.png)

## Inference Options

After training, choose from these working inference interfaces:

### CLI Chat (Recommended)
Simple command-line interface for chatting:
```bash
python cli_inference.py
```
- Fast and reliable
- Type messages and get responses
- Type 'quit' or 'exit' to stop

### Streamlit Chat with History
Full chat interface with conversation history:
```bash
pip install streamlit  # if not installed
streamlit run streamlit_chat.py
```
- Web-based chat with persistent history
- Open the provided URL in your browser

### Streamlit Single-Turn
Simple web interface for single messages:
```bash
pip install streamlit  # if not installed
streamlit run streamlit_inference.py
```
- Web-based input/output
- No history, just quick responses