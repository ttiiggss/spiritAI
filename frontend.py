# Install required packages:
# pip install transformers torch accelerate bitsandbytes gradio

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr

def setup_model():
    # Initialize model and tokenizer
    print("Loading model and tokenizer (this may take a few minutes)...")
    model_name = "LaierTwoLabsInc/Satoshi-7B"

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

# Load model and tokenizer globally
model, tokenizer = setup_model()

def generate_response(prompt, history):
    # Format the prompt
    formatted_prompt = f"<human>: {prompt}\n\n<assistant>: Let me help you with that."
    
    # Generate response
    print("Generating response...")
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Process response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(formatted_prompt, "").strip()
    
    # Update history
    history = history + [(prompt, response)]
    return "", history

def clear_history():
    return None

# Create Gradio interface with chat and clear button
with gr.Blocks() as iface:
    gr.HTML("""
        <div style="text-align: center; max-width: 700px; margin: 0 auto;">
            <h1>Satoshi-7B Chat Interface</h1>
            <p>Ask questions about blockchain, cryptocurrency, and finance</p>
        </div>
        """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Chat History",
                height=400
            )
            with gr.Row():
                with gr.Column(scale=8):
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Enter your prompt here...",
                        lines=2,
                        show_label=False
                    )
                with gr.Column(scale=1):
                    submit = gr.Button("Submit")
                with gr.Column(scale=1):
                    clear = gr.Button("Clear")

    # Set up event handlers
    submit.click(
        generate_response,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        generate_response,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    clear.click(
        clear_history,
        outputs=[chatbot]
    )

# Launch the interface
if __name__ == "__main__":
    print("Starting Gradio interface...")
    iface.launch(
        server_name="0.0.0.0",  # Makes it accessible on LAN
        server_port=7860,       # You can change this port
        share=False            # Set to True if you want a public URL
    )
