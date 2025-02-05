from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("models/creative_writing_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("models/creative_writing_gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route("/generate_story", methods=["POST"])
def generate_story():
    data = request.json
    input_text = data.get("prompt", "")
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    generated_story = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"generated_story": generated_story})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
