from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import argparse

# Load fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("models/creative_writing_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("models/creative_writing_gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate story
def generate_story(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for story generation")
    args = parser.parse_args()

    story = generate_story(args.prompt)
    print("Generated Story:\n", story)
