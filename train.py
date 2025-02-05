#This script handles the training and fine-tuning of the GPT-2 model.

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader
import torch

# Load dataset and tokenizer
dataset = load_dataset("writing_prompts")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "story"])
train_loader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True)

# Load model and optimizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Save the model
model.save_pretrained("models/creative_writing_gpt2")
tokenizer.save_pretrained("models/creative_writing_gpt2")
