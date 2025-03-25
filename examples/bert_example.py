from Advanced_AZ_NAS import auto_train_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

checkpoint = "prajjwal1/bert-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess_data(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = load_dataset("imdb")
dataset = dataset.map(preprocess_data, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_set = dataset["train"].shuffle(seed=42).select(range(5000))
test_set = dataset["test"].shuffle(seed=42).select(range(1000))

input_shape = (1, 128)

auto_train_model(
    model=model,
    train_dataset=train_set,
    test_dataset=test_set,
    input_shape=input_shape,
    num_trials=10,
    save_path="best_bert.pth",
    checkpoint=checkpoint,
    batch_size=64,
    tokenizer=tokenizer
    
)
