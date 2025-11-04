from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("sst2") 

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(record):
    return tokenizer(record["sentence"], truncation = True, padding = "max_length")

tokenized_dataset = dataset.map(tokenize, batched = True)

tokenized_dataset.set_format(type = "torch", columns = ["label", "input_ids", "attention_mask"])

print(tokenized_dataset["train"][0])