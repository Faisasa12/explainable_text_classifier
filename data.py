from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("sst2") 

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

