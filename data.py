from datasets import load_dataset
from transformers import AutoTokenizer
import pickle

dataset = load_dataset("sst2") 
print(dataset)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", clean_up_tokenization_spaces = True)

def tokenize(record):
    return tokenizer(record["sentence"], truncation = True, padding = "max_length")

tokenized_dataset = dataset.map(tokenize, batched = True)

tokenized_dataset.set_format(type = "torch", columns = ["label", "input_ids", "attention_mask"])

with open("data/tokenized_subaset_train.pkl", "wb") as file:
    pickle.dump(tokenized_dataset["train"].shuffle().select(range(5000)), file)
    
with open("data/tokenized_dataset_val.pkl", "wb") as file:
    pickle.dump(tokenized_dataset["validation"], file)
    
with open("data/tokenized_dataset_test.pkl", 'wb') as file:
    pickle.dump(tokenized_dataset["test"], file)