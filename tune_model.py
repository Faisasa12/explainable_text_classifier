import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

with open("data/tokenized_dataset.pkl", "rb") as file:
    tokenized_dataset = pickle.load(file)
    
print(tokenized_dataset["train"][0])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", clean_up_tokenization_spaces = True)

model = AutoModelForSequenceClassification("distilbert-base-uncased", num_labels = 2)
