import pickle

with open("data/tokenized_dataset.pkl", "rb") as file:
    tokenized_dataset = pickle.load(file)
    
print(tokenized_dataset["train"][0])