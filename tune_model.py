import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

with open("data/tokenized_dataset.pkl", "rb") as file:
    tokenized_dataset = pickle.load(file)
    
print(tokenized_dataset["train"][0])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", clean_up_tokenization_spaces = True)

model = AutoModelForSequenceClassification("distilbert-base-uncased", num_labels = 2)

training_arguments = TrainingArguments(
    output_dir = "./output",
    overwrite_output_dir = True,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 0.00003,
    per_device_eval_batch_size = 32,
    per_device_train_batch_size = 32,
    num_train_epochs = 4,
    logging_dir = "./log",
    logging_steps = 50,
    load_best_model_at_end = True
)

