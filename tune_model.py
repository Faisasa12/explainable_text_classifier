import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import evaluate

with open("data/tokenized_subaset_train.pkl", "rb") as file:
    train_dataset = pickle.load(file)
    
with open("data/tokenized_dataset_val.pkl", "rb") as file:
    val_dataset = pickle.load(file)
    
with open("data/tokenized_dataset_test.pkl", "rb") as file:
    test_dataset = pickle.load(file)
    

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", clean_up_tokenization_spaces = True)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 2)

training_arguments = TrainingArguments(
    output_dir = "./output",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 0.00003,
    dataloader_num_workers = 0,
    num_train_epochs = 2,
    logging_dir = "./log",
    logging_steps = 50,
    load_best_model_at_end = True
)

metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels =  eval_pred
    
    preds = logits.argmax(-1)
    
    return metric.compute(predictions = preds, references = labels)

trainer = Trainer(
    model = model,
    args = training_arguments,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

trainer.train()

trainer.save_model("./models/finetuned-distilbert")