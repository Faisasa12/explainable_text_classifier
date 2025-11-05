import pickle
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt



model = AutoModelForSequenceClassification.from_pretrained('./models/finetuned-distilbert')
tokenizer = AutoTokenizer.from_pretrained('./models/finetuned-distilbert', clean_up_tokenization_spaces=True)



with open("data/tokenized_dataset_val.pkl", "rb") as file:
    val_dataset = pickle.load(file)

val_loader = DataLoader(val_dataset, batch_size = 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



all_preds, all_labels = [], []


with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        


accuracy = accuracy_score(all_labels, all_preds)
f1_score = f1_score(all_labels, all_preds)


print(f"\n\nAccuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1_score:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["negative", "positive"]))


cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])

plt.figure(figsize=(5, 5))
disp.plot(ax = plt.gca())
plt.title("Confusion Matrix - SST-2 Val Set")
plt.show()
