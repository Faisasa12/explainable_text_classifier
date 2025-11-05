from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

model_path = "./models/finetuned-distilbert"
config = AutoConfig.from_pretrained(model_path)

config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.save_pretrained('./models/finetuned-distilbert-labeled')
tokenizer.save_pretrained('./models/finetuned-distilbert-labeled')