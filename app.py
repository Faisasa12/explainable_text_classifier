import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
from lime.lime_text import LimeTextExplainer
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

model_hf_path = 'Faisasa12/personal-finetuned-distilbert'

tokenizer = AutoTokenizer.from_pretrained(model_hf_path)
model = AutoModelForSequenceClassification.from_pretrained(model_hf_path)

device = 0 if torch.cuda.is_available() else -1

classifier_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)


def predict_proba_lime(texts):
    outputs = classifier_pipe(texts)
    probs = []
    
    for out in outputs:
        if out["label"] == "POSITIVE":
            probs.append([1-out["score"], out["score"]])
        else:
            probs.append([out["score"], 1-out["score"]])
            
    return np.array(probs)

def predict_proba_shap(texts):
    outputs = classifier_pipe(texts)
    probs = []
    
    for out in outputs:
        if out["label"] == "POSITIVE":
            probs.append([out["score"]])
        else:
            probs.append([1-out["score"]])
            
    return np.array(probs)

shap_explainer = shap.Explainer(classifier_pipe)
lime_explainer = LimeTextExplainer(class_names = ['NEGATIVE', 'POSITIVE'])

st.title("DistilBERT Sentiment Classifier with Explainability")

sentence = st.text_area("Enter a sentence for sentiment analysis:")

if st.button('Analyze') and sentence.strip() != '':

    result = classifier_pipe(sentence)[0]
    
    st.subheader('Prediction')
    st.write(f'Label: **{result["label"]}**  |  Confidence: **{result["score"]:.2f}**')
    

    st.subheader("SHAP Explanation")
    
    shap_vals = shap_explainer([sentence])
    
    shap_html = f"<head>{shap.getjs()}</head><body>{shap.plots.text(shap_vals[0], display = False)}</body>"
    components.html(shap_html, height=300)
    
    
    
    st.subheader("LIME Explanation")
    lime_exp = lime_explainer.explain_instance(sentence, predict_proba_lime, num_features = 10)
    fig = lime_exp.as_pyplot_figure()
    
    st.pyplot(fig)
    
    st.write("Top influential words (word, weight):")
    st.write(lime_exp.as_list())
