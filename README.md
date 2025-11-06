# Explainable Text Classifier (DistilBERT)

## Project Goals
- Predict the sentiment whether it is positive or negative.
- Build a fine-tuned **DistilBERT** model for sentiment classification.
- Add **SHAP** explainability to better understand the model.
- Add **LIME** explainability to better understand the model.
- Deploy the model through a **Streamlit app**

---

## Dataset
- **Source**: [Stanford Sentiment Treebank](https://huggingface.co/datasets/stanfordnlp/sst2)
- **Classes**: Binary Classification, **NEGATIVE** or **POSITIVE**

---

## How to Use?
1. **Running Code Locally**
    You can look at the code to better understand what goes through the usual process of loading and training a model from **Hugging Face**, you can also look at the **explainability** notebook to get a better feel for **SHAP** and **LIME**.

    ### **WARNING**
    A lot of the local code will not work since it assumes that you have the model loaded, but because the model was too big, I could not upload it to GitHub, so I had to deploy it through **Hugging Face**. If you run the code in the following order however, it should work:

    #### Install Requirements

    ```bash
    pip install -r requirements.txt

    ```

    #### Download Dataset

    ```bash
    python data.py
    ```
    - Downloads the **sst2** dataset
    - Takes a random subset of the training data, only **5000** records
    - Saves training, validation, and testing dataset into the folder **data**

    #### Fine-tune the Model
    
    ```bash
    python tune_model.py
    ```
    - Loads a pretrained **distilbert-base-uncased** model
    - Fine tunes it on the downloaded dataset
    - Saves it into the folder **models** under the name **finetuned-distilbert**


    #### Evaluate the Model

    ```bash
    python evaluate_model.py
    ```
    - Loads the validation dataset from the **data** folder
    - Loads the finetuned model **finetuned-distilbert** from **models**
    - Evaluates the model and calculates metrics such as **accuracy** and **F1 score**
    - Displays the **confusion matrix**

    #### Fix the Labels of the Model

    ```bash
    python fix_labels.py
    ```
    - Loads the trained model **finetuned-distilbert**
    - Adds the **NEGATIVE** and **POSITIVE** labels to the config of the model
    - Saves the model with the labels into the folder **models** under the name **finetuned-distilbert-labeled**
    

2. **Through the Streamlit App**

    #### Locally
    ```bash
    streamlit run app.py
    ```
    - Loads the model from **Hugging Face**
    - Opens a local streamlit webpage to test the model
    - You can write the sentence you want the model to analyze
    - Shows the **SHAP** and **LIME** explainability


    #### Online
    You can try the app [here](https://explainabletextclassifier-6jskhxsf2rxm5kjjbcai8t.streamlit.app/)