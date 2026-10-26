import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,Trainer,
    pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,f1_score,
    confusion_matrix,
    classification_report
)
dataset=load_dataset("ag_news",split="train[:2000]")
print("dataset size:",len(dataset))
print("label distribution:")
labels=dataset["label"]
for label in sorted(set(labels)):
    print(label,":",labels.count(label))

split_data=dataset.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="label"
)
train_data=split_data["train"]
val_data=split_data["test"]
print("Training rows:",len(train_data))
print("validation rows: ",len(val_data))
model_name="distilbert/distilbert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_name)
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
train_data=train_data.map(tokenize,batched=True)
val_data=val_data.map(tokenize,batched=True)
train_data=train_data.rename_column("label","labels")
val_data=val_data.rename_column("label","labels")
train_data.set_format(
    "torch",
    columns=["input_ids","attention_mask","labels"]
)
val_data.set_format(
    "torch",
    columns=["input_ids","attention_mask","labels"]
)
model=AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4
)
def compute_metrics(eval_pred):
    logits,actual_labels=eval_pred
    predictions=np.argmax(logits,axis=1)
    accuracy=accuracy_score(actual_labels,predictions)
    f1=f1_score(
        actual_labels,
        predictions,
        average="weighted"
    )
    return{
        "accuracy":accuracy,
        "f1":f1
    }

training_args=TrainingArguments(
    output_dir="./fine_tune_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    report_to="none",
    seed=42
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)
start_time=time.time()
trainer.train()
training_time=time.time() - start_time
print("Training time:",round(training_time,2),"seconds")
print("\nMetrics for each epoch:")
for result in trainer.state.log_history:
    if "eval_accuracy" in result:
        print(result)
results=trainer.evaluate()
print("\nValidation Results:")
print(results)
pred_output=trainer.predict(val_data)
predictions=np.argmax(pred_output.predictions,axis=1)
actual_labels=pred_output.label_ids
cm=confusion_matrix(
    actual_labels,
    predictions,
    labels=[0,1,2,3]
)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(
    actual_labels,
    predictions,
    labels=[0,1,2,3],
    target_names=[
        "World","Sports","Business","Sci/Tech"
    ],
    zero_division=0
))
plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("DistilBERT Confusion Matrix")
plt.xlabel("prediction Label")
plt.ylabel("Actual Lable")
plt.xticks(range(4),["World","Sports","Business","Sci/Tech"],rotation=30)
plt.yticks(range(4),["World","Sports","Business","Sci/Tech"])

for i in range(4):
    for j in range(4):
        plt.text(j,i,cm[i,j],ha="center",va="center")

plt.tight_layout()
plt.show()
trainer.save_model("./fine_tune_model")
tokenizer.save_pretrained("./fine_tune_model")
print("Model and tokenizer saved successfully")
classifier=pipeline(
    "text-classification",
    model="./fine_tune_model",
    tokenizer="./fine_tune_model",
    device=0 if torch.cuda.is_available() else -1
)

test_sentence=[
    "The government announced a new policy today",
    "The team won the football championship",
    "the company reported higher profits this year",
    "Scientists developed a new computer technology",
    "the stock market fell after the announcement"
]

print("\nPredictions for new sentences:")
for sentence in test_sentence:
    result=classifier(sentence)[0]
    print("\nsentences:",sentence)
    print("Label",result["label"])
    print("Confident:",round(result["score"],4))
train_texts=split_data["train"]["text"]
val_texts=split_data["test"]["text"]
y_train=split_data["train"]["label"]
y_val=split_data["test"]["label"]
vectorizer=TfidfVectorizer(max_features=5000)
X_train=vectorizer.fit_transform(train_texts)
X_val=vectorizer.transform(val_texts)
baseline=LogisticRegression(max_iter=1000)
start_time=time.time()
baseline.fit(X_train,y_train)
baseline_time=time.time()-start_time
baseline_predictions=baseline.predict(X_val)
baseline_accuracy=accuracy_score(
    y_val,baseline_predictions
)
baseline_f1=f1_score(
    y_val,baseline_predictions,
    average="weighted"
)
print("\nTF-IDF Baseline Results:")
print("Accuracy:",round(baseline_accuracy,4))
print("F1 Score:",round(baseline_f1,4))
print("Training Time:",round(baseline_time,2),"seconds")
print("\nDistilBERT vs TF-IDF:")
print("DistilBERT accuracy:",round(results["eval_accuracy"],4))
print("TF-IDF accuracy:",round(baseline_accuracy,4))