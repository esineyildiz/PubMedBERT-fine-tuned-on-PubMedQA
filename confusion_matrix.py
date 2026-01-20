from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load saved model and data
model = AutoModelForSequenceClassification.from_pretrained("./vanilla-bert-pubmedqa-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./vanilla-bert-pubmedqa-finetuned")
dataset = load_from_disk("preprocessed_pubmedqa")

# Get predictions
trainer = Trainer(model=model)
predictions = trainer.predict(dataset["test"])
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# Confusion matrix
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["yes", "no", "maybe"])
disp.plot()
plt.savefig("confusion_matrix.png")
plt.show()