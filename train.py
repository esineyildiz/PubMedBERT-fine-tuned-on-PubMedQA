from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Load preprocessed dataset
dataset = load_from_disk("preprocessed_pubmedqa")

# Load model and tokenizer
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Label mappings
label2id = {"yes": 0, "no": 1, "maybe": 2}
id2label = {0: "yes", 1: "no", 2: "maybe"}

# Load model for sequence classification (3 classes)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",

    # Training hyperparameters
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,  # Can be larger since no gradients stored
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",  # default 

    # Evaluation strategy
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    # Logging
    logging_dir="./logs",
    logging_steps=10,

    # Reproducibility
    seed=42,

    
)

# Early stopping callback - stops if f1_macro doesn't improve for 2 epochs
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,  # Number of evaluations with no improvement before stopping
    early_stopping_threshold=0.0,  # Minimum change to qualify as an improvement
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

# Train
print("Starting training...")
trainer.train()

# Evaluate on validation set
print("\nValidation results:")
val_results = trainer.evaluate()
print(val_results)

# Evaluate on test set
print("\nTest results:")
test_results = trainer.evaluate(dataset["test"])
print(test_results)

# Save the final model
trainer.save_model("./pubmedbert-pubmedqa-finetuned")
tokenizer.save_pretrained("./pubmedbert-pubmedqa-finetuned")
print("\nModel saved to './pubmedbert-pubmedqa-finetuned'")
