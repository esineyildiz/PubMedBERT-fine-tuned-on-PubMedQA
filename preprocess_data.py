from datasets import load_dataset
from transformers import AutoTokenizer

# Load PubMedQA labeled dataset
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

# Load PubMedBERT tokenizer
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Label mapping: convert string labels to integers
label2id = {"yes": 0, "no": 1, "maybe": 2}
id2label = {0: "yes", 1: "no", 2: "maybe"}


def preprocess_function(examples):
    """
    Preprocess PubMedQA examples for classification.

    Combines the question and context into a single input.
    Format: [CLS] question [SEP] context [SEP]
    """
    # Combine context sentences into a single string
    contexts = []
    for ctx in examples["context"]:
        # ctx["contexts"] is a list of sentences
        context_text = " ".join(ctx["contexts"])
        contexts.append(context_text)

    # Tokenize question + context pairs
    tokenized = tokenizer(
        examples["question"], # tokenizer automatically adds [CLS] and [SEP]
        contexts,
        truncation=True,
        padding="max_length", # simplest 
        max_length=512,
    )

    # Convert string labels to integers
    tokenized["labels"] = [label2id[label] for label in examples["final_decision"]]

    return tokenized


# Apply preprocessing to the dataset
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Split the train set into train/validation/test (since pqa_labeled only has train split)
# Using 70/15/15 split for proper evaluation
# First split: 70% train, 30% temp
train_temp = tokenized_dataset["train"].train_test_split(test_size=0.3, seed=42)
# Second split: split the 30% into 15% validation, 15% test
val_test = train_temp["test"].train_test_split(test_size=0.5, seed=42)

from datasets import DatasetDict
split_dataset = DatasetDict({
    "train": train_temp["train"],
    "validation": val_test["train"],
    "test": val_test["test"],
})

print("Preprocessed dataset:")
print(split_dataset)
print(f"\nTrain size: {len(split_dataset['train'])}")
print(f"Validation size: {len(split_dataset['validation'])}")
print(f"Test size: {len(split_dataset['test'])}")

# Save the preprocessed dataset for training
split_dataset.save_to_disk("preprocessed_pubmedqa")
print("\nDataset saved to 'preprocessed_pubmedqa/'")

# Also save as JSON for manual inspection 
import os
import json

# Save human-readable version (before tokenization) for easier inspection
# We need to split the original dataset with the same indices
original_train_temp = dataset["train"].train_test_split(test_size=0.3, seed=42)
original_val_test = original_train_temp["test"].train_test_split(test_size=0.5, seed=42)

original_splits = {
    "train": original_train_temp["train"],
    "validation": original_val_test["train"],
    "test": original_val_test["test"],
}

os.makedirs("preprocessed_pubmedqa_readable", exist_ok=True)

for split_name, split_data in original_splits.items():
    readable_data = []
    for example in split_data:
        readable_data.append({
            "question": example["question"],
            "context": " ".join(example["context"]["contexts"]),
            "label": example["final_decision"],
        })

    with open(f"preprocessed_pubmedqa_readable/{split_name}.json", "w") as f:
        json.dump(readable_data, f, indent=2)

print("Human-readable JSON files saved to 'preprocessed_pubmedqa_readable/'")
