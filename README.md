# Fine-tuning PubMedBERT on PubMedQA

This project fine-tunes PubMedBERT for biomedical question answering using the PubMedQA dataset. A comparison with vanilla BERT-base is included to evaluate the effect of domain-specific pretraining.

## Dataset

PubMedQA is a biomedical question answering dataset where the task is to answer yes/no/maybe given a research context and question.

- Source: `qiaojin/PubMedQA` (pqa_labeled subset)
- Total examples: 1000
- Split: 700 train / 150 validation / 150 test
- Class distribution: yes (55%), no (34%), maybe (11%)

## Models

- **PubMedBERT**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- **BERT-base**: `bert-base-uncased`

## Training Configuration

- Learning rate: 2e-5
- Batch size: 8
- Epochs: 5
- Weight decay: 0.01
- Warmup ratio: 0.1
- Scheduler: linear
- Early stopping: patience 2 (based on F1 macro)

## Results

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| PubMedBERT | 64.7% | 0.44 | 0.60 |
| BERT-base | 52.7% | 0.23 | 0.36 |

PubMedBERT outperformed BERT-base by 12 percentage points in accuracy and nearly doubled the F1 macro score.

## Observations

- Both models failed to predict the minority class ("maybe"), likely due to class imbalance.
- An experiment with class weighting and lower learning rate did not improve results.
- Domain-specific pretraining provides clear benefits for biomedical text classification. 

## Confusion Matrices 
### PubMedBert 

![PubMedBERT Confusion Matrix](pubmedbert.png) 
### BERT-base 

![BERT-base Confusion Matrix](vanilla_bert.png)

Neither model predicted "maybe" for any example, illustrating the effect of class imbalance. 

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Preprocess the data:

```bash
python preprocess_data.py
```

Train PubMedBERT:

```bash
python train.py
```

Train BERT-base:

```bash
python train_bert.py
```

## Files

- `preprocess_data.py`: Downloads and preprocesses the dataset
- `train.py`: Fine-tunes PubMedBERT
- `train_bert.py`: Fine-tunes BERT-base
- `requirements.txt`: Dependencies
- `confusion_matrix.py`: Visualize the confusion matrices 


## References

- PubMedQA: Jin et al., "PubMedQA: A Dataset for Biomedical Research Question Answering" (2019)
- PubMedBERT: Gu et al., "Domain-Specific Pretraining for Biomedical Natural Language Processing" (2021)
- Amit, "Fine-Tuning BERT for Classification: A Practical Guide", Medium. https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
- Pai, S. "Designing Large Language Model Applications: A Holistic Approach to LLMs". O'Reilly Media.
