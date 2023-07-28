import torch
import numpy as np
import evaluate
from datasets import load_dataset

# install kobert_tokenizer 
# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
from kobert_tokenizer import KoBERTTokenizer

from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# nsmc is a film review dataset with sentiment labels.
# It is one of the most popular benchmarks for Korean LLMs.
dataset = load_dataset("nsmc")
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

def tokenize_function(examples):
    return tokenizer(examples["document"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

model = AutoModelForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels = 2)
model.to(device)

training_args = TrainingArguments(output_dir="test_trainer", 
                                  save_strategy="epoch",
                                  evaluation_strategy = "steps", 
                                  eval_steps = 1000)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  compute_metrics=compute_metrics,
                  )

trainer.train()