from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
import pandas as pd

# Load a dataset from Local
train = pd.read_csv('data/persian_news/train.csv', delimiter='\t', encoding='utf-8-sig')
test = pd.read_csv('../data/persian_news/test.csv', delimiter='\t', encoding='utf-8-sig')
eval = pd.read_csv('../data/persian_news/dev.csv', delimiter='\t', encoding='utf-8-sig')

train.rename(columns={"content":"text"}, inplace=True)
eval.rename(columns={"content":"text"}, inplace=True)
test.rename(columns={"content":"text"}, inplace=True)

train_data = Dataset.from_pandas(train[["label", "text"]])
eval_data = Dataset.from_pandas(eval[["label", "text"]])
test_data = Dataset.from_pandas(test[["label", "text"]])

# Simulate the few-shot regime by sampling 8 examples per class
train_dataset = sample_dataset(train_data, label_column="label", num_samples=8)
eval_dataset = eval_data.select(range(100))
test_dataset = test_data.select(range(100, len(test_data)))

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained(
    "HooshvareLab/bert-fa-base-uncased-clf-digimag"
    #labels=["negative", "positive"],  # Adjust labels based on your dataset
)

args = TrainingArguments(
    batch_size=4,
    num_epochs=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric="accuracy",
    column_mapping={"text": "text", "label": "label"}  # Adjust column names based on your dataset
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate(test_dataset)
print(metrics)