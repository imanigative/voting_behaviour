from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score

# Load a dataset from Local
data_2017 = pd.read_csv('voters/filtered_GLES_vote.csv')
data_2017_loc = data_2017.loc[:, ["prompt_A", "vote"]]


# remove NA at vote column and print the dist of vote
data_2017_loc = data_2017_loc.dropna(subset=['vote'])
print(data_2017_loc['vote'].value_counts())


# stratify the data based on vote column so that each class has equal number of samples
data_2017_loc_train = data_2017_loc.groupby('vote').apply(lambda x: x.sample(n=10, random_state=42)).reset_index(drop=True)
data_2017_loc_val = data_2017_loc.groupby('vote').apply(lambda x: x.sample(n=5, random_state=76)).reset_index(drop=True)
data_2017_loc_test = data_2017_loc.groupby('vote').apply(lambda x: x.sample(n=100, random_state=98)).reset_index(drop=True)


# rename the columns
data_2017_loc_train.rename(columns={"prompt_A":"text", "vote":"label"}, inplace=True)
data_2017_loc_val.rename(columns={"prompt_A":"text", "vote":"label"}, inplace=True)


data_2017_loc.rename(columns={"prompt_A":"text", "vote":"label"}, inplace=True)


train_data = Dataset.from_pandas(data_2017_loc_train[["label", "text"]])
eval_data = Dataset.from_pandas(data_2017_loc_val[["label", "text"]])
test_data = Dataset.from_pandas(data_2017_loc[["label", "text"]])


# Step 1: Attempt to load the SentenceTransformer model
try:
    # Load base model using SentenceTransformers
    base_model = SentenceTransformer("T-Systems-onsite/german-roberta-sentence-transformer-v2")
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
    print("Falling back to manual tokenizer and model loading...")
    
    # Load the model and tokenizer manually if SentenceTransformer fails
    tokenizer = AutoTokenizer.from_pretrained("T-Systems-onsite/german-roberta-sentence-transformer-v2")
    transformer_model = AutoModel.from_pretrained("T-Systems-onsite/german-roberta-sentence-transformer-v2")
    
    # Use manual components to create SentenceTransformer
    from sentence_transformers.models import Transformer, Pooling
    transformer = Transformer(transformer_model)
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    base_model = SentenceTransformer(modules=[transformer, pooling])

# Step 2: Wrap the SentenceTransformer model into SetFit
try:
    model = SetFitModel(encoder=base_model)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error integrating with SetFit: {e}")

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained(
    #"T-Systems-onsite/german-roberta-sentence-transformer-v2"
    #"sentence-transformers/paraphrase-mpnet-base-v2"
    "google-bert/bert-base-german-cased"
    #labels=["negative", "positive"],  # Adjust labels based on your dataset
)

args = TrainingArguments(
    batch_size=16,
    num_epochs=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    metric="accuracy",
    column_mapping={"text": "text", "label": "label"}  # Adjust column names based on your dataset
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate(test_data)
print(metrics)

# predict on test data and print actual and predicted values
predictions = model.predict(test_data["text"])
print(predictions)

# Assuming the actual labels are in the "label" column
actual_labels = test_data["label"]

# Calculate accuracy score
accuracy = accuracy_score(actual_labels, predictions)

# Print actual and predicted values, along with accuracy
print(f"Accuracy: {accuracy:.2f}")

##>>> accuracy
#0.6014851485148515

# save the model in models folder with name model_bert_base_german_cased
model.save_pretrained("voters/models/bert_base_german_cased_setfit_60")
# {'accuracy': 0.4142857142857143}

model_saved = SetFitModel.from_pretrained(
    #"T-Systems-onsite/german-roberta-sentence-transformer-v2"
    #"sentence-transformers/paraphrase-mpnet-base-v2"
    "models/bert_base_german_cased_setfit_60"
    #labels=["negative", "positive"],  # Adjust labels based on your dataset
)

df_setfit = pd.read_csv('results/few_shot/final_df_fs.csv')

test_data_setfit = df_setfit.loc[:, ['prompt_A', 'vote']]

predictions = model_saved.predict(df_setfit["prompt_A"])

# Calculate accuracy score
accuracy = accuracy_score(df_setfit['vote'], predictions)
accuracy

classif_report = 