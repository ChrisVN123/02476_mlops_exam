import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer

# Define project root and file path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../exam_project_mlops")
)
file_path = os.path.join(project_root, "data", "raw", "Sentences_AllAgree.txt")

# Load and preprocess data
with open(file_path, "r", encoding="latin1") as file:
    lines = file.readlines()
data = [line.rsplit("@", 1) for line in lines]
df = pd.DataFrame(data, columns=["Text", "Sentiment"])
df["Text"] = df["Text"].str.strip()
df["Sentiment"] = df["Sentiment"].str.strip()

# Limit data for faster runs
df = df.sample(500, random_state=42)  # Use a subset of data


# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Prepare data
label_encoder = LabelEncoder()
df["Sentiment"] = label_encoder.fit_transform(df["Sentiment"])
texts = df["Text"].tolist()
labels = df["Sentiment"].tolist()
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
train_dataset = SentimentDataset(
    X_train, y_train, tokenizer, max_length=64
)  # Reduced max_length for speed
val_dataset = SentimentDataset(X_val, y_val, tokenizer, max_length=64)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


# Define the model
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        output = self.drop(pooled_output)
        return self.out(output)


# Initialize model, criterion, optimizer, and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentClassifier(n_classes=3)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
scaler = GradScaler()

# Training loop
for epoch in range(5):  # Use more epochs as needed
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), "sentiment_model.pth")


# Predict for new text
def predict_sentiment(text, model, tokenizer):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        max_length=64,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        predicted_class = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]


# Example usage
new_text = "This company is doing exceptionally well."
predicted_class = predict_sentiment(new_text, model, tokenizer)
print(f"Predicted Sentiment: {predicted_class}")
