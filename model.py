import gzip
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import time
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import label_binarize

# Load the dataset from the jsonl.gz file
def load_data(file_path: str) -> pd.DataFrame:
    data = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            review = json.loads(line)
            data.append({'text': review.get('text', ''), 'label': review.get('rating', 0)})
    return pd.DataFrame(data)

# Define a custom dataset class for our data
class ReviewDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long) - 1
        }

# Load the data and create a dataset instance
file_path = 'Data_set/All_Beauty.jsonl.gz'
data = load_data(file_path)

# Split the dataset into training, validation, and test sets
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the dataset and dataloader
max_len = 128
batch_size = 16

train_dataset = ReviewDataset(train_data, tokenizer, max_len)
val_dataset = ReviewDataset(val_data, tokenizer, max_len)
test_dataset = ReviewDataset(test_data, tokenizer, max_len)

# Define a custom collate function to handle padding dynamically within each batch
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True)
    labels = torch.stack(labels)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'label': labels
    }

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define a model class using pre-trained BERT
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

# Create an instance of the custom model
n_classes = 5
model = SentimentClassifier(n_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler and scaler for mixed precision
total_steps = len(train_loader) * 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
scaler = GradScaler()

# Early stopping criteria
early_stopping_patience = 3
min_val_loss = float('inf')
patience_counter = 0

# Track metrics for plotting
train_losses = []
val_losses = []
precisions = []
recalls = []
f1_scores = []

# Measure the total training and testing time
start_time = time.time()

# Training loop with early stopping
for epoch in range(5):
    model.train()
    total_loss = 0
    preds_list = []
    labels_list = []
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation step
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # Check early stopping criteria
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), f'./best_model_{timestamp}.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Testing loop
model.eval()
all_logits = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        all_logits.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate the probabilities
all_logits_tensor = torch.tensor(all_logits)
all_preds_prob = torch.softmax(all_logits_tensor, dim=1).numpy()

# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, torch.argmax(torch.tensor(all_logits), dim=1).numpy())
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, torch.argmax(torch.tensor(all_logits), dim=1).numpy(), average='weighted')

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')

# Plot confusion matrix
cm = confusion_matrix(all_labels, torch.argmax(torch.tensor(all_logits), dim=1).numpy())
cm_df = pd.DataFrame(cm, index=[f'Class {i}' for i in range(n_classes)], columns=[f'Class {i}' for i in range(n_classes)])

plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Calculate and plot ROC curves
all_labels_bin = label_binarize(all_labels, classes=range(n_classes))

fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Print total training and testing time
end_time = time.time()
print(f"Total training and testing time: {end_time - start_time:.2f} seconds")

# Save the model and tokenizer with versioning
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model.state_dict(), f'./sentiment_model_{timestamp}.pth')
tokenizer.save_pretrained(f'./sentiment_tokenizer_{timestamp}')
