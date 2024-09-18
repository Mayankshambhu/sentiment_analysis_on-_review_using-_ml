import argparse
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
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import label_binarize
import os
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a sentiment analysis model')
parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset file')
parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
args = parser.parse_args()

def load_data(file_path: str) -> pd.DataFrame:
    data = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            review = json.loads(line)
            text = review.get('text', '')
            rating = review.get('rating')
            if text and rating is not None:
                try:
                    rating = int(rating)
                    if 1 <= rating <= 5:
                        data.append({'text': text, 'label': rating})
                except ValueError:
                    logger.warning(f"Invalid rating value: {rating}")
    return pd.DataFrame(data)

class ReviewDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, n_epochs):
    best_val_loss = float('inf')
    early_stopping_patience = 3
    patience_counter = 0
    train_losses = []
    val_losses = []
    precisions = []
    recalls = []
    f1_scores = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, precision, recall, f1 = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        logger.info(f'Epoch {epoch + 1}/{n_epochs}:')
        logger.info(f'  Training Loss: {avg_train_loss:.4f}')
        logger.info(f'  Validation Loss: {val_loss:.4f}')
        logger.info(f'  Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{timestamp}.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, val_losses, precisions, recalls, f1_scores

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return avg_loss, precision, recall, f1

def plot_training_history(train_losses, val_losses, precisions, recalls, f1_scores):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(precisions, label='Precision')
    plt.plot(recalls, label='Recall')
    plt.plot(f1_scores, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Evaluation Metrics')

    plt.tight_layout()
    plt.savefig(f'training_history_{timestamp}.png')
    plt.close()

def main():
    global timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Loading and preprocessing data...")
    data = load_data(args.file_path)
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = ReviewDataset(train_data, tokenizer, args.max_len)
    val_dataset = ReviewDataset(val_data, tokenizer, args.max_len)
    test_dataset = ReviewDataset(test_data, tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    n_classes = 5
    model = SentimentClassifier(n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler()

    logger.info("Starting training...")
    start_time = time.time()
    train_losses, val_losses, precisions, recalls, f1_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, args.n_epochs
    )
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

    plot_training_history(train_losses, val_losses, precisions, recalls, f1_scores)

    logger.info("Evaluating on test set...")
    test_loss, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, device)
    logger.info(f'Test Loss: {test_loss:.4f}')
    logger.info(f'Test Precision: {test_precision:.4f}')
    logger.info(f'Test Recall: {test_recall:.4f}')
    logger.info(f'Test F1 Score: {test_f1:.4f}')

    # Save the final model and tokenizer
    torch.save(model.state_dict(), f'final_model_{timestamp}.pth')
    tokenizer.save_pretrained(f'tokenizer_{timestamp}')

if __name__ == "__main__":
    main()