import torch
import torch.nn as nn

class PaperRNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        out = out[:, -1, :]  # use output at last time step
        return self.fc(out)  # logits

# --- Training/evaluation loop (main pipeline) ---

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import preprocess_for_dl, stream_chronological_files

SEQUENCE_DIR = "../data/sequences"
SEQUENCE_BLOCKS_PER_DAY = 24
WINDOW_DAYS = 8
N_BLOCKS = WINDOW_DAYS * SEQUENCE_BLOCKS_PER_DAY
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.2  # As in paper, but may need to lower if unstable!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- 1. Prepare training data (first 8 days) --
window_files = []
file_stream = stream_chronological_files(SEQUENCE_DIR)
for i in range(N_BLOCKS):
    try:
        window_files.append(next(file_stream))
    except StopIteration:
        raise RuntimeError("Not enough blocks for 8 days training!")

train_dfs = []
for f in window_files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"Train: Skipping {f}, error: {e}")
        continue
    train_dfs.append(df)
df_train = pd.concat(train_dfs, ignore_index=True)
X_train, y_train, encoders, scaler = preprocess_for_dl(df_train, return_encoders=True)
num_classes = len(encoders['category'].classes_)
input_dim = X_train.shape[1]

# -- Torch dataset/loader --
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # [N, 1, input_dim]
y_train_t = torch.tensor(y_train, dtype=torch.long)
train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# -- 2. Build the RNN model as in paper --
model = PaperRNN(input_dim, num_classes, hidden_dim=64, dropout=0.2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in tqdm(range(EPOCHS), desc="Training epochs"):
    epoch_loss = 0
    for Xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1}", leave=False):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(yb)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_ds):.4f}")

# -- 3. Rolling block-by-block test --
metrics_log = []
block_index = N_BLOCKS + 1
for seq_file in tqdm(file_stream, desc="Testing RNN on session blocks"):
    try:
        df_test = pd.read_csv(seq_file)
    except Exception as e:
        print(f"Test: Skipping {seq_file}, error: {e}")
        continue
    if len(df_test) == 0: continue
    X_test, y_test = preprocess_for_dl(df_test, return_encoders=False)
    if X_test.shape[1] != input_dim:
        print(f"Block {seq_file} feature dim mismatch, skipping")
        continue
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    metrics_log.append({
        "block_index": block_index,
        "file": seq_file,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "support": len(y_test)
    })
    block_index += 1

# -- 4. Save results --
results_df = pd.DataFrame(metrics_log)
os.makedirs("../data/com", exist_ok=True)
results_df.to_csv("../data/com/static_model_metrics_rnn_w8.csv", index=False)
print("Block-level RNN metrics saved!")
print(results_df.describe())
