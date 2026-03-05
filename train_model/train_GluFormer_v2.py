import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import wandb
import random
import torch
import os
import sys
import pandas as pd
import pandas.core.indexes.base

# Comprehensive compatibility bridge for Pandas legacy objects
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
if not hasattr(pandas.core.indexes.base, 'Int64Index'):
    pandas.core.indexes.base.Int64Index = pandas.core.indexes.base.Index
if not hasattr(pandas.core.indexes.base, 'Float64Index'):
    pandas.core.indexes.base.Float64Index = pandas.core.indexes.base.Index

# Select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Updated Hyperparameters for CGMacros
hyperparameter_defaults = dict(
    epochs=100,
    lr=5e-5,
    weight_decay=1e-4,
    step_size=100,
    gamma=0.99,
    dropout=0.1,
    n_embd=768,           # Increased model capacity
    n_heads=12,
    n_layers=8,           # Increased depth for better pattern recognition
    dim_feedforward=2048, # Increased FF dimension
    max_seq_length=1200,  # Safe limit for 11GB VRAM with these model dims
    seed=42,
    batch_per_gpu=4       # Stable batch size
)

print("Starting Training with config:", hyperparameter_defaults)
wandb.init(config=hyperparameter_defaults, project="CGM_Foundation_CGMacros", allow_val_change=True)
config = wandb.config

# Fix seeds for reproducibility
seed = config.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

VOCAB_SIZE = 1000
PAD_TOKEN = VOCAB_SIZE 

class GlucoseDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if torch.is_tensor(sample):
            return sample.long()
        return torch.tensor(sample, dtype=torch.long)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads=8, n_layers=4, max_seq_length=1000, dropout=0.1, dim_feedforward=512):
        super(TransformerModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size + 1, n_embd)
        
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, n_embd))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=n_embd, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Better performance
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.linear = nn.Linear(n_embd, vocab_size)

    def forward(self, tokens, mask=None):
        seq_length = tokens.size(1)
        token_embeddings = self.embedding(tokens)
        
        # Adding position embeddings (sliced to current seq_length)
        embeddings = token_embeddings + self.pos_embedding[:, :seq_length, :]

        # Causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(tokens.device)
        
        # Transformer forward pass
        transformer_output = self.transformer(embeddings, mask=causal_mask, src_key_padding_mask=mask)
        
        logits = self.linear(transformer_output)
        return logits

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading split files
train_path = "train_cgmacros_split.pt"
val_path = "val_cgmacros_split.pt"

print(f"Loading datasets...")
train_data = torch.load(train_path, weights_only=False)['tokens']
val_data = torch.load(val_path, weights_only=False)['tokens']

train_dataset = GlucoseDataset(train_data)
val_dataset = GlucoseDataset(val_data)

batch_size = config.batch_per_gpu * (torch.cuda.device_count() if torch.cuda.is_available() else 1)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Removed batch_size * 2 to prevent OOM
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = TransformerModel(VOCAB_SIZE, config.n_embd, n_heads=config.n_heads, n_layers=config.n_layers, 
                         max_seq_length=config.max_seq_length, dropout=config.dropout, 
                         dim_feedforward=config.dim_feedforward).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
wandb.log({"num_parameters": sum(p.numel() for p in model.parameters())})

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

# Training Loop
for epoch in range(config.epochs):
    model.train()
    print(f"\n--- Epoch {epoch} ---")
    
    for i, batch in enumerate(train_dataloader):
        # FIX: Explicitly truncate the sequence to max_seq_length
        batch = batch[:, :config.max_seq_length]
        
        inputs = batch.to(device)
        # Shift inputs/targets for next-token prediction
        inputs, targets = inputs[:, :-1], inputs[:, 1:]
        
        padding_mask = (inputs == PAD_TOKEN)
        
        optimizer.zero_grad()
        logits = model(inputs, mask=padding_mask)
        
        # Reshape for CrossEntropy
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=PAD_TOKEN, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            acc = (logits.argmax(dim=-1) == targets).float()[targets != PAD_TOKEN].mean()
            print(f"Batch {i} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
            wandb.log({"train_loss": loss.item(), "train_acc": acc.item(), "lr": scheduler.get_last_lr()[0]})

    scheduler.step()

    # Validation Phase
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch in val_dataloader:
            # FIX: Truncate validation sequences too
            batch = batch[:, :config.max_seq_length]
            
            inputs = batch.to(device)
            inputs, targets = inputs[:, :-1], inputs[:, 1:]
            padding_mask = (inputs == PAD_TOKEN)
            
            logits = model(inputs, mask=padding_mask)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=PAD_TOKEN)
            
            val_loss += loss.item()
            val_acc += (logits.argmax(dim=-1) == targets).float()[targets != PAD_TOKEN].mean().item()

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_acc = val_acc / len(val_dataloader)
    print(f">> Validation Results | Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f}")
    wandb.log({"val_loss": avg_val_loss, "val_acc": avg_val_acc, "epoch": epoch})

# Save results
save_dir = f"Models/{wandb.run.name}"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{save_dir}/GluFormer_CGMacros_v2.pt"
torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_path)
print(f"Model saved to {save_path}")