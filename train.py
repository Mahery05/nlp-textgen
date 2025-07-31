import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import TextDataset, build_vocab
from models.embedding_lstm import LSTMGenerator

with open("data/corpus.txt", encoding='utf-8') as f:
    text = f.read().lower()

vocab, stoi, itos = build_vocab(text)
dataset = TextDataset(text, stoi, seq_length=40)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LSTMGenerator(vocab_size=len(vocab), embed_dim=128, hidden_dim=256)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(10):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.view(-1, len(vocab)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "models/lstm_textgen.pth")
print("Model trained and saved.")
