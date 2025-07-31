import torch
from models.embedding_lstm import LSTMGenerator
from utils.preprocessing import build_vocab

with open("data/corpus.txt", encoding='utf-8') as f:
    text = f.read().lower()

vocab, stoi, itos = build_vocab(text)
model = LSTMGenerator(vocab_size=len(vocab), embed_dim=128, hidden_dim=256)
model.load_state_dict(torch.load("models/lstm_textgen.pth"))
model.eval()

def generate_text(start_seq, length=100):
    input_seq = torch.tensor([[stoi[c] for c in start_seq]], dtype=torch.long)
    generated = start_seq
    hidden = None

    for _ in range(length):
        out, hidden = model(input_seq[:, -1].unsqueeze(1), hidden)
        prob = torch.softmax(out[:, -1], dim=-1)
        char_idx = torch.multinomial(prob, num_samples=1).item()
        generated += itos[char_idx]
        input_seq = torch.cat([input_seq, torch.tensor([[char_idx]])], dim=1)
    return generated

print(generate_text("le ciel est "))