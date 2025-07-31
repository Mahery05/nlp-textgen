import torch
from torch.utils.data import Dataset
from collections import Counter

# Construit un vocabulaire à partir d'un texte brut
# Renvoie les caractères uniques + les mappings char->index et index->char
def build_vocab(text):
    chars = sorted(list(set(text)))  # Trie tous les caractères uniques du texte
    stoi = {c: i for i, c in enumerate(chars)}  # String-to-index
    itos = {i: c for c, i in stoi.items()}      # Index-to-string
    return chars, stoi, itos

# Dataset personnalisé PyTorch pour le texte
class TextDataset(Dataset):
    def __init__(self, text, stoi, seq_length):
        self.seq_length = seq_length
        self.data = [stoi[c] for c in text]  # Convertit le texte en indices numériques

    def __len__(self):
        return len(self.data) - self.seq_length  # Nombre de séquences exploitables

    def __getitem__(self, idx):
        # Retourne une séquence d'entrée (x) et la séquence cible correspondante (y)
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y
