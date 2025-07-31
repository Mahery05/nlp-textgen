import torch.nn as nn

# Définition d'un modèle de génération de texte basé sur LSTM
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        # Couche d'embedding pour convertir les indices de caractères en vecteurs denses
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # LSTM pour apprendre la séquence de texte
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # Couche linéaire pour prédire le caractère suivant
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # Applique l'embedding
        x = self.embedding(x)
        # Passe à travers le LSTM (renvoie la sortie et l'état caché)
        out, hidden = self.lstm(x, hidden)
        # Passe la sortie à travers la couche fully connected
        out = self.fc(out)
        return out, hidden
