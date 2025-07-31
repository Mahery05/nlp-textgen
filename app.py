from flask import Flask, render_template, request
import torch
from models.embedding_lstm import LSTMGenerator
from utils.preprocessing import build_vocab

# Création de l'application Flask
app = Flask(__name__)

# Chargement du corpus et génération du vocabulaire
with open("data/corpus.txt", encoding='utf-8') as f:
    text = f.read().lower()

# Construction du vocabulaire et chargement du modèle entraîné
vocab, stoi, itos = build_vocab(text)
model = LSTMGenerator(vocab_size=len(vocab), embed_dim=128, hidden_dim=256)
model.load_state_dict(torch.load("models/lstm_textgen.pth"))
model.eval()  # Passage en mode évaluation

# Fonction de génération de texte à partir d'un prompt
# Utilise le modèle LSTM pour produire un texte caractère par caractère
def generate_text(start_seq, length=300):
    # Encodage du prompt en indices
    input_seq = torch.tensor([[stoi.get(c, 0) for c in start_seq]], dtype=torch.long)
    generated = start_seq
    hidden = None  # État initial caché du LSTM

    # Boucle de génération
    for _ in range(length):
        # Prédiction du caractère suivant
        out, hidden = model(input_seq[:, -1].unsqueeze(1), hidden)
        prob = torch.softmax(out[:, -1], dim=-1)  # Distribution de probas
        char_idx = torch.multinomial(prob, num_samples=1).item()  # Échantillonnage
        generated += itos[char_idx]  # Ajout du caractère généré
        input_seq = torch.cat([input_seq, torch.tensor([[char_idx]])], dim=1)  # Mise à jour de la séquence
    return generated

# Route principale de l'application web
@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""
    if request.method == "POST":
        # Récupération du prompt et de la longueur depuis le formulaire
        prompt = request.form.get("prompt", "")
        length = int(request.form.get("length", 300))
        # Génération du texte à partir du modèle
        generated_text = generate_text(prompt, length)
    # Affichage dans le template HTML
    return render_template("index.html", generated_text=generated_text)

# Point d'entrée de l'application
if __name__ == "__main__":
    app.run(debug=True)
