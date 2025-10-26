"""
Minimal Skip-gram (Word2Vec) with Full Softmax.

Given a center word, predict its surrounding context words.

Architecture:
  - Two embedding matrices: W_in, W_out
  - Softmax over full vocab (no negative sampling)
"""

# ============================================================
# 1. LOAD & PREPROCESS CORPUS
# ============================================================

from collections import Counter
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# --- Load the corpus ---
with open("text8", "r") as f:
    text = f.read().lower()
    tokens = text.split()

min_freq = 5
tokens = tokens[:1_000_000]
# --- Build vocabulary ---
word_freq = Counter(tokens)
vocab = [(w, c) for w, c in word_freq.items() if c >= min_freq]
vocab.sort(key=lambda x: (-x[1], x[0]))

# Add <UNK> to handle rare words
unk_count = sum(c for w, c in word_freq.items() if c < min_freq)
vocab.append(('<UNK>', unk_count))
vocab_size = len(vocab)

# --- Mapping dictionaries ---
word_to_idx = {w: i for i, (w, _) in enumerate(vocab)}
idx_to_word = {i: w for i, (w, _) in enumerate(vocab)}

print(f"Original token count: {len(tokens):,}")
print(f"Filtered vocabulary size: {vocab_size:,}")

# --- Replace tokens with their IDs ---
train_data = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
# train_data = train_data[:1_000_000]  # Truncate for faster experimentation


# ============================================================
# 2. DATASET CREATION
# ============================================================

class SkipGramDataset(Dataset):
    """
    A PyTorch Dataset for generating skip-gram (center_word, context_word) pairs.
    """
    def __init__(self, token_ids, window_size):
        self.token_ids = token_ids
        self.window_size = window_size
        self.pairs = []

        print("Generating training pairs...")
        self._generate_pairs()
        print(f"Generated {len(self.pairs):,} training pairs.")

    def _generate_pairs(self):
        num_tokens = len(self.token_ids)
        for center_word_idx in range(num_tokens):
            center_word_id = self.token_ids[center_word_idx]
            current_window_size = random.randint(1, self.window_size)

            start_idx = max(0, center_word_idx - current_window_size)
            end_idx   = min(num_tokens, center_word_idx + current_window_size + 1)

            for context_word_idx in range(start_idx, end_idx):
                if center_word_idx == context_word_idx:
                    continue
                context_word_id = self.token_ids[context_word_idx]
                self.pairs.append((center_word_id, context_word_id))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center_id, context_id = self.pairs[idx]
        return (
            torch.tensor(center_id, dtype=torch.long),
            torch.tensor(context_id, dtype=torch.long)
        )


# ============================================================
# 3. DATALOADER
# ============================================================

n_embd = 300
window_size = 5
batch_size = 1024
max_steps = 100_000

dataloader = DataLoader(
    dataset=SkipGramDataset(train_data, window_size),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)


# ============================================================
# 4. MODEL DEFINITION
# ============================================================

class SkipGramModel(nn.Module):
    """
    Implements the Naive Skip-Gram model with Full Softmax.
    """
    def __init__(self):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        self.input_embeddings  = nn.Embedding(vocab_size, n_embd)
        self.output_embeddings = nn.Embedding(vocab_size, n_embd)

        # Initialize embeddings
        self.input_embeddings.weight.data.uniform_(-0.5, 0.5)
        self.output_embeddings.weight.data.uniform_(-0.5, 0.5)

    def forward(self, center_words, context_words=None):
        center_embeds = self.input_embeddings(center_words)  # (B, n_embd)
        logits = center_embeds @ self.output_embeddings.weight.T  # (B, vocab_size)
        loss = None
        if context_words is not None:
            loss = F.cross_entropy(logits, context_words)
        return logits, loss


# ============================================================
# 5. TRAINING LOOP
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = SkipGramModel().to(device)
print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

lossi = []
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
data_iter = iter(dataloader)

for i in range(max_steps):
    try:
        center_batch, context_batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        center_batch, context_batch = next(data_iter)

    center_batch  = center_batch.to(device)
    context_batch = context_batch.to(device)

    logits, loss = model(center_batch, context_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1_000 == 0:
        print(f"{i:7d}/{max_steps:7d}: loss = {loss.item():.4f}")

    lossi.append(loss.log10().item())

print("Training finished.")


# ============================================================
# 6. SAVE / LOAD MODEL
# ============================================================

model_save_path = "naive_skipgram_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load trained model (optional)
model = SkipGramModel().to(device)
model.load_state_dict(torch.load("naive_skipgram_model.pth"))
model.eval()
print("Model loaded from naive_skipgram_model.pth")


# ============================================================
# 7. PLOT TRAINING LOSS
# ============================================================

print("Plotting training loss...")

loss_tensor   = torch.tensor(lossi)
smoothed_loss = loss_tensor.view(-1, 1000).mean(dim=1)

plt.figure(figsize=(10, 5))
plt.plot(smoothed_loss)
plt.title("Smoothed Training Loss (Log10)")
plt.xlabel("Step (x1000)")
plt.ylabel("Log10 Loss")
plt.grid(True)
plt.savefig("training_loss.png")

print("Plot saved as training_loss.png")


# ============================================================
# 8. EXTRACT & NORMALIZE EMBEDDINGS
# ============================================================

embeddings = model.input_embeddings.weight.data.cpu()
embeddings = F.normalize(embeddings, p=2, dim=1)
print(f"Embedding matrix shape: {embeddings.shape}")  # [vocab_size, 300]


# ============================================================
# 9. WORD SIMILARITY & ANALOGY TESTS
# ============================================================

def get_most_similar(word, k=8):
    """Finds the k most similar words to a given word."""
    if word not in word_to_idx:
        print(f"'{word}' not in vocabulary.")
        return

    word_id  = word_to_idx[word]
    word_vec = embeddings[word_id]

    print(f"--- Finding words similar to: '{word}' ---")

    similarities = embeddings @ word_vec
    top_k_vals, top_k_indices = torch.topk(similarities, k + 1)

    for i in range(1, k + 1):
        similar_word_id = top_k_indices[i].item()
        similar_word = idx_to_word[similar_word_id]
        similarity_score = top_k_vals[i].item()
        print(f"{i}. {similar_word:<15} (Score: {similarity_score:.4f})")


def get_analogy(a, b, c, k=1):
    """Finds the word d such that: a - b + c = d."""
    for word in [a, b, c]:
        if word not in word_to_idx:
            print(f"'{word}' not in vocabulary.")
            return

    vec_a = embeddings[word_to_idx[a]]
    vec_b = embeddings[word_to_idx[b]]
    vec_c = embeddings[word_to_idx[c]]

    target_vec = vec_a - vec_b + vec_c
    similarities = embeddings @ target_vec

    top_k_vals, top_k_indices = torch.topk(similarities, k + 3)

    print(f"--- Analogy: '{a}' - '{b}' + '{c}' = ? ---")
    for i in range(k + 3):
        result_id = top_k_indices[i].item()
        result_word = idx_to_word[result_id]
        if result_word not in [a, b, c]:
            print(f"Result: {result_word} (Score: {top_k_vals[i].item():.4f})")
            if k == 1:
                return


# --- Run sample tests ---
with torch.no_grad():
    print("\n--- Similarity Tests ---")
    get_most_similar('king')
    get_most_similar('france')
    get_most_similar('three')
    get_most_similar('walking')
    get_most_similar('sadly')

    print("\n--- Analogy Tests ---")
    get_analogy('king', 'man', 'woman')     # -> queen
    get_analogy('walk', 'walking', 'swim')  # -> swimming
    get_analogy('paris', 'france', 'rome')  # -> italy
