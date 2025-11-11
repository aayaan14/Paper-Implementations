"""
Skip-gram (Word2Vec) with Hierarchical Softmax Implementation

Given a center word, predict its surrounding context words using a Huffman tree
to avoid computing softmax over the entire vocabulary (O(log V) instead of O(V)).
"""

import random
import heapq
from collections import Counter
from itertools import count
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# HYPERPARAMETERS
# ============================================================

batch_size = 1024
n_embd = 300
window_size = 5
min_freq = 5
max_tokens = 1_000_000
max_steps = 50_000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

print("Loading corpus...")
with open("text8", "r") as f:
    text = f.read().lower()
    tokens = text.split()[:max_tokens]

print(f"Loaded {len(tokens):,} tokens")

# Build vocabulary
word_freq = Counter(tokens)
vocab = [(w, c) for w, c in word_freq.items() if c >= min_freq]
vocab.sort(key=lambda x: (-x[1], x[0]))  # sort by frequency, then alphabetically

# Add <UNK> token for rare words
unk_count = sum(c for w, c in word_freq.items() if c < min_freq)
vocab.append(('<UNK>', unk_count))

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size:,} (min_freq={min_freq})")

# Create mappings
word_to_idx = {w: i for i, (w, _) in enumerate(vocab)}
idx_to_word = {i: w for i, (w, _) in enumerate(vocab)}

# Convert tokens to indices
train_data = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]

# ============================================================
# HUFFMAN TREE CONSTRUCTION
# ============================================================

class LeafNode:
    """Leaf node representing a vocabulary word."""
    def __init__(self, word_id: int, freq: int):
        self.word_id = word_id
        self.freq = freq
        self.is_leaf = True

class InternalNode:
    """Internal node in the Huffman tree for binary decisions."""
    def __init__(self, internal_id: int, freq: int, left, right):
        self.internal_id = internal_id
        self.freq = freq
        self.left = left
        self.right = right
        self.is_leaf = False

def build_huffman_tree(word_freqs: Dict[int, int]) -> InternalNode:
    """
    Build a Huffman tree from word frequencies.
    """
    heap = []
    tie_breaker = count()  # ensures stable sorting
    
    # Initialize heap with leaf nodes
    for word_id, freq in word_freqs.items():
        node = LeafNode(word_id, freq)
        heapq.heappush(heap, (freq, next(tie_breaker), node))
    
    internal_id = 0
    
    # Build tree bottom-up
    while len(heap) > 1:
        freq1, _, node1 = heapq.heappop(heap)
        freq2, _, node2 = heapq.heappop(heap)
        
        merged = InternalNode(
            internal_id=internal_id,
            freq=freq1 + freq2,
            left=node1,
            right=node2
        )
        
        heapq.heappush(heap, (merged.freq, next(tie_breaker), merged))
        internal_id += 1
    
    _, _, root = heap[0]
    print(f"Built Huffman tree: {internal_id} internal nodes")
    return root

def extract_paths(root: InternalNode) -> Dict[int, Tuple[List[int], List[int]]]:
    """
    Extract path from root to each word.
    Returns: {word_id: (path_nodes, path_codes)}
      - path_nodes: internal node IDs to traverse
      - path_codes: 0=left, 1=right
    """
    paths = {}
    
    def dfs(node, path_nodes, path_codes):
        if node.is_leaf:
            paths[node.word_id] = (list(path_nodes), list(path_codes))
            return
        
        # Go left (code=0)
        path_nodes.append(node.internal_id)
        path_codes.append(0)
        dfs(node.left, path_nodes, path_codes)
        path_nodes.pop()
        path_codes.pop()
        
        # Go right (code=1)
        path_nodes.append(node.internal_id)
        path_codes.append(1)
        dfs(node.right, path_nodes, path_codes)
        path_nodes.pop()
        path_codes.pop()
    
    dfs(root, [], [])
    
    max_path_len = max(len(p[0]) for p in paths.values())
    avg_path_len = sum(len(p[0]) for p in paths.values()) / len(paths)
    print(f"Extracted paths: max_len={max_path_len}, avg_len={avg_path_len:.1f}")
    
    return paths, max_path_len

# Build tree and extract paths
word_freqs = {word_to_idx[w]: c for w, c in vocab}
root = build_huffman_tree(word_freqs)
paths, max_path_len = extract_paths(root)

# ============================================================
# DATASET
# ============================================================

class HSSkipGramDataset(Dataset):
    """
    Skip-gram dataset with Huffman tree paths.
    For each (center, context) pair, we return the center word
    and the Huffman path to the context word.
    """
    def __init__(self, token_ids, window_size, paths, max_path_len):
        self.token_ids = token_ids
        self.window_size = window_size
        self.paths = paths
        self.max_path_len = max_path_len
        self.pairs = []
        
        self._generate_pairs()
    
    def _generate_pairs(self):
        """Generate all (center, context) pairs."""
        print("Generating training pairs...")
        num_tokens = len(self.token_ids)
        
        for i in range(num_tokens):
            center_id = self.token_ids[i]
            
            # Dynamic window size (as in original word2vec)
            w = random.randint(1, self.window_size)
            
            # Get context words
            start = max(0, i - w)
            end = min(num_tokens, i + w + 1)
            
            for j in range(start, end):
                if i != j:
                    context_id = self.token_ids[j]
                    self.pairs.append((center_id, context_id))
        
        print(f"Generated {len(self.pairs):,} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center_id, context_id = self.pairs[idx]
        path_nodes, path_codes = self.paths[context_id]
        
        # Pad paths to max_path_len
        pad_len = self.max_path_len - len(path_nodes)
        path_nodes = path_nodes + [0] * pad_len
        path_codes = path_codes + [0] * pad_len
        
        return (
            torch.tensor(center_id, dtype=torch.long),
            torch.tensor(path_nodes, dtype=torch.long),
            torch.tensor(path_codes, dtype=torch.long)
        )

# Create dataset and dataloader
dataset = HSSkipGramDataset(train_data, window_size, paths, max_path_len)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

# ============================================================
# MODEL
# ============================================================

class HS_SkipGramModel(nn.Module):
    """
    Skip-gram with Hierarchical Softmax.
    
    Instead of softmax over V words (expensive), we make log(V) binary
    decisions following a Huffman tree path.
    """
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        
        # Input embeddings: one per word
        self.in_embd = nn.Embedding(vocab_size, n_embd)
        
        # Output embeddings: one per internal node
        num_internal = vocab_size - 1
        self.out_embd = nn.Embedding(num_internal, n_embd)
        
        # Initialize
        nn.init.uniform_(self.in_embd.weight, -0.5/n_embd, 0.5/n_embd)
        nn.init.uniform_(self.out_embd.weight, -0.5/n_embd, 0.5/n_embd)
    
    def forward(self, center_words, path_nodes, path_codes):
        """
        Args:
            center_words: [B] - center word IDs
            path_nodes: [B, L] - internal nodes in paths
            path_codes: [B, L] - binary codes (0=left, 1=right)
        
        Returns:
            loss: scalar
        """
        center_emb = self.in_embd(center_words)  # [B, D]
        node_emb = self.out_embd(path_nodes)      # [B, L, D]
        
        scores = torch.sum(node_emb * center_emb.unsqueeze(1), dim=2)  # [B, L]
        
        # Sigmoid for binary classification at each node
        logits = torch.sigmoid(scores)
        
        # Probability of taking the correct path (0=left, 1=right)
        logits = torch.where(path_codes == 0, logits, 1 - logits)
        
        # Negative log-likelihood
        loss = -torch.log(logits + 1e-10).sum(dim=1).mean()
        
        return loss
    
    @torch.no_grad()
    def get_embeddings(self):
        """Return normalized word embeddings."""
        return F.normalize(self.in_embd.weight, p=2, dim=1)

# ============================================================
# TRAINING
# ============================================================

model = HS_SkipGramModel(vocab_size, n_embd).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

lossi = []
data_iter = iter(dataloader)

print("\nTraining...")
for step in range(max_steps):
    
    # Get batch
    try:
        center, path_nodes, path_codes = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        center, path_nodes, path_codes = next(data_iter)
    
    # Move to device
    center = center.to(device)
    path_nodes = path_nodes.to(device)
    path_codes = path_codes.to(device)
    
    # Forward
    loss = model(center, path_nodes, path_codes)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Logging
    lossi.append(loss.log10().item())
    
    if step % 1000 == 0:
        print(f"step {step:5d}/{max_steps}: loss = {loss.item():.4f}")

print("Training complete!")

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(model.state_dict(), "skipgram_hs.pth")
print("Model saved to skipgram_hs.pth")

# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(figsize=(10, 5))
loss_tensor = torch.tensor(lossi)
smoothed = loss_tensor.view(-1, 1000).mean(dim=1)
plt.plot(smoothed)
plt.title("Training Loss (Log10, smoothed)")
plt.xlabel("Step (x1000)")
plt.ylabel("Log10 Loss")
plt.grid(True)
plt.savefig("loss.png")
print("Loss plot saved to loss.png")

# ============================================================
# EVALUATION
# ============================================================

embeddings = model.get_embeddings().cpu()
print(f"Embeddings shape: {embeddings.shape}")

def get_similar(word, k=8):
    """Find k most similar words."""
    if word not in word_to_idx:
        print(f"'{word}' not in vocabulary")
        return
    
    idx = word_to_idx[word]
    vec = embeddings[idx]
    
    # Cosine similarity
    sims = embeddings @ vec
    vals, idxs = torch.topk(sims, k + 1)
    
    print(f"\nMost similar to '{word}':")
    for i in range(1, k + 1):
        w = idx_to_word[idxs[i].item()]
        s = vals[i].item()
        print(f"  {i}. {w:<15} ({s:.3f})")

def analogy(a, b, c, k=1):
    """Solve: a - b + c = ?"""
    for word in [a, b, c]:
        if word not in word_to_idx:
            print(f"'{word}' not in vocabulary")
            return
    
    vec_a = embeddings[word_to_idx[a]]
    vec_b = embeddings[word_to_idx[b]]
    vec_c = embeddings[word_to_idx[c]]
    
    target = vec_a - vec_b + vec_c
    target = F.normalize(target, dim=0)
    
    sims = embeddings @ target
    vals, idxs = torch.topk(sims, k + 3)
    
    print(f"\nAnalogy: {a} - {b} + {c} = ?")
    for i in range(k + 3):
        w = idx_to_word[idxs[i].item()]
        if w not in [a, b, c]:
            print(f"  {w} ({vals[i].item():.3f})")
            if k == 1:
                break

# Run tests
print("\n" + "="*50)
print("EVALUATION")
print("="*50)

get_similar('king')
get_similar('france')
get_similar('three')
get_similar('walking')

analogy('king', 'man', 'woman')
analogy('paris', 'france', 'rome')
analogy('walk', 'walking', 'swim')