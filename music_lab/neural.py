"""Small learned conditional note predictor.

This is an *output-predictive-entropy* signal, NOT attention entropy and NOT
validated artistic intention. Model bundled with the prototype is trained on
synthetic melodies only; replace with a model trained on held-out real music
for meaningful research claims.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from math import log
import numpy as np
import torch
from torch import nn

VOCAB_SIZE = 156
SEQ_LEN = 4
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / 'models' / 'demo_synthetic.pt'


class NoteTransformer(nn.Module):
    def __init__(self, emb: int = 32, heads: int = 2, layers: int = 1):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, emb)
        self.position = nn.Embedding(SEQ_LEN, emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=heads,
                                                     dim_feedforward=emb * 2,
                                                     dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.output = nn.Linear(emb, 128)

    def forward(self, tokens):
        pos = torch.arange(SEQ_LEN, device=tokens.device).unsqueeze(0)
        x = self.embeddings(tokens) + self.position(pos)
        encoded = self.transformer(x)
        return self.output(encoded[:, -1])


def encode_context(previous_soprano: int, prev_root: int, next_root: int, beat: int):
    return [int(previous_soprano), 128+int(prev_root)%12,
            140+int(next_root)%12, 152+int(beat)%4]


class Predictor:
    def __init__(self, path: str | Path = DEFAULT_MODEL):
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Neural checkpoint missing: {self.path}. Run train_model.py first.")
        self.model = NoteTransformer()
        # Load our own generated local checkpoint, never arbitrary untrusted model weights.
        data = torch.load(self.path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(data)
        self.model.eval()

    @lru_cache(maxsize=8192)
    def distribution(self, previous_soprano: int, prev_root: int, next_root: int,
                     beat: int, allowed: tuple[int, ...]):
        tokens = encode_context(previous_soprano, prev_root, next_root, beat)
        with torch.no_grad():
            logits = self.model(torch.tensor([tokens], dtype=torch.long))[0]
            selected_logits = logits[list(allowed)]
            probs = torch.softmax(selected_logits, dim=0).cpu().numpy()
        ent = -sum(float(p) * log(max(float(p), 1e-12)) for p in probs)
        normalized = ent / log(len(allowed)) if len(allowed) > 1 else 0.0
        return dict(zip(allowed, [float(p) for p in probs])), max(0.0, min(1.0, normalized))


class HeuristicPredictor:
    """Transparent fallback for debugging only. Not a Transformer baseline."""
    @lru_cache(maxsize=8192)
    def distribution(self, previous_soprano: int, prev_root: int, next_root: int,
                     beat: int, allowed: tuple[int, ...]):
        scores = np.array([-abs(note-previous_soprano)/3.5
                           - abs(note-73)/21.0 for note in allowed], dtype=np.float64)
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        ent = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        normalized = ent / log(len(allowed)) if len(allowed)>1 else 0.0
        return dict(zip(allowed, [float(p) for p in probs])), float(normalized)
