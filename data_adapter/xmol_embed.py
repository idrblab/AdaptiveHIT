import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

XMOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'xmol_weights', 'FT_to_embedding')
CKPT_DIR  = os.path.join(XMOL_DIR, 'data', 'model', 'step_400000')
VOCAB_PATH = os.path.join(XMOL_DIR, 'package', 'mol', 'molecule_dict')


# ── Weight loading ─────────────────────────────────────────────────────────────

def _load_param(name, shape):
    """Load a PaddlePaddle fluid parameter file; header size auto-detected from shape."""
    path = os.path.join(CKPT_DIR, name)
    n = int(np.prod(shape))
    file_sz = os.path.getsize(path)
    hdr = file_sz - n * 4
    with open(path, 'rb') as f:
        f.seek(hdr)
        data = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
    return torch.from_numpy(data.reshape(shape))


# ── Tokeniser ──────────────────────────────────────────────────────────────────

def _load_vocab(path):
    vocab = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2:
                vocab[parts[0]] = int(parts[1])
    return vocab


def _tokenize(smiles: str, vocab: dict) -> list:
    unk = vocab.get('[UNK]', 111)
    ids = [vocab['[CLS]']]
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == '[':
            j = smiles.find(']', i)
            tok = smiles[i:j + 1] if j != -1 else ch
            i = j + 1 if j != -1 else i + 1
        elif ch in ('B', 'C') and i + 1 < len(smiles) and smiles[i:i + 2] in ('Br', 'Cl'):
            tok = smiles[i:i + 2]
            i += 2
        elif ch == '%' and i + 2 < len(smiles) and smiles[i + 1:i + 3].isdigit():
            tok = smiles[i:i + 3]
            i += 3
        else:
            tok = ch
            i += 1
        ids.append(vocab.get(tok, unk))
    ids.append(vocab['[SEP]'])
    return ids


# ── ERNIE BERT-base (post-norm, 12 layers, hidden=768, heads=12) ───────────────

class _XMolERNIE(nn.Module):
    H = 768
    HEADS = 12
    LAYERS = 12
    FFN = 3072
    MAX_POS = 512
    VOCAB = 112
    SENT = 4

    def __init__(self):
        super().__init__()
        H, HEADS, LAYERS, FFN = self.H, self.HEADS, self.LAYERS, self.FFN

        self.tok_emb  = nn.Embedding(self.VOCAB, H)
        self.pos_emb  = nn.Embedding(self.MAX_POS, H)
        self.sent_emb = nn.Embedding(self.SENT, H)
        self.pre_ln   = nn.LayerNorm(H)

        self.q   = nn.ModuleList([nn.Linear(H, H) for _ in range(LAYERS)])
        self.k   = nn.ModuleList([nn.Linear(H, H) for _ in range(LAYERS)])
        self.v   = nn.ModuleList([nn.Linear(H, H) for _ in range(LAYERS)])
        self.out = nn.ModuleList([nn.Linear(H, H) for _ in range(LAYERS)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(H) for _ in range(LAYERS)])

        self.fc0 = nn.ModuleList([nn.Linear(H, FFN) for _ in range(LAYERS)])
        self.fc1 = nn.ModuleList([nn.Linear(FFN, H) for _ in range(LAYERS)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(H) for _ in range(LAYERS)])

    def load_weights(self):
        H, HEADS, LAYERS, FFN = self.H, self.HEADS, self.LAYERS, self.FFN

        self.tok_emb.weight.data  = _load_param('word_embedding', (self.VOCAB, H))
        self.pos_emb.weight.data  = _load_param('pos_embedding',  (self.MAX_POS, H))
        self.sent_emb.weight.data = _load_param('sent_embedding', (self.SENT, H))
        self.pre_ln.weight.data   = _load_param('pre_encoder_layer_norm_scale', (H,))
        self.pre_ln.bias.data     = _load_param('pre_encoder_layer_norm_bias',  (H,))

        for i in range(LAYERS):
            p = f'encoder_layer_{i}'
            # PaddlePaddle Linear stores W as [in, out]; PyTorch nn.Linear needs [out, in]
            self.q[i].weight.data   = _load_param(f'{p}_multi_head_att_query_fc.w_0',  (H, H)).T.contiguous()
            self.q[i].bias.data     = _load_param(f'{p}_multi_head_att_query_fc.b_0',  (H,))
            self.k[i].weight.data   = _load_param(f'{p}_multi_head_att_key_fc.w_0',    (H, H)).T.contiguous()
            self.k[i].bias.data     = _load_param(f'{p}_multi_head_att_key_fc.b_0',    (H,))
            self.v[i].weight.data   = _load_param(f'{p}_multi_head_att_value_fc.w_0',  (H, H)).T.contiguous()
            self.v[i].bias.data     = _load_param(f'{p}_multi_head_att_value_fc.b_0',  (H,))
            self.out[i].weight.data = _load_param(f'{p}_multi_head_att_output_fc.w_0', (H, H)).T.contiguous()
            self.out[i].bias.data   = _load_param(f'{p}_multi_head_att_output_fc.b_0', (H,))
            self.ln1[i].weight.data = _load_param(f'{p}_post_att_layer_norm_scale', (H,))
            self.ln1[i].bias.data   = _load_param(f'{p}_post_att_layer_norm_bias',  (H,))

            self.fc0[i].weight.data = _load_param(f'{p}_ffn_fc_0.w_0', (H, FFN)).T.contiguous()
            self.fc0[i].bias.data   = _load_param(f'{p}_ffn_fc_0.b_0', (FFN,))
            self.fc1[i].weight.data = _load_param(f'{p}_ffn_fc_1.w_0', (FFN, H)).T.contiguous()
            self.fc1[i].bias.data   = _load_param(f'{p}_ffn_fc_1.b_0', (H,))
            self.ln2[i].weight.data = _load_param(f'{p}_post_ffn_layer_norm_scale', (H,))
            self.ln2[i].bias.data   = _load_param(f'{p}_post_ffn_layer_norm_bias',  (H,))

    def _attn(self, x, i, mask):
        B, L, H = x.shape
        hd = H // self.HEADS
        def split(t):
            return t.view(B, L, self.HEADS, hd).transpose(1, 2)
        Q, K, V = split(self.q[i](x)), split(self.k[i](x)), split(self.v[i](x))
        sc = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hd)
        if mask is not None:
            sc = sc + mask
        ctx = torch.matmul(F.softmax(sc, dim=-1), V)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, H)
        return self.out[i](ctx)

    def forward(self, token_ids):
        B, L = token_ids.shape
        pos  = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        sent = torch.zeros(B, L, dtype=torch.long, device=token_ids.device)
        x = self.pre_ln(self.tok_emb(token_ids) + self.pos_emb(pos) + self.sent_emb(sent))
        for i in range(self.LAYERS):
            x = self.ln1[i](x + self._attn(x, i, None))
            x = self.ln2[i](x + self.fc1[i](F.gelu(self.fc0[i](x))))
        return x  # [B, L, 768]


# ── Module state ───────────────────────────────────────────────────────────────

_model = None
_vocab = None


def _ensure_loaded() -> bool:
    global _model, _vocab
    if _model is not None:
        return True
    try:
        _vocab = _load_vocab(VOCAB_PATH)
        m = _XMolERNIE()
        m.load_weights()
        m.eval()
        _model = m
        return True
    except Exception:
        return False


def embed(smiles: str):
    """Return 768-dim X-MOL embedding tensor, or None if unavailable."""
    if not _ensure_loaded():
        return None
    ids = _tokenize(smiles, _vocab)
    if len(ids) > 512:
        ids = ids[:511] + [_vocab['[SEP]']]
    tok = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        hidden = _model(tok)  # [1, L, 768]
    return hidden[0].mean(dim=0)  # [768]


if __name__ == "__main__":
    # Batch-embed a CSV of drugs into the per-drug .npy layout
    # prebuild_xmol_cache.py expects: {output_dir}/{datatype}/{drugid}/{drugid}.npy
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description='Batch X-Mol embedding generation using the trimmed pretrained weights '
                    '(no PaddlePaddle/full X-Mol toolchain needed -- this is a pure-PyTorch '
                    're-implementation of the same ERNIE architecture)')
    parser.add_argument('--drugs_csv', type=str, required=True,
                        help='CSV with drugid and SMILES columns (see data_adapter/id/*.csv)')
    parser.add_argument('--datatype', type=str, required=True,
                        help='Subdirectory name under --output_dir (matches meta_config.data_subdir)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root drug_emb_dir -- writes {output_dir}/{datatype}/{drugid}/{drugid}.npy')
    args = parser.parse_args()

    df = pd.read_csv(args.drugs_csv)
    out_dir = os.path.join(args.output_dir, args.datatype)
    n_ok, n_fail = 0, 0
    for _, row in df.iterrows():
        drugid, smiles = row['drugid'], row['Compound_ID']
        drug_dir = os.path.join(out_dir, str(drugid))
        npy_path = os.path.join(drug_dir, f"{drugid}.npy")
        if os.path.exists(npy_path):
            continue
        vec = embed(smiles)
        if vec is None:
            print(f"  Warning: failed to embed {drugid} ({smiles})")
            n_fail += 1
            continue
        os.makedirs(drug_dir, exist_ok=True)
        np.save(npy_path, vec.numpy())
        n_ok += 1
    print(f"Done: {n_ok} embedded, {n_fail} failed, out of {len(df)}")
