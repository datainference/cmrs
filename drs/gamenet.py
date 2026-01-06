import math
import numpy as np
import pandas as pd 
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def read_df(filename):
    df = pd.read_pickle(filename)
    df = df[df.columns[:-2]] 
    prefixes = ("hash_hn", "hash_doctor", "dep", "dep_name", "clinic", "date_visit", "age", "sex", "note_token", )
    df = df.loc[:, ~df.columns.str.startswith(prefixes)]
    return df

# Helpers: parsing + history depth
def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))

def parse_code_list(x: Any) -> List[str]:
    """Robust-ish parser. Best case: your DF already stores Python lists."""
    if _is_nan(x):
        return []
    if isinstance(x, list):
        return [str(t) for t in x if not _is_nan(t)]
    if isinstance(x, tuple):
        return [str(t) for t in x if not _is_nan(t)]
    if isinstance(x, str):
        s = x.strip()
        # Try literal list like "['A', 'B']"
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple)):
                    return [str(t) for t in v if not _is_nan(t)]
            except Exception:
                pass
        # Fallback split
        for sep in ["|", ";", ","]:
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]
        if s:
            return [s]
    # Unknown type -> string it
    return [str(x)]

def infer_max_history(df: pd.DataFrame, prefix: str = "y_") -> int:
    mx = 0
    for c in df.columns:
        if c.startswith(prefix):
            try:
                k = int(c[len(prefix):])
                mx = max(mx, k)
            except Exception:
                pass
    return mx

# Vocab
@dataclass
class Vocabs:
    diag2id: Dict[str, int]          
    id2diag: List[str]
    med2id: Dict[str, int]           
    id2med: List[str]

    @property
    def n_diag(self) -> int:
        return len(self.id2diag)  

    @property
    def n_med(self) -> int:
        return len(self.id2med)

def build_vocabs_from_train_df(df_train: pd.DataFrame) -> Vocabs:
    K = infer_max_history(df_train, prefix="y_")

    # meds from y only (current visit meds). This avoids duplicating visits via y_k.
    all_meds = []
    all_diags = []

    # current visit
    all_meds.extend(df_train["y"].apply(parse_code_list).tolist())
    all_diags.extend(df_train["icd10"].apply(parse_code_list).tolist())

    # history diags (optional but recommended)
    for k in range(1, K + 1):
        dk = f"icd10_{k}"
        if dk in df_train.columns:
            all_diags.extend(df_train[dk].apply(parse_code_list).tolist())

    med_set = sorted({m for row in all_meds for m in row})
    diag_set = sorted({d for row in all_diags for d in row})

    # diag: reserve 0 for PAD
    id2diag = ["<PAD>"] + diag_set
    diag2id = {d: i for i, d in enumerate(id2diag)}

    id2med = med_set
    med2id = {m: i for i, m in enumerate(id2med)}

    return Vocabs(diag2id=diag2id, id2diag=id2diag, med2id=med2id, id2med=id2med)

# DDI adjacency builders
def add_self_loops_sparse(A: torch.Tensor) -> torch.Tensor:
    """A: sparse COO (V,V) -> A + I"""
    A = A.coalesce()
    V = A.size(0)
    idx = torch.arange(V, dtype=torch.long, device=A.device)
    I = torch.sparse_coo_tensor(
        torch.stack([idx, idx], dim=0),
        torch.ones(V, dtype=torch.float32, device=A.device),
        size=(V, V),
    ).coalesce()
    return (A + I).coalesce()

def ddi_matrix_atcmap_to_model_sparse(
    A_ddi: np.ndarray,
    atcmap: dict,              
    med2id: dict,              
    device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      A_ddi_raw  : sparse COO [V,V] (binary) aligned to med2id (for DDI penalty)
      A_ddi_norm : sparse COO [V,V] normalized (+self-loops) aligned to med2id (for DDI-GCN)
    """
    V = len(med2id)

    # Ensure we can map matrix indices -> ATC codes
    # (atcmap keys should cover 0..A_ddi.shape[0]-1)
    n = A_ddi.shape[0]
    assert A_ddi.shape[0] == A_ddi.shape[1], "A_ddi must be square"
    # Build array index->code (fast)
    idx2code = [atcmap[i] for i in range(n)]

    # Collect edges from matrix and remap to model ids
    r, c = np.nonzero(A_ddi > 0)
    rows, cols = [], []
    for i, j in zip(r.tolist(), c.tolist()):
        ci = idx2code[i]
        cj = idx2code[j]
        if ci in med2id and cj in med2id and ci != cj:
            rows.append(med2id[ci])
            cols.append(med2id[cj])

    if len(rows) == 0:
        A_raw = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            size=(V, V),
        ).coalesce()
    else:
        idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
        val = torch.ones(len(rows), dtype=torch.float32, device=device)
        A_raw = torch.sparse_coo_tensor(idx, val, size=(V, V), device=device).coalesce()

    # Make sure it's binary and symmetric (safe)
    A_raw = (A_raw + A_raw.transpose(0, 1)).coalesce()
    A_raw = torch.sparse_coo_tensor(A_raw.indices(), torch.ones_like(A_raw.values()), A_raw.size(), device=device).coalesce()

    # For GCN, normalize (usually with self-loops)
    A_norm = normalize_sparse_adj(add_self_loops_sparse(A_raw))

    return A_raw, A_norm
    
# EHR co-prescription adjacency (from y)
def build_ehr_coprescription_adj_sparse(
    df_train: pd.DataFrame,
    med2id: Dict[str, int],
    y_col: str = "y",
    add_self_loops: bool = True,
    device: Optional[torch.device] = None,
    ) -> torch.Tensor:
    """
    Build EHR graph adjacency: connect two meds if they appear in the same visit medication set.
    Uses only df_train[y_col] (current visit meds).
    Returns sparse COO adjacency (V,V).
    """
    V = len(med2id)
    edge_set = set()

    for meds in df_train[y_col].apply(parse_code_list).tolist():
        ids = [med2id[m] for m in meds if m in med2id]
        ids = list(sorted(set(ids)))
        # clique edges
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edge_set.add((ids[i], ids[j]))
                edge_set.add((ids[j], ids[i]))

    if add_self_loops:
        for i in range(V):
            edge_set.add((i, i))

    if len(edge_set) == 0:
        idx = torch.zeros((2, 0), dtype=torch.long)
        val = torch.zeros((0,), dtype=torch.float32)
    else:
        rows, cols = zip(*edge_set)
        idx = torch.tensor([rows, cols], dtype=torch.long)
        val = torch.ones((idx.shape[1],), dtype=torch.float32)

    A = torch.sparse_coo_tensor(idx, val, size=(V, V)).coalesce()
    if device is not None:
        A = A.to(device)
    return A

def normalize_sparse_adj(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}
    A is sparse COO (coalesced)
    """
    A = A.coalesce()
    idx = A.indices()
    val = A.values()
    V = A.size(0)

    deg = torch.zeros((V,), device=val.device, dtype=val.dtype)
    deg.scatter_add_(0, idx[0], val)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)

    val_norm = deg_inv_sqrt[idx[0]] * val * deg_inv_sqrt[idx[1]]
    An = torch.sparse_coo_tensor(idx, val_norm, size=A.size()).coalesce()
    return An

# Dataset: flat DF -> (sequence, history meds, target meds)

class FlatVisitDataset(Dataset):
    """
    Each item yields:
      - visits_diags: List[List[int]] (history visits diags ... current visit diags)
      - visits_hist_meds: List[List[int]] (history meds only; len = len(visits_diags)-1)
      - target_meds: List[int] (current y)
    """
    def __init__(self, df: pd.DataFrame, vocabs: Vocabs):
        self.df = df.reset_index(drop=True)
        self.v = vocabs
        self.K = infer_max_history(self.df, prefix="y_")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]

        # Build history in chronological order: (_K ... _1) are older -> more recent
        hist_diags: List[List[int]] = []
        hist_meds: List[List[int]] = []

        for k in range(self.K, 0, -1):
            yk = f"y_{k}"
            dk = f"icd10_{k}"
            if yk not in self.df.columns or dk not in self.df.columns:
                continue

            meds = parse_code_list(row[yk])
            diags = parse_code_list(row[dk])
            if len(meds) == 0 and len(diags) == 0:
                continue

            med_ids = [self.v.med2id[m] for m in meds if m in self.v.med2id]
            diag_ids = [self.v.diag2id[d] for d in diags if d in self.v.diag2id]

            hist_meds.append(med_ids)
            hist_diags.append(diag_ids)

        # Current visit
        cur_diags = parse_code_list(row["icd10"])
        cur_diag_ids = [self.v.diag2id[d] for d in cur_diags if d in self.v.diag2id]

        target_meds = parse_code_list(row["y"])
        target_med_ids = [self.v.med2id[m] for m in target_meds if m in self.v.med2id]

        visits_diags = hist_diags + [cur_diag_ids]

        return {
            "visits_diags": visits_diags,       # len T
            "hist_meds": hist_meds,             # len T-1
            "target_meds": target_med_ids,      # current meds
        }

def collate_flat_visits(batch: List[dict], n_med: int) -> dict:
    """
    Create padded tensors:
      diag_ids: [B, T, L]
      diag_mask: [B, T, L]
      hist_med_mh: [B, T-1, V]
      hist_mask: [B, T-1] indicates real history positions
      lengths: [B] number of real visits (history + current)
      target_mh: [B, V]
    """
    B = len(batch)
    T = max(len(x["visits_diags"]) for x in batch)
    # max diag codes per visit (within batch)
    L = 1
    for x in batch:
        for v in x["visits_diags"]:
            L = max(L, len(v))

    diag_ids = torch.zeros((B, T, L), dtype=torch.long)  # 0 is PAD
    diag_mask = torch.zeros((B, T, L), dtype=torch.bool)

    # history meds: max length is T-1
    hist_med_mh = torch.zeros((B, max(T - 1, 1), n_med), dtype=torch.float32)
    hist_mask = torch.zeros((B, max(T - 1, 1)), dtype=torch.bool)

    lengths = torch.zeros((B,), dtype=torch.long)
    target_mh = torch.zeros((B, n_med), dtype=torch.float32)

    for i, x in enumerate(batch):
        visits = x["visits_diags"]
        hist_meds = x["hist_meds"]
        lengths[i] = len(visits)

        # pad AFTER current visit (right padding)
        for t in range(len(visits)):
            codes = visits[t]
            if len(codes) == 0:
                continue
            diag_ids[i, t, :len(codes)] = torch.tensor(codes, dtype=torch.long)
            diag_mask[i, t, :len(codes)] = True

        # history meds fill positions [0 .. hist_len-1]
        hist_len = max(len(visits) - 1, 0)
        for t in range(hist_len):
            meds = hist_meds[t] if t < len(hist_meds) else []
            if len(meds) > 0:
                hist_med_mh[i, t, meds] = 1.0
            hist_mask[i, t] = True

        # target meds
        if len(x["target_meds"]) > 0:
            target_mh[i, x["target_meds"]] = 1.0

    return {
        "diag_ids": diag_ids,
        "diag_mask": diag_mask,
        "hist_med_mh": hist_med_mh,
        "hist_mask": hist_mask,
        "lengths": lengths,
        "target_mh": target_mh,
    }

# GAMENet model 
class SparseGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, A_norm: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # A_norm: sparse (V,V), X: (V,in_dim)
        H = torch.sparse.mm(A_norm, X)
        H = F.relu(self.W1(H))
        H = torch.sparse.mm(A_norm, H)
        H = self.W2(H)
        return H

def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    scores: [B, T]
    mask: [B, T] bool
    returns: [B, T] where rows with all-false mask -> zeros
    """
    if scores.numel() == 0:
        return scores
    very_neg = torch.finfo(scores.dtype).min
    scores2 = scores.masked_fill(~mask, very_neg)
    # if a row has no valid positions, avoid NaNs
    has_any = mask.any(dim=dim, keepdim=True)
    probs = torch.softmax(scores2, dim=dim)
    probs = torch.where(has_any, probs, torch.zeros_like(probs))
    return probs

# Train step / loop (simple)
class GAMENet(nn.Module):
    """
    GAMENet core pieces:
      - Patient query from GRU over visit embeddings (diagnosis only here)
      - Memory Bank: medication embeddings refined by 2 GCNs (EHR + DDI) and fused
      - Dynamic Memory: key-value over historical visits (keys=query_t, values=multi-hot meds_t)
      - Output: concat(query, o_b, o_d) -> logits
      - DDI loss: p^T A_ddi p
    """
    def __init__(
        self,
        n_diag: int,
        n_med: int,
        emb_dim: int = 64,
        rnn_hidden: int = 128,
        gcn_hidden: int = 64,
        ddi_lambda: float = 0.1,
    ):
        super().__init__()
        self.n_med = n_med
        self.ddi_lambda = ddi_lambda

        # diagnosis embedding (0 is PAD)
        self.diag_emb = nn.Embedding(n_diag, emb_dim, padding_idx=0)

        # GRU over visits
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=rnn_hidden, batch_first=True)

        # query transform
        self.query_fc = nn.Sequential(
            nn.Linear(rnn_hidden, emb_dim),
            nn.ReLU(),
        )

        # medication node initial features
        self.med_emb = nn.Embedding(n_med, emb_dim)

        # two GCNs for Memory Bank
        self.gcn_ehr = SparseGCN(in_dim=emb_dim, hidden_dim=gcn_hidden, out_dim=emb_dim)
        self.gcn_ddi = SparseGCN(in_dim=emb_dim, hidden_dim=gcn_hidden, out_dim=emb_dim)

        # fusion weight (learned scalar in (0,1))
        self.beta_logit = nn.Parameter(torch.tensor(0.0))

        # final predictor
        self.out = nn.Linear(emb_dim * 3, n_med)

    def build_memory_bank(self, A_ehr_norm: torch.Tensor, A_ddi_norm: torch.Tensor) -> torch.Tensor:
        X0 = self.med_emb.weight  # (V, d)
        M_ehr = self.gcn_ehr(A_ehr_norm, X0)
        M_ddi = self.gcn_ddi(A_ddi_norm, X0)
        beta = torch.sigmoid(self.beta_logit)
        MB = beta * M_ehr + (1.0 - beta) * M_ddi
        return MB  # (V, d)

    def forward(
        self,
        diag_ids: torch.Tensor,     # [B, T, L]
        diag_mask: torch.Tensor,    # [B, T, L]
        lengths: torch.Tensor,      # [B]
        hist_med_mh: torch.Tensor,  # [B, T-1, V]
        hist_mask: torch.Tensor,    # [B, T-1]
        A_ehr_norm: torch.Tensor,
        A_ddi_norm: torch.Tensor,
    ) -> torch.Tensor:
        B, T, L = diag_ids.shape

        # Visit embeddings: sum/avg of diag embeddings per visit
        E = self.diag_emb(diag_ids)  # [B,T,L,d]
        m = diag_mask.unsqueeze(-1).float()  # [B,T,L,1]
        summed = (E * m).sum(dim=2)  # [B,T,d]
        cnt = diag_mask.sum(dim=2).clamp(min=1).unsqueeze(-1).float()
        visit_emb = summed / cnt  # [B,T,d]

        # GRU
        out_seq, _ = self.gru(visit_emb)  # [B,T,h]

        # Transform every step into query vectors
        q_all = self.query_fc(out_seq)  # [B,T,d]

        # Current query = at index lengths-1 for each sample
        idx = (lengths - 1).clamp(min=0)  # [B]
        q_cur = q_all[torch.arange(B, device=q_all.device), idx]  # [B,d]

        # Keys for dynamic memory = historical queries (positions 0..hist_len-1)
        # Note: hist_mask already indicates which positions are real history (aligned to first T-1 slots)
        q_hist = q_all[:, :max(T - 1, 1), :]  # [B,T-1,d]

        # Memory bank (graph-augmented)
        MB = self.build_memory_bank(A_ehr_norm, A_ddi_norm)  # [V,d]

        # Read Memory Bank (content attention)
        attn_b = torch.softmax(q_cur @ MB.t(), dim=-1)  # [B,V]
        o_b = attn_b @ MB  # [B,d]

        # Read Dynamic Memory
        # similarity: [B, T-1]
        sim = torch.einsum("bd,btd->bt", q_cur, q_hist)
        alpha = masked_softmax(sim, hist_mask, dim=1)  # [B, T-1]
        # history medication distribution: [B,V]
        a = torch.einsum("bt,btv->bv", alpha, hist_med_mh)
        # project distribution back to embedding space via MB: [B,d]
        o_d = a @ MB

        logits = self.out(torch.cat([q_cur, o_b, o_d], dim=-1))  # [B,V]
        return logits

    def ddi_loss(self, probs: torch.Tensor, A_ddi: torch.Tensor) -> torch.Tensor:
        """
        probs: [B,V] in (0,1)
        A_ddi: sparse COO [V,V] (major-only)
        returns mean p^T A p
        """
        # (A @ p^T)^T -> [B,V]
        Ap = torch.sparse.mm(A_ddi, probs.t()).t()
        # sum_j p_j * (Ap)_j
        loss = (probs * Ap).sum(dim=1).mean()
        return loss

def train_gamenet(
    model: GAMENet,
    train_loader: DataLoader,
    A_ehr_norm: torch.Tensor,
    A_ddi_norm: torch.Tensor,
    A_ddi_raw: torch.Tensor,  
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 3,
    grad_clip: float = 5.0,
    ) -> None:
    model.to(device)
    A_ehr_norm = A_ehr_norm.to(device)
    A_ddi_norm = A_ddi_norm.to(device)
    A_ddi_raw = A_ddi_raw.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for batch in train_loader:
            diag_ids = batch["diag_ids"].to(device)
            diag_mask = batch["diag_mask"].to(device)
            lengths = batch["lengths"].to(device)
            hist_med_mh = batch["hist_med_mh"].to(device)
            hist_mask = batch["hist_mask"].to(device)
            target = batch["target_mh"].to(device)

            logits = model(
                diag_ids=diag_ids,
                diag_mask=diag_mask,
                lengths=lengths,
                hist_med_mh=hist_med_mh,
                hist_mask=hist_mask,
                A_ehr_norm=A_ehr_norm,
                A_ddi_norm=A_ddi_norm,
            )

            bce = F.binary_cross_entropy_with_logits(logits, target)

            probs = torch.sigmoid(logits)
            ddi_pen = model.ddi_loss(probs, A_ddi_raw)

            loss = bce + model.ddi_lambda * ddi_pen

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            
            total += float(loss.item())
            n += 1
        torch.save(model, '../save_model/gamenet/model_best_gamenet.pth')

        print(f"[epoch {ep}] loss={total/max(n,1):.4f}")

# Batch Inference for Prediction Data

@torch.no_grad()
def get_probs_and_targets(model, df, vocabs, A_ehr_norm, A_ddi_norm, device,
                          batch_size=256, num_workers=0):
    ds = FlatVisitDataset(df, vocabs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_flat_visits(b, n_med=vocabs.n_med),
    )

    model.eval()
    model.to(device)

    probs_all = []
    y_all = []
    for batch in loader:
        logits = model(
            diag_ids=batch["diag_ids"].to(device),
            diag_mask=batch["diag_mask"].to(device),
            lengths=batch["lengths"].to(device),
            hist_med_mh=batch["hist_med_mh"].to(device),
            hist_mask=batch["hist_mask"].to(device),
            A_ehr_norm=A_ehr_norm.to(device),
            A_ddi_norm=A_ddi_norm.to(device),
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        y = batch["target_mh"].numpy()

        probs_all.append(probs)
        y_all.append(y)

    probs_all = np.concatenate(probs_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return probs_all, y_all


def tune_threshold_f1(probs_valid, y_valid, thr_grid=None):
    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 199)

    trueb = (y_valid > 0.5)
    best_thr, best_f1 = None, -1.0

    for thr in thr_grid:
        pred = (probs_valid >= thr)
        f1 = micro_f1(pred, trueb)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return {"best_threshold": best_thr, "best_f1": float(best_f1)}


def micro_f1(pred_bin: np.ndarray, true_bin: np.ndarray) -> float:
    tp = (pred_bin & true_bin).sum()
    fp = (pred_bin & ~true_bin).sum()
    fn = (~pred_bin & true_bin).sum()
    denom = (2*tp + fp + fn)
    return float((2*tp / denom) if denom > 0 else 1.0)


    order = np.argsort(-scores, kind="mergesort")
    y_true = y_true[order]

    tp_cum = np.cumsum(y_true)
    k = np.arange(1, y_true.size + 1)
    precision = tp_cum / k

    ap = (precision * y_true).sum() / pos
    return float(ap)
    
def evaluate_from_probs(
    probs: np.ndarray,
    true: np.ndarray,
    thr: float,
    A_ddi_dense: np.ndarray | None = None
) -> dict:
    trueb = (true > 0.5)
    pred = (probs >= thr)

    out = {
        "threshold": float(thr),
        "jaccard": jaccard(pred, trueb),
        "micro_f1": micro_f1(pred, trueb),
        "macro_f1": macro_f1(pred, trueb),
        "auprc_micro": auprc_micro(trueb, probs),   # threshold-free
        "avg_pred_size": float(pred.sum(axis=1).mean()),
        "avg_true_size": float(trueb.sum(axis=1).mean()),
    }
    if A_ddi_dense is not None:
        out["ddi_rate_major"] = ddi_rate(pred, A_ddi_dense)
    return out