import numpy as np
import pandas as pd 
from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# Decoder token scheme
PAD_ID = 0
START_ID = 1
MED_OFFSET = 2 

def read_df(filename):
    df = pd.read_pickle(filename)
    df = df[df.columns[:-2]]
    prefixes = ("has_doctor", "note_token", "date_visit", "clinic", "hash_hn", "hash_doctor", "sex", "dep", "dep_name")
    df = df.loc[:, ~df.columns.str.startswith(prefixes)]
    suffixes = ("_4", "_5", "_6")
    df = df.loc[:, ~df.columns.str.endswith(suffixes)]
    return df
    
# Parsing helpers
def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))

def parse_code_list(x: Any) -> List[str]:
    """Accept python list/tuple; parse stringified lists; fallback split; NaN->[]"""
    if _is_nan(x):
        return []
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x if not _is_nan(t)]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple)):
                    return [str(t) for t in v if not _is_nan(t)]
            except Exception:
                pass
        for sep in ["|", ";", ","]:
            if sep in s:
                return [p.strip() for p in s.split(sep) if p.strip()]
        return [s]
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
    diag2id: Dict[str, int]    # ICD10 -> id (0 is PAD)
    id2diag: List[str]
    med2id: Dict[str, int]     # ATC4 -> id (0..V-1)
    id2med: List[str]

    @property
    def n_diag(self) -> int:
        return len(self.id2diag)

    @property
    def n_med(self) -> int:
        return len(self.id2med)

def build_vocabs(
    df_train: pd.DataFrame,
    atcmap: Optional[Dict[int, str]] = None,
    use_atcmap_for_meds: bool = True,
    ) -> Vocabs:
    """
    Recommended: use_atcmap_for_meds=True with provided atcmap
      => fixed label space aligned to A_ddi (no label shrink).
    """
    K = infer_max_history(df_train, prefix="y_")

    # ICD10 from current + history icd10_k
    all_diags = []
    all_diags.extend(df_train["icd10"].apply(parse_code_list).tolist())
    for k in range(1, K + 1):
        dk = f"icd10_{k}"
        if dk in df_train.columns:
            all_diags.extend(df_train[dk].apply(parse_code_list).tolist())

    diag_set = sorted({d for row in all_diags for d in row})
    id2diag = ["<PAD>"] + diag_set
    diag2id = {d: i for i, d in enumerate(id2diag)}

    if use_atcmap_for_meds and atcmap is not None:
        n = len(atcmap)
        id2med = [atcmap[i] for i in range(n)]
    else:
        # meds only from current y to avoid double-counting through y_k
        all_meds = df_train["y"].apply(parse_code_list).tolist()
        med_set = sorted({m for row in all_meds for m in row})
        id2med = med_set

    med2id = {m: i for i, m in enumerate(id2med)}
    return Vocabs(diag2id=diag2id, id2diag=id2diag, med2id=med2id, id2med=id2med)

# Graphs: EHR co-prescription + DDI-major
def build_ehr_coprescription_adj_sparse(
    df_train: pd.DataFrame,
    med2id: Dict[str, int],
    y_col: str = "y",
    add_self_loops: bool = True,
    device: Optional[torch.device] = None,
    ) -> torch.Tensor:
    """
    EHR graph from df_train[y] only (each visit counted once).
    """
    V = len(med2id)
    edges = set()

    for meds in df_train[y_col].apply(parse_code_list).tolist():
        ids = [med2id[m] for m in meds if m in med2id]
        ids = sorted(set(ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                edges.add((a, b))
                edges.add((b, a))

    if add_self_loops:
        for i in range(V):
            edges.add((i, i))

    if len(edges) == 0:
        idx = torch.zeros((2, 0), dtype=torch.long)
        val = torch.zeros((0,), dtype=torch.float32)
    else:
        r, c = zip(*edges)
        idx = torch.tensor([r, c], dtype=torch.long)
        val = torch.ones((idx.shape[1],), dtype=torch.float32)

    A = torch.sparse_coo_tensor(idx, val, size=(V, V)).coalesce()
    return A.to(device) if device is not None else A

def add_self_loops_sparse(A: torch.Tensor) -> torch.Tensor:
    A = A.coalesce()
    V = A.size(0)
    idx = torch.arange(V, dtype=torch.long, device=A.device)
    I = torch.sparse_coo_tensor(
        torch.stack([idx, idx], dim=0),
        torch.ones(V, dtype=torch.float32, device=A.device),
        size=(V, V),
    ).coalesce()
    return (A + I).coalesce()

def normalize_sparse_adj(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetric norm: D^{-1/2} A D^{-1/2}
    """
    A = A.coalesce()
    idx = A.indices()
    val = A.values()
    V = A.size(0)

    deg = torch.zeros((V,), device=val.device, dtype=val.dtype)
    deg.scatter_add_(0, idx[0], val)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)

    val_norm = deg_inv_sqrt[idx[0]] * val * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, val_norm, size=A.size(), device=A.device).coalesce()

def ddi_matrix_to_sparse(
    A_ddi_np: np.ndarray,
    device: Optional[torch.device] = None,
    ) -> torch.Tensor:
    """
    Convert dense numpy (V,V) adjacency to torch sparse COO (binary).
    """
    A = (A_ddi_np > 0).astype(np.int64)
    r, c = np.nonzero(A)
    idx = torch.tensor([r, c], dtype=torch.long)
    val = torch.ones((idx.shape[1],), dtype=torch.float32)
    V = A.shape[0]
    T = torch.sparse_coo_tensor(idx, val, size=(V, V)).coalesce()
    if device is not None:
        T = T.to(device)
    return T

def ddi_matrix_atcmap_to_model_sparse(
    A_ddi_np: np.ndarray,
    atcmap: Dict[int, str],
    med2id: Dict[str, int],
    device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remap A_ddi (atcmap order) into model med vocab order.
    Returns (A_raw_sparse, A_norm_sparse(+selfloops)).
    If you used vocab from atcmap, model order == atcmap order => mapping is identity.
    """
    V = len(med2id)
    n = A_ddi_np.shape[0]
    idx2code = [atcmap[i] for i in range(n)]

    r, c = np.nonzero(A_ddi_np > 0)
    rows, cols = [], []
    for i, j in zip(r.tolist(), c.tolist()):
        ci = idx2code[i]
        cj = idx2code[j]
        if (ci in med2id) and (cj in med2id) and (ci != cj):
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
        val = torch.ones((len(rows),), dtype=torch.float32, device=device)
        A_raw = torch.sparse_coo_tensor(idx, val, size=(V, V), device=device).coalesce()

    # force symmetric binary
    A_raw = (A_raw + A_raw.transpose(0, 1)).coalesce()
    A_raw = torch.sparse_coo_tensor(
        A_raw.indices(),
        torch.ones_like(A_raw.values()),
        A_raw.size(),
        device=device
    ).coalesce()

    A_norm = normalize_sparse_adj(add_self_loops_sparse(A_raw))
    return A_raw, A_norm

def build_med_rank_rare_first(df_train: pd.DataFrame, vocabs: Vocabs) -> Dict[int, int]:
    """
    Rare-first ordering by training frequency (ascending).
    """
    cnt = Counter()
    for meds in df_train["y"].apply(parse_code_list).tolist():
        for m in meds:
            if m in vocabs.med2id:
                cnt[vocabs.med2id[m]] += 1
    items = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))  # rare first
    return {mid: r for r, (mid, _) in enumerate(items)}

# cognet dataset
class COGNetFlatDataset(Dataset):
    """
    For each row:
      - visits_diags: history visits ICD10 ids + current visit ICD10 ids
      - hist_meds: history visit meds ids (aligned with history visits)
      - dec_in/dec_out: teacher forcing sequences (START + meds + PAD...)
      - target_mh: multi-hot meds of current visit (for eval)
    """
    def __init__(self, df: pd.DataFrame, vocabs: Vocabs, med_rank: Dict[int, int], max_len: int = 45):
        self.df = df.reset_index(drop=True)
        self.v = vocabs
        self.K = infer_max_history(self.df, prefix="y_")
        self.max_len = int(max_len)
        self.med_rank = med_rank
        self.rank_default = (max(med_rank.values()) + 1) if len(med_rank) else 10**9

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.df.iloc[i]

        hist_diags: List[List[int]] = []
        hist_meds: List[List[int]] = []

        # history: older -> more recent (y_K ... y_1)
        for k in range(self.K, 0, -1):
            yk = f"y_{k}"
            dk = f"icd10_{k}"
            if (yk not in self.df.columns) or (dk not in self.df.columns):
                continue

            meds = parse_code_list(row[yk])
            diags = parse_code_list(row[dk])
            if len(meds) == 0 and len(diags) == 0:
                continue

            med_ids = [self.v.med2id[m] for m in meds if m in self.v.med2id]
            diag_ids = [self.v.diag2id[d] for d in diags if d in self.v.diag2id]
            hist_meds.append(med_ids)
            hist_diags.append(diag_ids)

        # current
        cur_diag_ids = [self.v.diag2id[d] for d in parse_code_list(row["icd10"]) if d in self.v.diag2id]
        cur_med_ids  = [self.v.med2id[m] for m in parse_code_list(row["y"]) if m in self.v.med2id]

        visits_diags = hist_diags + [cur_diag_ids]

        # rare-first ordering for sequence
        uniq = sorted(set(cur_med_ids), key=lambda mid: (self.med_rank.get(mid, self.rank_default), mid))
        meds_tok = [mid + MED_OFFSET for mid in uniq[: self.max_len]]
        seq = [START_ID] + meds_tok + [PAD_ID] * (self.max_len - len(meds_tok))
        dec_in = seq[:-1]   # length max_len
        dec_out = seq[1:]   # length max_len

        target_mh = np.zeros((self.v.n_med,), dtype=np.float32)
        if len(cur_med_ids) > 0:
            target_mh[list(set(cur_med_ids))] = 1.0

        return {
            "visits_diags": visits_diags,
            "hist_meds": hist_meds,
            "dec_in": np.array(dec_in, dtype=np.int64),
            "dec_out": np.array(dec_out, dtype=np.int64),
            "target_mh": target_mh,
        }

def collate_cognet(batch: List[Dict[str, Any]], n_med: int, max_len: int) -> Dict[str, torch.Tensor]:
    B = len(batch)
    T = max(len(x["visits_diags"]) for x in batch)
    L = 1
    for x in batch:
        for v in x["visits_diags"]:
            L = max(L, len(v))

    diag_ids  = torch.zeros((B, T, L), dtype=torch.long)     # 0 is PAD
    diag_mask = torch.zeros((B, T, L), dtype=torch.bool)
    lengths   = torch.zeros((B,), dtype=torch.long)

    H = max(T - 1, 1)  # history slots
    hist_visit_mask = torch.zeros((B, H), dtype=torch.bool)

    # Flatten all history med instances for hierarchical copy
    hist_counts = [sum(len(v) for v in x["hist_meds"]) for x in batch]
    Nmax = max(max(hist_counts), 1)

    hist_tok  = torch.zeros((B, Nmax), dtype=torch.long)   # token ids (PAD or MED_OFFSET+mid)
    hist_vidx = torch.zeros((B, Nmax), dtype=torch.long)   # visit slot [0..H-1]
    hist_mask = torch.zeros((B, Nmax), dtype=torch.bool)

    dec_in  = torch.zeros((B, max_len), dtype=torch.long)
    dec_out = torch.zeros((B, max_len), dtype=torch.long)
    target_mh = torch.zeros((B, n_med), dtype=torch.float32)

    for i, x in enumerate(batch):
        visits = x["visits_diags"]
        lengths[i] = len(visits)

        # diag padding
        for t in range(len(visits)):
            codes = visits[t]
            if len(codes) > 0:
                diag_ids[i, t, :len(codes)] = torch.tensor(codes, dtype=torch.long)
                diag_mask[i, t, :len(codes)] = True

        # history visit mask
        hlen = max(len(visits) - 1, 0)
        if hlen > 0:
            hist_visit_mask[i, :hlen] = True

        # flatten hist meds with visit indices
        ptr = 0
        for vj, meds in enumerate(x["hist_meds"][:H]):
            for mid in meds:
                if ptr >= Nmax:
                    break
                hist_tok[i, ptr] = int(mid) + MED_OFFSET
                hist_vidx[i, ptr] = int(vj)
                hist_mask[i, ptr] = True
                ptr += 1

        dec_in[i] = torch.from_numpy(x["dec_in"])
        dec_out[i] = torch.from_numpy(x["dec_out"])
        target_mh[i] = torch.from_numpy(x["target_mh"])

    return {
        "diag_ids": diag_ids,
        "diag_mask": diag_mask,
        "lengths": lengths,
        "hist_visit_mask": hist_visit_mask,
        "hist_tok": hist_tok,
        "hist_vidx": hist_vidx,
        "hist_mask": hist_mask,
        "dec_in": dec_in,
        "dec_out": dec_out,
        "target_mh": target_mh,
    }

# COGNet Model

def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    scores: [..., K]
    mask:   [..., K] bool
    """
    if scores.numel() == 0:
        return scores
    very_neg = torch.finfo(scores.dtype).min
    s2 = scores.masked_fill(~mask, very_neg)
    has_any = mask.any(dim=dim, keepdim=True)
    p = torch.softmax(s2, dim=dim)
    return torch.where(has_any, p, torch.zeros_like(p))

class SparseGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, A_norm: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        H = torch.sparse.mm(A_norm, X)
        H = F.relu(self.W1(H))
        H = torch.sparse.mm(A_norm, H)
        return self.W2(H)

class COGNet(nn.Module):
    """
    Copy-or-predict with:
      - Visit encoder (diag attention)
      - Medication relation embeddings via GNN over EHR + DDI graphs
      - Hierarchical copy: visit-level attention * med-instance attention
      - Autoregressive decoder with teacher forcing
    """
    def __init__(
        self,
        n_diag: int,
        n_med: int,
        emb_dim: int = 64,
        gcn_hidden: int = 64,
        max_len: int = 45,
    ):
        super().__init__()
        self.n_med = n_med
        self.emb_dim = emb_dim
        self.max_len = int(max_len)
        self.n_token = n_med + MED_OFFSET  # PAD, START, meds...

        self.diag_emb = nn.Embedding(n_diag, emb_dim, padding_idx=0)
        self.med_emb = nn.Embedding(n_med, emb_dim)

        # diag attention within a visit
        self.diag_att1 = nn.Linear(emb_dim, emb_dim)
        self.diag_att2 = nn.Linear(emb_dim, 1)

        # GNN for relation-aware med embeddings
        self.gcn_ehr = SparseGCN(emb_dim, gcn_hidden, emb_dim)
        self.gcn_ddi = SparseGCN(emb_dim, gcn_hidden, emb_dim)
        self.beta_logit = nn.Parameter(torch.tensor(0.0))  # fuse ehr vs ddi

        # decoder
        self.h0_fc = nn.Linear(emb_dim, emb_dim)
        self.dec_cell = nn.GRUCell(emb_dim, emb_dim)

        self.start_emb = nn.Parameter(torch.randn(emb_dim) * 0.02)

        # generation head + copy head + gate
        self.gen_fc = nn.Linear(emb_dim, self.n_token)
        self.copy_q = nn.Linear(emb_dim, emb_dim)
        self.gate_fc = nn.Linear(emb_dim, 1)

    def build_relation_emb(self, A_ehr_norm: torch.Tensor, A_ddi_norm: torch.Tensor) -> torch.Tensor:
        X0 = self.med_emb.weight  # (V,d)
        E_ehr = self.gcn_ehr(A_ehr_norm, X0)
        E_ddi = self.gcn_ddi(A_ddi_norm, X0)
        beta = torch.sigmoid(self.beta_logit)
        return E_ehr - beta * E_ddi  # (V,d)

    def token_embed(self, tok: torch.Tensor, E_g: torch.Tensor) -> torch.Tensor:
        """
        tok: [B] in {0=PAD, 1=START, >=2 meds}
        """
        B = tok.size(0)
        d = self.emb_dim
        out = torch.zeros((B, d), device=tok.device, dtype=torch.float32)

        is_start = (tok == START_ID)
        if is_start.any():
            out[is_start] = self.start_emb.unsqueeze(0).expand(int(is_start.sum()), d)

        is_med = (tok >= MED_OFFSET)
        if is_med.any():
            mid = (tok[is_med] - MED_OFFSET).clamp(min=0, max=self.n_med - 1)
            out[is_med] = self.med_emb(mid) + E_g[mid]
        return out

    def encode_visits(self, diag_ids: torch.Tensor, diag_mask: torch.Tensor) -> torch.Tensor:
        """
        diag_ids: [B,T,L], diag_mask: [B,T,L] -> visit vectors [B,T,d]
        """
        E = self.diag_emb(diag_ids)  # [B,T,L,d]
        G = torch.tanh(self.diag_att1(E))          # [B,T,L,d]
        S = self.diag_att2(G).squeeze(-1)          # [B,T,L]
        S = S.masked_fill(~diag_mask, torch.finfo(S.dtype).min)
        alpha = torch.softmax(S, dim=-1)           # [B,T,L]
        alpha = torch.where(diag_mask.any(dim=-1, keepdim=True), alpha, torch.zeros_like(alpha))
        v = torch.einsum("btl,btld->btd", alpha, E)  # [B,T,d]
        return v

    def visit_level_weights(
        self,
        v_all: torch.Tensor,          # [B,T,d]
        lengths: torch.Tensor,        # [B]
        hist_visit_mask: torch.Tensor # [B,H]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns v_cur [B,d], c_visit [B,H]
        """
        B, T, d = v_all.shape
        H = hist_visit_mask.size(1)
        idx = (lengths - 1).clamp(min=0)
        v_cur = v_all[torch.arange(B, device=v_all.device), idx]  # [B,d]
        v_hist = v_all[:, :H, :]                                   # [B,H,d]
        scores = (v_hist * v_cur.unsqueeze(1)).sum(-1) / (d ** 0.5) # [B,H]
        c_visit = masked_softmax(scores, hist_visit_mask, dim=1)
        return v_cur, c_visit

    def copy_distribution(
        self,
        h: torch.Tensor,         # [B,d]
        E_g: torch.Tensor,       # [V,d]
        c_visit: torch.Tensor,   # [B,H]
        hist_tok: torch.Tensor,  # [B,N] token ids (PAD or MED_OFFSET+mid)
        hist_vidx: torch.Tensor, # [B,N] visit slot
        hist_mask: torch.Tensor  # [B,N] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns p_c [B,n_token], copy_mass [B,1]
        """
        B, d = h.shape
        N = hist_tok.size(1)

        is_med = (hist_tok >= MED_OFFSET) & hist_mask
        mid = (hist_tok - MED_OFFSET).clamp(min=0, max=self.n_med - 1)

        hist_emb = torch.zeros((B, N, d), device=h.device, dtype=torch.float32)
        if is_med.any():
            hist_emb[is_med] = self.med_emb(mid[is_med]) + E_g[mid[is_med]]

        # med-instance attention q
        qhat = torch.einsum("bd,bnd->bn", self.copy_q(h), hist_emb) / (d ** 0.5)  # [B,N]
        q = masked_softmax(qhat, hist_mask, dim=1)                                # [B,N]

        # hierarchical: multiply by visit-level weight
        vidx = hist_vidx.clamp(min=0, max=c_visit.size(1) - 1)
        c_inst = c_visit.gather(1, vidx)                 # [B,N]
        w = q * c_inst * hist_mask.float()               # [B,N]

        p_hat = torch.zeros((B, self.n_token), device=h.device, dtype=torch.float32)
        p_hat.scatter_add_(1, hist_tok, w)

        # no copy to PAD/START
        p_hat[:, PAD_ID] = 0.0
        p_hat[:, START_ID] = 0.0

        mass = p_hat.sum(dim=1, keepdim=True)
        p_c = torch.where(mass > 0, p_hat / (mass + 1e-12), torch.zeros_like(p_hat))
        return p_c, mass

    def forward(
        self,
        diag_ids: torch.Tensor,
        diag_mask: torch.Tensor,
        lengths: torch.Tensor,
        hist_visit_mask: torch.Tensor,
        hist_tok: torch.Tensor,
        hist_vidx: torch.Tensor,
        hist_mask: torch.Tensor,
        dec_in: torch.Tensor,     # [B,max_len]
        dec_out: torch.Tensor,    # [B,max_len]
        A_ehr_norm: torch.Tensor,
        A_ddi_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher forcing NLL on mixed distribution.
        """
        B = diag_ids.size(0)
        d = self.emb_dim

        v_all = self.encode_visits(diag_ids, diag_mask)             # [B,T,d]
        v_cur, c_visit = self.visit_level_weights(v_all, lengths, hist_visit_mask)

        E_g = self.build_relation_emb(A_ehr_norm, A_ddi_norm)       # [V,d]
        h = torch.tanh(self.h0_fc(v_cur))                           # [B,d]

        loss = 0.0
        for t in range(self.max_len):
            x = self.token_embed(dec_in[:, t], E_g)                 # [B,d]
            h = self.dec_cell(x, h)                                 # [B,d]

            # generation
            logits = self.gen_fc(h)                                 # [B,n_token]
            logits[:, START_ID] = -1e9                              # never generate START
            p_g = torch.softmax(logits, dim=-1)

            # copy
            p_c, copy_mass = self.copy_distribution(h, E_g, c_visit, hist_tok, hist_vidx, hist_mask)

            # mixture gate (if no copy available -> force generate)
            w_g = torch.sigmoid(self.gate_fc(h))                    # [B,1]
            w_g = torch.where(copy_mass > 0, w_g, torch.ones_like(w_g))

            p = w_g * p_g + (1.0 - w_g) * p_c                       # [B,n_token]

            tgt = dec_out[:, t].clamp(min=0, max=self.n_token - 1)
            p_tgt = p.gather(1, tgt.unsqueeze(1)).squeeze(1).clamp(min=1e-12)
            loss = loss + (-torch.log(p_tgt)).mean()

        return loss / float(self.max_len)

    @torch.no_grad()
    def greedy_scores_and_set(
        self,
        diag_ids: torch.Tensor,
        diag_mask: torch.Tensor,
        lengths: torch.Tensor,
        hist_visit_mask: torch.Tensor,
        hist_tok: torch.Tensor,
        hist_vidx: torch.Tensor,
        hist_mask: torch.Tensor,
        A_ehr_norm: torch.Tensor,
        A_ddi_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          scores [B,V] : per-med probability scores (good for AUPRC + thresholding)
          pred_mh[B,V] : greedy predicted set (no thresholding needed; used if you want)
        Scoring rule:
          - average p(med) across decode steps for all meds
          - for meds actually selected, use the probability at the step it was first selected
        """
        B = diag_ids.size(0)
        V = self.n_med
        d = self.emb_dim

        v_all = self.encode_visits(diag_ids, diag_mask)
        v_cur, c_visit = self.visit_level_weights(v_all, lengths, hist_visit_mask)
        E_g = self.build_relation_emb(A_ehr_norm, A_ddi_norm)

        h = torch.tanh(self.h0_fc(v_cur))
        prev_tok = torch.full((B,), START_ID, dtype=torch.long, device=diag_ids.device)

        avg_sum = torch.zeros((B, V), device=diag_ids.device, dtype=torch.float32)
        chosen_prob = torch.full((B, V), -1.0, device=diag_ids.device, dtype=torch.float32)
        chosen_mask = torch.zeros((B, V), device=diag_ids.device, dtype=torch.bool)

        pred_mh = torch.zeros((B, V), device=diag_ids.device, dtype=torch.float32)
        selected = torch.zeros((B, V), device=diag_ids.device, dtype=torch.bool)
        done = torch.zeros((B,), device=diag_ids.device, dtype=torch.bool)

        for t in range(self.max_len):
            x = self.token_embed(prev_tok, E_g)
            h = self.dec_cell(x, h)

            logits = self.gen_fc(h)
            logits[:, START_ID] = -1e9
            p_g = torch.softmax(logits, dim=-1)

            p_c, copy_mass = self.copy_distribution(h, E_g, c_visit, hist_tok, hist_vidx, hist_mask)
            w_g = torch.sigmoid(self.gate_fc(h))
            w_g = torch.where(copy_mass > 0, w_g, torch.ones_like(w_g))
            p = w_g * p_g + (1.0 - w_g) * p_c  # [B,n_token]

            p_med = p[:, MED_OFFSET:]          # [B,V]
            avg_sum += p_med

            # avoid duplicates
            p_sel = p.clone()
            p_sel[:, START_ID] = 0.0
            p_sel[:, MED_OFFSET:] = p_sel[:, MED_OFFSET:] * (~selected).float()

            # if already done -> force PAD
            if done.any():
                p_sel[done] = 0.0
                p_sel[done, PAD_ID] = 1.0

            tok = torch.argmax(p_sel, dim=-1)
            newly_done = (tok == PAD_ID)
            done = done | newly_done

            is_med = (tok >= MED_OFFSET) & (~done)
            if is_med.any():
                mid = (tok[is_med] - MED_OFFSET).clamp(min=0, max=V - 1)
                pred_mh[is_med, mid] = 1.0

                # store first-step probability
                b_idx = torch.where(is_med)[0]
                already = selected[is_med, mid]
                if (~already).any():
                    nb = b_idx[~already]
                    nm = mid[~already]
                    chosen_prob[nb, nm] = p_med[nb, nm]
                    chosen_mask[nb, nm] = True

                selected[is_med, mid] = True

            prev_tok = tok

        avg = avg_sum / float(self.max_len)
        scores = avg.clone()
        scores[chosen_mask] = chosen_prob[chosen_mask]
        scores = scores.clamp(0.0, 1.0)
        return scores, pred_mh

# Train + batch inference
def train_cognet(
    model: COGNet,
    train_loader: DataLoader,
    A_ehr_norm: torch.Tensor,
    A_ddi_norm: torch.Tensor,
    device: torch.device,
    lr: float = 1e-4,
    epochs: int = 2,
    grad_clip: float = 5.0,
    ):
    model.to(device)
    A_ehr_norm = A_ehr_norm.to(device)
    A_ddi_norm = A_ddi_norm.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    total_batches = len(train_loader)
    if total_batches <= 0:
        raise ValueError("train_loader is empty")

    # log every 5% of an epoch (at least 1 batch)
    log_every = max(1, int(round(total_batches * 0.01)))
    
    print(f'Training started; Log every: {log_every} steps')
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        
        for bi, batch in enumerate(train_loader, start=1):
            # move tensors
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            loss = model(
                diag_ids=batch["diag_ids"],
                diag_mask=batch["diag_mask"],
                lengths=batch["lengths"],
                hist_visit_mask=batch["hist_visit_mask"],
                hist_tok=batch["hist_tok"],
                hist_vidx=batch["hist_vidx"],
                hist_mask=batch["hist_mask"],
                dec_in=batch["dec_in"],
                dec_out=batch["dec_out"],
                A_ehr_norm=A_ehr_norm,
                A_ddi_norm=A_ddi_norm,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            total += float(loss.item())
            n += 1

            if (bi % log_every == 0) or (bi == total_batches):
                pct = (bi / total_batches) * 100.0
                avg = total / max(n, 1)
                print(f"[epoch {ep}] {pct:5.1f}% ({bi}/{total_batches}) avg_loss={avg:.4f} last_loss={float(loss.item()):.4f}")
                
        torch.save(model, '../save_model/cognet/model_best_cognet.pth')
        print(f"[epoch {ep}] loss={total/max(n,1):.4f}")

# Metrics + threshold tuning
def micro_f1(pred_bin: np.ndarray, true_bin: np.ndarray) -> float:
    tp = (pred_bin & true_bin).sum()
    fp = (pred_bin & ~true_bin).sum()
    fn = (~pred_bin & true_bin).sum()
    denom = (2*tp + fp + fn)
    return float((2*tp / denom) if denom > 0 else 1.0)

def tune_threshold_micro_f1(probs_valid: np.ndarray, y_valid: np.ndarray, thr_grid=None) -> Dict[str, float]:
    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 99)
    trueb = (y_valid > 0.5)
    best_thr, best_f1 = None, -1.0
    for thr in thr_grid:
        pred = (probs_valid >= thr)
        f1 = micro_f1(pred, trueb)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return {"best_threshold": float(best_thr), "best_micro_f1": float(best_f1)}

@torch.no_grad()
def get_scores_and_targets(
    model: COGNet,
    df: pd.DataFrame,
    vocabs: Vocabs,
    med_rank: Dict[int, int],
    A_ehr_norm: torch.Tensor,
    A_ddi_norm: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
    max_len: int = 45,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = COGNetFlatDataset(df, vocabs, med_rank=med_rank, max_len=max_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False, 
        #prefetch_factor=1,
        #persistent_workers=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_cognet(b, n_med=vocabs.n_med, max_len=max_len),
    )
    model.eval().to(device)
    A_ehr_norm = A_ehr_norm.to(device)
    A_ddi_norm = A_ddi_norm.to(device)

    scores_all, y_all = [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

        scores, _ = model.greedy_scores_and_set(
            diag_ids=batch["diag_ids"],
            diag_mask=batch["diag_mask"],
            lengths=batch["lengths"],
            hist_visit_mask=batch["hist_visit_mask"],
            hist_tok=batch["hist_tok"],
            hist_vidx=batch["hist_vidx"],
            hist_mask=batch["hist_mask"],
            A_ehr_norm=A_ehr_norm,
            A_ddi_norm=A_ddi_norm,
        )
        scores_all.append(scores.cpu().numpy())
        y_all.append(batch["target_mh"].cpu().numpy())

    return np.concatenate(scores_all, axis=0), np.concatenate(y_all, axis=0)

