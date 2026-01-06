import torch 
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def _f1(precision, recall, eps=1e-8):
    return 2 * precision * recall / (precision + recall + eps)  

def _jaccard(tp, fp, fn, eps=1e-8):
    return tp / (tp + fp + fn + eps)

def _precision(tp, fp, eps=1e-8):
    return tp / (tp + fp + eps)

def _recall(tp, fn, eps=1e-8):
    return tp / (tp + fn + eps)

def _conf_metrics(y_pred, y_true):
    tp = (y_pred & y_true).sum().float()
    fp = (y_pred & (1 - y_true)).sum().float()
    fn = ((1 - y_pred) & y_true).sum().float()
    return tp, fp, fn

def _prauc(scores, y_true, eps=1e-8):
    _device = scores.device
    # ---- micro PR-AUC (flatten all labels) ----
    y = y_true.float()
    s = scores.float()

    pos = y.sum()

    order = torch.argsort(s, descending=True)
    y_sorted = y[order]

    y_sorted_cpu = y_sorted.cpu()
    tp_cum = torch.cumsum(y_sorted_cpu, dim=0)
    fp_cum = torch.cumsum(1 - y_sorted_cpu, dim=0)
    
    # move back if you need GPU tensors
    tp_cum = tp_cum.to(_device)
    fp_cum = fp_cum.to(_device)

    prec_curve = tp_cum / (tp_cum + fp_cum + eps)
    rec_curve = tp_cum / (pos + eps)

    rec_curve = torch.cat([torch.zeros(1, device=rec_curve.device), rec_curve])
    prec_curve = torch.cat([torch.ones(1, device=prec_curve.device), prec_curve])

    prauc_micro = torch.trapz(prec_curve, rec_curve)

    return prauc_micro

@torch.no_grad()
def _micro_ddi_rate(input_med, adj_ddi_matrix):
    input_med = input_med.cpu().numpy()
    all_cnt = 0
    ddi_cnt = 0
    for label_set in input_med:
        list_med = np.where(label_set)[0]
        for i, med_i in enumerate(list_med):
            for med_j in list_med[i+1:]:
                all_cnt += 1
                ddi_cnt += adj_ddi_matrix[med_i, med_j]
                
    ddi_rate = ddi_cnt/ all_cnt if all_cnt > 0 else 0
    
    return torch.tensor(ddi_rate)

@torch.no_grad()
def _macro_ddi_rate(input_med, adj_ddi_matrix):
    input_med = input_med.cpu().numpy()
    ddi_per_case = []
    for label_set in input_med:
        
        all_cnt = 0
        ddi_cnt = 0
        list_med = np.where(label_set)[0]
        for i, med_i in enumerate(list_med):
            for med_j in list_med[i+1:]:
                all_cnt += 1
                ddi_cnt += adj_ddi_matrix[med_i, med_j]
                
        ddi_per_case.append(ddi_cnt/ all_cnt if all_cnt > 0 else 0)        
    # ddi_rate = ddi_cnt/ all_cnt if all_cnt > 0 else 0
    
    return torch.tensor(np.array(ddi_per_case).mean())

@torch.no_grad()
def _evaluate_metrics(scores: torch.Tensor,
                                y_true: torch.Tensor,
                                major_adj_ddi_matrix,
                                moder_adj_ddi_matrix,
                                threshold: float = 0.5,
                                average: str = 'micro',
                                eps: float = 1e-8):
    """
    Returns scalar tensors: (f1_micro, jaccard_micro, prauc_micro, ddi_rate)
    """
    if scores.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: scores {scores.shape} vs y_true {y_true.shape}")

    y_true = y_true.to(dtype=torch.int64)
    y_pred = (scores >= threshold).to(dtype=torch.int64)
    _, n = y_pred.shape

    if average == 'micro':
        tp, fp, fn = _conf_metrics(y_pred, y_true)

        precision = _precision(tp, fp)
        recall = _recall(tp, fn)
        f1 = _f1(precision, recall, eps)
        jaccard = _jaccard(tp, fp, fn, eps)
        prauc = _prauc(scores.reshape(-1), y_true.reshape(-1),eps)

        # ---- DDI rate (from predicted meds) ----
        major_ddi = _micro_ddi_rate(y_pred, major_adj_ddi_matrix)
        moder_ddi = _micro_ddi_rate(y_pred, moder_adj_ddi_matrix)

        return f1, jaccard, prauc, major_ddi, moder_ddi

    elif average == 'macro':
        f1_per_class = []
        jaccard_per_class = []
        prauc_per_class = []
        for class_idx in range(n):
            y_true_i = y_true[:, class_idx]
            y_pred_i = y_pred[:, class_idx]
            scores_i = scores[:, class_idx]
            
            tp, fp, fn = _conf_metrics(y_pred_i, y_true_i)

            precision = _precision(tp, fp)
            recall = _recall(tp, fn)
            f1 = _f1(precision, recall, eps)
            jaccard = _jaccard(tp, fp, fn, eps)
            prauc = _prauc(scores_i, y_true_i, eps)
            
            f1_per_class.append(f1)
            jaccard_per_class.append(jaccard)
            prauc_per_class.append(prauc)

        
        f1 = torch.stack(f1_per_class).mean()
        jaccard = torch.stack(jaccard_per_class).mean()
        prauc = torch.stack(prauc_per_class).mean()

        major_ddi = _macro_ddi_rate(y_pred, major_adj_ddi_matrix)
        moder_ddi = _macro_ddi_rate(y_pred, moder_adj_ddi_matrix)

        return f1, jaccard, prauc, major_ddi, moder_ddi


    else:
        raise ValueError(f"Unknown average type: {average}")





@torch.no_grad()
def metrics_with_ci(scores: torch.Tensor,
                            y_true: torch.Tensor,
                            major_adj_ddi_matrix,
                            moder_adj_ddi_matrix,
                            threshold: float = 0.5,
                            n_bootstrap: int = 1000,
                            average: str = 'micro',
                            ci: float = 0.95,
                            eps: float = 1e-8,
                            use_tqdm: bool = True,
                            format_4dp: bool = False) -> pd.DataFrame:

    device = scores.device
    B = scores.size(0)
    if B < 2:
        raise ValueError("Need at least 2 samples (batch rows) for bootstrap CI.")

    # point estimates
    f1, jac, pr, major, moder = _evaluate_metrics(
        scores, y_true, major_adj_ddi_matrix, moder_adj_ddi_matrix, threshold, average, eps
    )

    boot = {"f1": [], "jaccard": [], "prauc": [], "major_ddi": [], "moder_ddi": []}
    alpha = 1.0 - ci

    iterator = range(n_bootstrap)
    if use_tqdm:
        iterator = tqdm(iterator, desc="Bootstrapping (+DDI)", leave=False)

    for _ in iterator:
        idx = torch.randint(0, B, (B,), device=device)
        bf1, bjac, bpr, bma, bmo = _evaluate_metrics(
            scores[idx], y_true[idx], major_adj_ddi_matrix, moder_adj_ddi_matrix, threshold, average, eps
        )
        
        boot["f1"].append(bf1)
        boot["jaccard"].append(bjac)
        boot["prauc"].append(bpr)
        boot["major_ddi"].append(bma)
        boot["moder_ddi"].append(bmo)


    # stack
    for k in boot:
        boot[k] = torch.stack(boot[k])

    def _ci(x):
        lo = torch.quantile(x, alpha / 2).item()
        hi = torch.quantile(x, 1 - alpha / 2).item()
        return lo, hi

    rows = []
    for name, val in [("f1", f1), ("jaccard", jac), ("prauc", pr), ("major_ddi", major), ("moder_ddi", moder)]:
        lo, hi = _ci(boot[name])
        rows.append([name, val.item(), lo, hi])

    df = pd.DataFrame(rows, columns=["metric", "value", "ci_low", "ci_high"])

    if format_4dp:
        if torch.is_tensor(threshold):
            method_thr = 'local'
        else:
            method_thr = 'global'
        df_result = (
                df.assign(
                    val_ci=lambda x: x.apply(
                        lambda r: f"{r['value']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]",
                        axis=1
                    )
                )
                .set_index("metric")["val_ci"]
                .to_frame().T
            )
        df_result.index = [f'{method_thr}']
        
    return df_result


@torch.no_grad()
def _topk_metrics(scores: torch.Tensor, y_true: torch.Tensor, k: int, eps: float = 1e-8):
    """
    Calculate precision, recall, f1, and hitrate at top-k predictions.
    
    Args:
        scores: (B, N) prediction scores
        y_true: (B, N) binary ground truth labels
        k: number of top predictions to consider
        eps: small constant for numerical stability
    
    Returns:
        precision_k, recall_k, f1_k, hitrate_k: scalar tensors
    """
    _precision_at_k, N = scores.shape
    k = min(k, N)

    # Get top-k indices
    topk_idx = scores.topk(k, dim=1).indices  # (B, k)
    
    # Create binary prediction matrix
    y_pred = torch.zeros_like(scores)
    y_pred.scatter_(1, topk_idx, 1)  # Set top-k positions to 1
    
    # Per-sample metrics
    tp_per_sample = (y_pred * y_true).sum(dim=1)  # (B,)
    fp_per_sample = (y_pred * (1 - y_true)).sum(dim=1)  # (B,)
    
    # Precision at k (per sample)
    precision_per_sample = tp_per_sample / (tp_per_sample + fp_per_sample + eps)
    
    # Recall at k (per sample)
    num_positives_per_sample = y_true.sum(dim=1)
    recall_per_sample = tp_per_sample / (num_positives_per_sample + eps)
    
    # Hit rate at k (per sample): 1 if at least one relevant item in top-k, else 0
    hitrate_per_sample = (tp_per_sample > 0).float()
    
    # Average across samples
    precision_k = precision_per_sample.mean()
    recall_k = recall_per_sample.mean()
    f1_k = 2 * precision_k * recall_k / (precision_k + recall_k + eps)
    hitrate_k = hitrate_per_sample.mean()
    
    return precision_k, recall_k, f1_k, hitrate_k


@torch.no_grad()
def topk_all_ddi(scores: torch.Tensor,
                  y_true: torch.Tensor,
                  major_adj_ddi_matrix,
                  moder_adj_ddi_matrix,
                  k_values: list = [1, 3, 5],
                  eps: float = 1e-8):

    rows = []

    for k in k_values:
        # calculate top-k metrics
        prec_k, recall_k, f1_k, hitrate_k = _topk_metrics(scores, y_true, k, eps)
        # calculate DDI rates
        topk_idx = scores.topk(k, dim=1).indices
        y_pred = torch.zeros_like(scores, dtype=torch.int64)
        y_pred.scatter_(1, topk_idx, 1)
        major_ddi = _micro_ddi_rate(y_pred, major_adj_ddi_matrix)
        moder_ddi = _micro_ddi_rate(y_pred, moder_adj_ddi_matrix)
        # append to rows
        rows.append({
            'k': k,
            'precision': prec_k.item(),
            'recall': recall_k.item(),
            'f1': f1_k.item(),
            'hitrate': hitrate_k.item(),
            'major_ddi': major_ddi.item(),
            'moder_ddi': moder_ddi.item()
        })
    # convert to dataframe
    df = pd.DataFrame(rows)

    return df