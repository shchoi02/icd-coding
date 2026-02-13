import os
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def best_f1_per_class_sklearn(logits: torch.Tensor,
                              targets: torch.Tensor):
    """
    각 클래스별로 precision‑recall 곡선을 이용해
    F1 최댓값과 그때의 threshold 반환
    """
    # ─── 1. Tensor → NumPy ───────────────────────────────────────────
    y_true  = targets.detach().cpu().numpy()
    y_score = logits.detach().cpu().numpy()

    n_classes = y_true.shape[1]
    best_f1   = np.zeros(n_classes)
    best_thr  = np.zeros(n_classes)

    # ─── 2. 클래스별 PR‑Curve → F1 최적화 ─────────────────────────────
    for c in range(n_classes):
        precision, recall, thrs = precision_recall_curve(y_true[:, c],
                                                         y_score[:, c])
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        idx = np.argmax(f1)

        best_f1[c]  = f1[idx]
        # precision_recall_curve 는 thresholds 길이가 n‑1 ⇒
        # idx==len(thrs) 이면 threshold = 1.0 으로 간주
        best_thr[c] = thrs[idx] if idx < len(thrs) else 1.0

    return best_f1, best_thr


def f1_score_db_tuning(
    logits, targets, groups,
    average="micro",
    type="single",
    debug_viz=True,
    save_dir='/mnt/zvrs202/mixlab/mixlab/tabular/icd-coding/vis',
    json_path="/mnt/zvrs202/mixlab/mixlab/tabular/icd-coding/files/data/mimiciv_icd10/icd10_longtail_split.json",
    tag="val",              # 파일명 구분용
    min_pos_tail=5,         # Figure2에서 tail 후보 선택 기준
):
    device, dtype = logits.device, logits.dtype
    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")

    # ---------- optional: make output dir ----------
    if debug_viz and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # ---------- helper: save or show ----------
    def _finalize_fig(fname):
        if debug_viz:
            if save_dir is not None:
                path = os.path.join(save_dir, fname)
                plt.savefig(path, bbox_inches="tight")
                plt.close()
            else:
                plt.show()
        else:
            plt.close()
            
    def _summarize_thr_stats_from_groups(best_db, title_suffix="",
                                         save_stats_csv=True,
                                         make_boxplot=True):
        """
        best_db: torch.Tensor (C,)
        groups: dict with keys {"head","medium","tail"} and values=list of class indices
        targets: used for positives per class on validation
        """
        import numpy as np
        import pandas as pd

        if groups is None:
            raise ValueError("groups is required (head/medium/tail indices).")
        for k in ["head", "medium", "tail"]:
            if k not in groups:
                raise ValueError(f"groups must contain key='{k}'")

        # --- pos per class (validation) ---
        y_true = targets.detach().cpu().numpy().astype(np.int32)
        pos = y_true.sum(axis=0).astype(int)  # (C,)

        thr = best_db.detach().cpu().numpy().astype(np.float64)  # (C,)
        C = thr.shape[0]

        def _clean(idxs):
            idxs = np.asarray(idxs, dtype=np.int64)
            idxs = idxs[(idxs >= 0) & (idxs < C)]
            return idxs

        h_idx = _clean(groups["head"])
        m_idx = _clean(groups["medium"])
        t_idx = _clean(groups["tail"])

        def _stats(name, idxs):
            thr_g = thr[idxs]
            pos_g = pos[idxs]

            # labels that appear at least once in val
            mask = pos_g > 0
            thr_g2 = thr_g[mask]
            pos_g2 = pos_g[mask]

            if len(thr_g2) == 0:
                return {
                    "group": name,
                    "n_labels": int(len(idxs)),
                    "n_pos_labels": 0,
                    "pos_mean": np.nan, "pos_median": np.nan,
                    "thr_mean": np.nan, "thr_std": np.nan,
                    "thr_median": np.nan, "thr_iqr": np.nan,
                    "thr_q10": np.nan, "thr_q90": np.nan,
                }

            q10, q25, q75, q90 = np.percentile(thr_g2, [10, 25, 75, 90])

            return {
                "group": name,
                "n_labels": int(len(idxs)),
                "n_pos_labels": int(mask.sum()),
                "pos_mean": float(pos_g2.mean()),
                "pos_median": float(np.median(pos_g2)),
                "thr_mean": float(thr_g2.mean()),
                "thr_std": float(thr_g2.std(ddof=1) if len(thr_g2) > 1 else 0.0),
                "thr_median": float(np.median(thr_g2)),
                "thr_iqr": float(q75 - q25),
                "thr_q10": float(q10),
                "thr_q90": float(q90),
            }

        rows = [
            _stats("head", h_idx),
            _stats("medium", m_idx),
            _stats("tail", t_idx),
        ]
        df = pd.DataFrame(rows)

        print("\n[Per-class threshold dispersion stats]" + title_suffix)
        print(df.to_string(index=False))

        # --- CSV 저장 ---
        if debug_viz and save_dir is not None and save_stats_csv:
            out_csv = os.path.join(save_dir, f"{tag}_thr_stats_{type}.csv")
            df.to_csv(out_csv, index=False)
            print(f"💾 saved: {out_csv}")

        # --- (옵션) boxplot SVG: variance를 한 방에 보여줌 ---
        if debug_viz and make_boxplot:
            def _thr_pos(idxs):
                thr_g = thr[idxs]
                pos_g = pos[idxs]
                return thr_g[pos_g > 0]

            data = [_thr_pos(h_idx), _thr_pos(m_idx), _thr_pos(t_idx)]
            labels = ["Head", "Medium", "Tail"]

            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            ax.boxplot(
                data, labels=labels, showfliers=False,
                medianprops=dict(linewidth=2),
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
            )
            ax.set_ylabel("F1-optimal threshold")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, axis="y", ls="--", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            _finalize_fig(f"{tag}_fig_stats_thr_boxplot_{type}.svg")

        return df


    # ---------- grid thresholds ----------
    dbs = torch.linspace(0, 1, 100, device=device, dtype=dtype)
    n_cls = targets.size(1)

    tp = torch.zeros((len(dbs), targets.shape[1]), device=device, dtype=dtype)
    fp = torch.zeros((len(dbs), targets.shape[1]), device=device, dtype=dtype)
    fn = torch.zeros((len(dbs), targets.shape[1]), device=device, dtype=dtype)

    for idx, db in enumerate(dbs):
        predictions = (logits > db).to(dtype=torch.long)
        tp[idx] = torch.sum(predictions * targets, dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)

    if average == "micro":
        f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
    else:
        f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)

    # ---------- return: single ----------
    if type == "single":
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db

    # ---------- return: per_class ----------
    if type == "per_class":
        y_true = targets.detach().cpu().numpy()
        y_prob = logits.detach().cpu().numpy()
        best_f1, best_db = [], []
        for c in range(n_cls):
            p, r, th = precision_recall_curve(y_true[:, c], y_prob[:, c])
            f1 = 2 * p * r / (p + r + 1e-12)
            j = int(f1.argmax())
            best_f1.append(float(f1[j]))
            best_db.append(float(th[j]) if j < len(th) else 1.0)

        best_f1 = torch.tensor(best_f1, device=device, dtype=dtype)
        best_db = torch.tensor(best_db, device=device, dtype=dtype)

        # ✅ debug viz
        if debug_viz:
            _summarize_thr_stats_from_groups(best_db)

        return best_f1, best_db

    # ---------- return: per_class2 (sanity backoff) ----------
    if type == "per_class2":
        MIN_POS = 10
        MIN_PRED_POS = 1

        best_idx = int(f1_scores.argmax().item())
        thr_global = float(dbs[best_idx].item())

        y_true = targets.detach().cpu().numpy().astype(np.int32)
        y_prob = logits.detach().cpu().numpy().astype(np.float64)

        best_f1_list, best_db_list = [], []
        backed_off_lowpos = 0
        backed_off_degen = 0

        for c in range(n_cls):
            pos_cnt = int(y_true[:, c].sum())

            if pos_cnt < MIN_POS:
                best_f1_list.append(0.0)
                best_db_list.append(thr_global)
                backed_off_lowpos += 1
                continue

            p, r, th = precision_recall_curve(y_true[:, c], y_prob[:, c])
            f1 = 2 * p * r / (p + r + 1e-12)
            j = int(f1.argmax())
            thr_c = float(th[j]) if j < len(th) else 1.0

            pred_pos = int((y_prob[:, c] > thr_c).sum())
            if pred_pos < MIN_PRED_POS:
                best_f1_list.append(0.0)
                best_db_list.append(thr_global)
                backed_off_degen += 1
                continue

            best_f1_list.append(float(f1[j]))
            best_db_list.append(float(thr_c))

        best_f1_t = torch.tensor(best_f1_list, device=device, dtype=dtype)
        best_db_t = torch.tensor(best_db_list, device=device, dtype=dtype)

        print(f"[per_class sanity A] thr_global={thr_global:.4f} | "
              f"backed_off_lowpos={backed_off_lowpos}/{n_cls} | "
              f"backed_off_degen={backed_off_degen}/{n_cls}")

        return best_f1_t, best_db_t

    # ---------- return: per_group ----------
    if type == "per_group":
        thr_vec = torch.full((targets.shape[1],), 0.5, device=device, dtype=dtype)
        cls_f1 = tp / (tp + 0.5 * (fp + fn) + 1e-10)
        best_f1_g, best_db_g = {}, {}
        for g, idxs in groups.items():
            idxs = torch.as_tensor(idxs, device=device)
            if average == "micro":
                g_tp = tp[:, idxs].sum(1)
                g_fp = fp[:, idxs].sum(1)
                g_fn = fn[:, idxs].sum(1)
                g_f1 = g_tp / (g_tp + 0.5 * (g_fp + g_fn) + 1e-10)
            else:
                g_f1 = cls_f1[:, idxs].mean(1)
            best = int(g_f1.argmax().item())
            best_f1_g[g] = float(g_f1[best].item())
            best_db_g[g] = float(dbs[best].item())
            thr_vec[idxs.long()] = best_db_g[g]

        # (원하면) per-group도 Figure 1 같은 분포 그림을 그릴 수 있지만
        # 지금 목표가 per-class instability 설명이니까 생략해도 됨.

        return best_f1_g, thr_vec

    raise ValueError(f"Unknown type: {type}")