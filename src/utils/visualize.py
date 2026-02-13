import json, numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import ceil
from typing import Optional
from matplotlib.patches import Patch
import pandas as pd

@torch.no_grad()
def plot_per_class_best_threshold(
    logits: torch.Tensor,           # (N, C) – GPU tensor
    targets: torch.Tensor,          # (N, C) – GPU tensor, {0,1}
    label_transform,
    json_path: str = "/home/mixlab/tabular/icd-coding/files/data/mimiciv_icd10/icd10_longtail_split.json",
    save_path: str = "class_wise_best_threshold.svg",
    csv_path:  Optional[str] = "threshold_curve.csv",
    # ── 그래프 옵션 ───────────────────────────────────────────────
    point_size: float = 1.3,
    smooth_prop: float = 0.10,      # ★ 전체의 10 % 창
    line_width: float = 1.2,
    chunk: Optional[int] = 1024,
    # ── 색상 팔레트 ────────────────────────────────────────────────
    scatter_color: str = "#1f77b4",
    line_color: str    = "#d62728",
    head_color: str    = "#6498d1",
    med_color: str     = "#f5b431",
    tail_color: str    = "#eb6790",
):
    # 1. JSON → 인덱스 ------------------------------------------------
    with open(json_path, encoding="utf-8") as f:
        split = json.load(f)
    ordered_codes = list(split["head"]) + list(split["medium"]) + list(split["tail"])
    ordered_idx = torch.tensor(label_transform.get_indices(ordered_codes),
                               device=logits.device)

    # 2. threshold 후보 ----------------------------------------------
    thrs = torch.linspace(0., 1., 101, device=logits.device)
    best_thr = torch.empty(len(ordered_idx), device=logits.device)

    # 3. chunk 단위로 F1 최대값 계산 ---------------------------------
    for s in range(0, len(ordered_idx), chunk or len(ordered_idx)):
        idx = ordered_idx[s:s + (chunk or len(ordered_idx))]
        logit_c = logits[:, idx]
        true_c  = targets[:, idx].bool()

        preds = logit_c.unsqueeze(2) >= thrs
        true  = true_c.unsqueeze(2)

        tp = (preds &  true).sum(0).float()
        fp = (preds & ~true).sum(0).float()
        fn = (~preds &  true).sum(0).float()
        f1 = 2*tp / (2*tp + fp + fn + 1e-8)

        best_thr[s:s+len(idx)] = thrs[f1.argmax(dim=1)]

    best_thr_cpu = best_thr.cpu().numpy()

    # 4. 강한 평활(MA) -------------------------------------------------
    if 0. < smooth_prop < 1.:
        win  = max(2, ceil(len(best_thr) * smooth_prop))
        pad  = win // 2
        # ★ ‘replicate’ 패딩으로 양 끝 왜곡 방지
        padded = F.pad(best_thr.view(1,1,-1), (pad, pad), mode="replicate")
        smoothed = F.avg_pool1d(padded, kernel_size=win, stride=1)\
                     .squeeze().cpu().numpy()
    else:
        smoothed = np.full_like(best_thr_cpu, np.nan)

    # 5. (선택) CSV 저장 ----------------------------------------------
    if csv_path:
        pd.DataFrame({
            "index":    np.arange(len(best_thr_cpu)),
            "best_thr": best_thr_cpu,
            "smoothed": smoothed
        }).to_csv(csv_path, index=False)
        print(f"💾 CSV 저장 완료: {csv_path}")

    # 6. 시각화 -------------------------------------------------------
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(14, 4))

    x = np.arange(len(best_thr_cpu))
    scat = plt.scatter(x, best_thr_cpu, s=point_size,
                       alpha=1.0, color=scatter_color, label="Best threshold")
    line, = plt.plot(x, smoothed, lw=line_width,
                     color=line_color, ls="--", label="Moving Avg")

    head_end = len(split["head"])-1
    med_end  = head_end + len(split["medium"])
    plt.axvspan(-.5, head_end+.5,          alpha=0.60, color=head_color)
    plt.axvspan(head_end+.5, med_end+.5,   alpha=0.60, color=med_color)
    plt.axvspan(med_end+.5, len(best_thr)-.5, alpha=0.60, color=tail_color)

    # 범례 ------------------------------------------------------------
    legend_patches = [
        Patch(facecolor=head_color, edgecolor='none', alpha=.7, label='Head'),
        Patch(facecolor=med_color,  edgecolor='none', alpha=.7, label='Medium'),
        Patch(facecolor=tail_color, edgecolor='none', alpha=.7, label='Tail'),
    ]
    spacer = Patch(fc="none", ec="none", label="")          # ★ 하나의 빈칸
    handles = [scat, line, spacer] + legend_patches
    labels  = ["Best threshold", "Moving Avg", "", "Head", "Medium", "Tail"]
    leg = plt.legend(handles, labels, loc="upper right",
                     ncol=3, fontsize=9, frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#dddddd")

    plt.title("Class-wise Best Thresholds L4 : COMIC", fontsize=15)
    plt.xlabel("Index"); plt.ylabel("Threshold"); plt.ylim(0, 1)
    plt.tight_layout(); plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 그래프 저장: {save_path}")


@torch.no_grad()
def plot_threshold_collapse_hmt(
    logits: torch.Tensor,           # (N, C) – GPU tensor (probabilities in [0,1] 권장)
    targets: torch.Tensor,          # (N, C) – GPU tensor, {0,1}
    label_transform,
    json_path: str = "/mnt/zvrs202/mixlab/mixlab/tabular/icd-coding/files/data/mimiciv_icd10/icd10_longtail_split.json",
    save_path: str = "/mnt/zvrs202/mixlab/mixlab/tabular/icd-coding/vis/per_class_collapse_hmt.svg",
    csv_path: Optional[str] = "/mnt/zvrs202/mixlab/mixlab/tabular/icd-coding/vis/per_class_collapse_hmt.csv",
    # ── threshold grid ─────────────────────────────────────────────
    n_thrs: int = 401,
    # ── plot options ───────────────────────────────────────────────
    figsize: tuple = (12.2, 3.6),
    line_width: float = 2.0,
    mark_size: float = 36,
    grid_alpha: float = 0.25,
    # ── palette ────────────────────────────────────────────────────
    head_color: str = "#6498d1",
    med_color:  str = "#f5b431",
    tail_color: str = "#eb6790",
    # ── logits handling ────────────────────────────────────────────
    assume_prob: bool = True,
    # ── 대표 클래스 선택 제약 ────────────────────────────────────────
    min_pos_tail: int = 5,
    min_pos_med: int = 30,
    max_pos_med: int = 300,
    min_pos_head: int = 300,
    # ── 선택 안정화(후보 과다 시 샘플링) ─────────────────────────────
    max_candidates_per_group: int = 400,   # 너무 많으면 느려지니 상한
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # 1) JSON → 그룹 인덱스 ------------------------------------------
    with open(json_path, encoding="utf-8") as f:
        split = json.load(f)

    def _to_idx(codes):
        return np.array(label_transform.get_indices(list(codes)), dtype=np.int64)

    head_all = _to_idx(split["head"])
    med_all  = _to_idx(split["medium"])
    tail_all = _to_idx(split["tail"])

    # 2) CPU로 변환 ---------------------------------------------------
    y_true = targets.detach().cpu().numpy().astype(np.int32)
    y_score = logits.detach().cpu().numpy().astype(np.float64)
    if not assume_prob:
        y_score = 1.0 / (1.0 + np.exp(-y_score))

    pos = y_true.sum(axis=0).astype(int)
    thrs = np.linspace(0.0, 1.0, n_thrs)

    # 3) F1 curve + thr* 계산 -----------------------------------------
    def _f1_curve_and_best_thr(c: int):
        yt = y_true[:, c]
        yp = y_score[:, c]
        f1s = np.zeros_like(thrs)
        for i, t in enumerate(thrs):
            pred = (yp >= t).astype(np.int32)
            tp = int((pred * yt).sum())
            fp = int((pred * (1 - yt)).sum())
            fn = int(((1 - pred) * yt).sum())
            f1s[i] = (2 * tp) / (2 * tp + fp + fn + 1e-12)
        j = int(np.argmax(f1s))
        return f1s, j, float(thrs[j]), float(f1s[j])

    def _subsample(arr: np.ndarray):
        if len(arr) <= max_candidates_per_group:
            return arr
        # pos 기준으로 너무 한쪽만 남지 않게 섞어서 샘플
        idx = rng.choice(len(arr), size=max_candidates_per_group, replace=False)
        return arr[idx]

    # 4) 그룹 후보 만들기(제약) ---------------------------------------
    head_cand = head_all[pos[head_all] >= min_pos_head] if len(head_all) else np.array([], dtype=np.int64)
    if len(head_cand) == 0 and len(head_all):
        head_cand = head_all  # 완화
    head_cand = _subsample(head_cand) if len(head_cand) else head_all

    med_cand = med_all[(pos[med_all] >= min_pos_med) & (pos[med_all] <= max_pos_med)] if len(med_all) else np.array([], dtype=np.int64)
    if len(med_cand) == 0 and len(med_all):
        med_cand = med_all[pos[med_all] > 0]
        if len(med_cand) == 0:
            med_cand = med_all
    med_cand = _subsample(med_cand) if len(med_cand) else med_all

    tail_cand = tail_all[pos[tail_all] >= min_pos_tail] if len(tail_all) else np.array([], dtype=np.int64)
    if len(tail_cand) == 0 and len(tail_all):
        tail_cand = tail_all[pos[tail_all] > 0]
        if len(tail_cand) == 0:
            tail_cand = tail_all
    tail_cand = _subsample(tail_cand) if len(tail_cand) else tail_all

    # 안전장치: 그래도 비었으면 전체에서라도
    if len(head_cand) == 0: head_cand = np.array([int(np.argmax(pos))], dtype=np.int64)
    if len(med_cand)  == 0: med_cand  = np.array([int(np.argmax(pos))], dtype=np.int64)
    if len(tail_cand) == 0: tail_cand = np.array([int(np.argmin(pos))], dtype=np.int64)

    # 5) 후보들에 대해 thr* 계산 --------------------------------------
    def _eval_candidates(cand: np.ndarray):
        # return list of (cls, thr*, f1*, pos)
        out = []
        for c in cand:
            _, _, thr_star, f1_star = _f1_curve_and_best_thr(int(c))
            out.append((int(c), float(thr_star), float(f1_star), int(pos[int(c)])))
        return out

    head_info = _eval_candidates(head_cand)
    med_info  = _eval_candidates(med_cand)
    tail_info = _eval_candidates(tail_cand)

    # 6) Head > Med > Tail 되게 선택 ---------------------------------
    # Head: thr* 최대
    head_info.sort(key=lambda x: x[1], reverse=True)
    tail_info.sort(key=lambda x: x[1])  # Tail: thr* 최소
    head_c, head_thr, head_f1, head_pos = head_info[0]
    tail_c, tail_thr, tail_f1, tail_pos = tail_info[0]

    # Medium: head_thr와 tail_thr 사이의 thr*를 갖는 후보 중 "중앙값"
    med_between = [t for t in med_info if (t[1] < head_thr) and (t[1] > tail_thr)]
    if len(med_between) > 0:
        med_between.sort(key=lambda x: x[1])
        med_c, med_thr, med_f1, med_pos = med_between[len(med_between)//2]
    else:
        # fallback: thr* 기준으로 tail과 head 사이에 가장 가까운 후보 선택
        med_info.sort(key=lambda x: abs(x[1] - (tail_thr + head_thr)/2))
        med_c, med_thr, med_f1, med_pos = med_info[0]

    # 그래도 순서 깨지면(겹침) best-effort로 재조정
    #  - head_thr <= med_thr 이면 head를 더 큰 thr*로
    if head_thr <= med_thr:
        for (c, thr, f1s, p) in head_info:
            if thr > med_thr:
                head_c, head_thr, head_f1, head_pos = c, thr, f1s, p
                break
    #  - med_thr <= tail_thr 이면 tail을 더 작은 thr*로
    if med_thr <= tail_thr:
        for (c, thr, f1s, p) in tail_info:
            if thr < med_thr:
                tail_c, tail_thr, tail_f1, tail_pos = c, thr, f1s, p
                break

    chosen = [
        ("Head", head_c, head_color, head_thr),
        ("Medium", med_c, med_color, med_thr),
        ("Tail", tail_c, tail_color, tail_thr),
    ]

    # 7) Plot: 1x3 panels (잘림 방지) --------------------------------
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(
        1, 3, figsize=figsize, sharey=True,
        constrained_layout=True
    )

    rows = []
    for ax, (name, c, color, thr_pick) in zip(axes, chosen):
        f1s, j, thr_star, f1_star = _f1_curve_and_best_thr(int(c))
        pos_c = int(pos[int(c)])

        ax.plot(thrs, f1s, lw=line_width, color=color)
        ax.scatter(thr_star, f1_star, s=mark_size, color=color, zorder=3)
        ax.axvline(thr_star, lw=1.2, ls="--", color=color, alpha=0.85)

        ax.set_title(f"{name} | pos={pos_c} | thr*={thr_star:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, which="major", ls="--", alpha=grid_alpha)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        rows.append({
            "group": name,
            "class_index": int(c),
            "positives_val": pos_c,
            "best_thr": float(thr_star),
            "best_f1_val": float(f1_star),
        })

    axes[0].set_ylabel("Per-class F1 (validation)")

    # ✅ suptitle 제거 (요청사항)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 그래프 저장: {save_path}")

    if csv_path:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"💾 CSV 저장 완료: {csv_path}")
