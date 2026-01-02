import torch

class EstimatorCV:
    """
    ICD용:
    - 전역 EMA로 유지 가능한 건 벡터(Prop/Cov_pos/Cov_neg/Amount)만 유지
    - Sigma/Ro/Tao (Ca×Ca)는 배치 로컬로만 계산해서 반환 (절대 전역 저장 X)
    """
    def __init__(self, num_classes: int, eps: float = 1e-8):
        self.num_classes = num_classes
        self.eps = eps

        # ✅ 전역 상태는 CPU float32로 (메모리/안정성)
        self.Amount  = torch.zeros(num_classes, dtype=torch.float32)  # 누적 pos count
        self.Prop    = torch.zeros(num_classes, dtype=torch.float32)
        self.Cov_pos = torch.zeros(num_classes, dtype=torch.float32)
        self.Cov_neg = torch.zeros(num_classes, dtype=torch.float32)

    @torch.no_grad()
    def update(self, labels_ca: torch.Tensor, logits_ca: torch.Tensor, active_idx: torch.Tensor, device: torch.device):
        """
        labels_ca/logits_ca: [B, Ca]
        active_idx: [Ca] (원래 클래스 인덱스)
        return:
          prop_ca/cov_pos_ca/cov_neg_ca: [Ca] (GPU)
          sigma/ro/tao: [Ca, Ca] (GPU)  -> 배치 로컬
        """
        eps = self.eps
        B, Ca = labels_ca.shape
        labels = labels_ca.to(torch.float32)
        logits = logits_ca.to(torch.float32)

        # ---- Prop (batch) ----
        sum_pos = labels.sum(0)                       # [Ca]
        prop = sum_pos / float(max(B, 1))            # [Ca]

        # ---- pos/neg variance (logits의 row_sum 기반이 아니라, "각 클래스 로짓 s_c" 기반이 더 안정적) ----
        # CXR(원본)도 실제로 s_c 축 기반 통계를 쓰는 게 ICD에선 훨씬 낫습니다.
        cov_pos = torch.zeros(Ca, device=device)
        cov_neg = torch.zeros(Ca, device=device)

        for i in range(Ca):
            pos = labels[:, i] > 0.5
            neg = ~pos
            s = logits[:, i]
            if pos.sum() > 1:
                cov_pos[i] = torch.var(s[pos], unbiased=False)
            else:
                cov_pos[i] = 0.0
            if neg.sum() > 1:
                cov_neg[i] = torch.var(s[neg], unbiased=False)
            else:
                cov_neg[i] = 0.0

        cov_pos = torch.nan_to_num(cov_pos, nan=0.0, posinf=0.0, neginf=0.0)
        cov_neg = torch.nan_to_num(cov_neg, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- Sigma/Ro/Tao : Ca×Ca 배치 로컬 ----
        # co-occurrence
        M = labels  # [B, Ca]
        Co = M.t() @ M  # [Ca, Ca]

        # sigma_cj: co-occur subset에서 "s_c" 분산 (원본의 result=logits[co,:]의 var(전체)보다 ICD에선 이게 훨씬 싸고 안정)
        sigma = torch.zeros((Ca, Ca), device=device)
        n_per = sum_pos.clamp_min(eps)  # [Ca]

        for c in range(Ca):
            pos_c = M[:, c] > 0.5
            s_c = logits[:, c]
            for j in range(Ca):
                co = pos_c & (M[:, j] > 0.5)
                if co.sum() > 1:
                    sigma[c, j] = torch.var(s_c[co], unbiased=False)
                else:
                    sigma[c, j] = 0.0

        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

        ro = Co / n_per.view(1, Ca)  # |c∧j| / |j|
        denom = (n_per.view(1, Ca) - Co).clamp_min(eps)
        tao = (float(B) - n_per.view(Ca, 1)) / denom

        # minmax norm (배치 로컬)
        def minmax(x):
            xmin = x.amin()
            xmax = x.amax()
            return (x - xmin) / (xmax - xmin + eps)

        sigma_n = minmax(sigma)
        ro_n = minmax(ro)
        tao_n = minmax(tao)

        # ---- 전역 EMA 업데이트 (CPU) : active_idx 위치만 sparse update ----
        idx_cpu = active_idx.detach().cpu()
        amount_act = self.Amount[idx_cpu].to(device=device)

        w_pr = sum_pos / (sum_pos + amount_act + eps)  # [Ca]
        w_pr_cpu = w_pr.detach().cpu()

        self.Prop[idx_cpu] = self.Prop[idx_cpu] * (1 - w_pr_cpu) + prop.detach().cpu() * w_pr_cpu
        self.Cov_pos[idx_cpu] = self.Cov_pos[idx_cpu] * (1 - w_pr_cpu) + cov_pos.detach().cpu() * w_pr_cpu

        # neg는 (B - pos) 기반 weight
        cnt_neg = (float(B) - sum_pos).clamp_min(eps)
        w_pr_neg = cnt_neg / (cnt_neg + amount_act + eps)
        w_pr_neg_cpu = w_pr_neg.detach().cpu()
        self.Cov_neg[idx_cpu] = self.Cov_neg[idx_cpu] * (1 - w_pr_neg_cpu) + cov_neg.detach().cpu() * w_pr_neg_cpu

        self.Amount[idx_cpu] += sum_pos.detach().cpu()

        return (
            prop.detach(), cov_pos.detach(), cov_neg.detach(),
            sigma_n.detach(), ro_n.detach(), tao_n.detach()
        )
