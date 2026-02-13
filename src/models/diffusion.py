"""
Classifier free guidance diffusion model from [1]
[1]: https://github.com/jcwang-gh/classifier-free-diffusion-guidance-Pytorch
"""
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _safe_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion class"""

    def __init__(
        self,
        dtype: torch.dtype,
        model,
        betas: np.ndarray,
        w: float,
        v: float,
        device: torch.device,
    ):
        super().__init__()
        self.dtype = dtype
        param_sample = next(model.parameters(), None)
        if param_sample is not None and param_sample.device != device:
            self.model = model.to(device, dtype=dtype)
        else:
            self.model = model

        self.betas = torch.tensor(betas, dtype=self.dtype)
        self.w = w
        self.v = v
        self.T = len(betas)
        self.device = device
        self.alphas = 1 - self.betas
        self.log_alphas = torch.log(self.alphas)

        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim=0)
        self.alphas_bar = torch.exp(self.log_alphas_bar)
        # self.alphas_bar = torch.cumprod(self.alphas, dim = 0)

        self.log_alphas_bar_prev = F.pad(
            self.log_alphas_bar[:-1], [1, 0], "constant", 0
        )
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev)
        self.log_one_minus_alphas_bar_prev = torch.log(1.0 - self.alphas_bar_prev)
        # self.alphas_bar_prev = F.pad(self.alphas_bar[:-1],[1,0],'constant',1)

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas)
        # self.sqrt_alphas = torch.sqrt(self.alphas)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar)
        # self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar)

        # calculate parameters for q(x_{t-1}|x_t,x_0)
        # log calculation clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * torch.exp(
            self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar
        )
        self.log_tilde_betas_clipped = torch.log(
            torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0)
        )
        self.mu_coef_x0 = self.betas * torch.exp(
            0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar
        )
        self.mu_coef_xt = torch.exp(
            0.5 * self.log_alphas
            + self.log_one_minus_alphas_bar_prev
            - self.log_one_minus_alphas_bar
        )
        self.vars = self.tilde_betas
        self.coef1 = torch.exp(-self.log_sqrt_alphas)
        self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar)
        # self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(
            self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar
        )
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)

    @staticmethod
    def _extract(coef: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        input:

        coef : an array
        t : timestep
        x_shape : the shape of tensor x that has K dims
        (the value of first dim is batch size)

        output:

        a tensor of shape [batchsize,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]

        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        coef = coef.to(t.device)
        chosen = coef[t]
        chosen = chosen.to(t.device)

        return chosen.reshape(neo_shape)

    def q_mean_variance(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the parameters of q(x_t|x_0)"""
        mean = self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
        var = self._extract(1.0 - self.alphas_bar, t, x_0.shape)
        return mean, var

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t|x_0)"""
        eps = torch.randn_like(x_0, requires_grad=False)
        return (
            self._extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
            + self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * eps,
            eps,
        )

    def q_posterior_mean_variance(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the parameters of q(x_{t-1}|x_t,x_0)"""
        posterior_mean = (
            self._extract(self.mu_coef_x0, t, x_0.shape) * x_0
            + self._extract(self.mu_coef_xt, t, x_t.shape) * x_t
        )
        posterior_var_max = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min = self._extract(
            self.log_tilde_betas_clipped, t, x_t.shape
        )
        log_posterior_var_max = self._extract(torch.log(self.betas), t, x_t.shape)
        log_posterior_var = (
            self.v * log_posterior_var_max + (1 - self.v) * log_posterior_var_min
        )
        neo_posterior_var = torch.exp(log_posterior_var)

        return posterior_mean, posterior_var_max, neo_posterior_var

    def p_mean_variance(
        self, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the parameters of p_{theta}(x_{t-1}|x_t)"""

        full_cemb = model_kwargs.get("full_cemb", None)  # conditional embedding layer
        cemb = model_kwargs.get("cemb", None)  # current conditional embedding if any
        subset_info = model_kwargs.get("subset_prior", None)  # dict or None
        discrete = bool(model_kwargs.get("discrete_label", False))

        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        if "cemb" in model_kwargs:
            cemb = model_kwargs["cemb"]
            pred_eps_cond = self.model(x_t, t.long(), **model_kwargs)
            uncond_kwargs = dict(model_kwargs)
            uncond_kwargs["cemb"] = torch.zeros_like(cemb, device=self.device)
            pred_eps_uncond = self.model(x_t, t.long(), **uncond_kwargs)
        else:
            # No conditioning provided: just one forward
            pred_eps_uncond = pred_eps_cond = self.model(x_t, t.long(), **model_kwargs)

        if subset_info is not None:
            K = int(subset_info["K"])
            support = subset_info[
                "support"
            ]  # tensor of class ids (discrete) or embeddings (cont.)
            pred_eps_cond = self._mc_full_cond_score(
                x_t, t, K, support, full_cemb, discrete
            )

        pred_eps = pred_eps_uncond + self.w * (pred_eps_cond - pred_eps_uncond)

        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert (
            torch.isnan(pred_eps).int().sum() == 0
        ), f"nan in tensor pred_eps when t = {t[0]}"
        p_mean = self._predict_xt_prev_mean_from_eps(
            x_t, t.type(dtype=torch.long), pred_eps
        )
        p_var = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        return p_mean, p_var

    def _predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            self._extract(coef=self.sqrt_recip_alphas_bar, t=t, x_shape=x_t.shape) * x_t
            - self._extract(coef=self.sqrt_recipm1_alphas_bar, t=t, x_shape=x_t.shape)
            * eps
        )

    def _predict_xt_prev_mean_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            self._extract(coef=self.coef1, t=t, x_shape=x_t.shape) * x_t
            - self._extract(coef=self.coef2, t=t, x_shape=x_t.shape) * eps
        )

    @torch.no_grad()
    def _mc_full_cond_score(
        self,
        x_t,
        t,
        K,
        support,
        full_cemb,
        discrete: bool,
        z_t,  # [B,d_dino] current DINO embedding (normalized)
        topK: int = 128,  # size of local neighborhood
        kappa: float = 30.0,  # vMF / softmax concentration
    ):
        """Monte-Carlo mean of conditional scores using a LOCAL prior around z_t."""
        B = x_t.size(0)
        device = x_t.device
        N = len(support)

        # --- prepare time tensor
        if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.ndim == 0):
            t_rep = torch.full((B,), int(t), device=device, dtype=torch.long)
        else:
            t_rep = t.to(device).long().view(B)

        # --- find nearest neighbors in DINO space
        # z_t, support_feats already L2-normalized

        z_t = z_t / (z_t.norm(dim=-1, keepdim=True) + 1e-6)

        support = support / (support.norm(dim=-1, keepdim=True) + 1e-6)

        sims = z_t @ support.T  # [B, N], cosine similarity
        top_idx = torch.topk(sims, k=min(topK, N), dim=1).indices  # [B, topK]

        # --- per-sample local sampling
        eps_sum = torch.zeros_like(x_t)
        for b in range(B):
            idx_pool = top_idx[b]  # [topK]
            sims_b = sims[b, idx_pool]  # [topK]
            weights = torch.softmax(kappa * sims_b, dim=0)

            # sample K embeddings from the local pool with soft weights
            sel = torch.multinomial(weights, num_samples=K, replacement=False)
            idx_sel = idx_pool[sel]

            # build conditional embeddings for these K samples
            if discrete:
                y = support[idx_sel].long().to(device)
                c = full_cemb(y)
            else:
                e = support[idx_sel].to(device).float()
                c = full_cemb(e)

            # repeat x_t[b] and t_rep[b] to match K
            x_rep = x_t[b : b + 1].expand(K, *x_t.shape[1:])
            t_b = t_rep[b : b + 1].expand(K)

            # model forward
            eps = self.model(x_rep, t_b, c)  # [K,C,H,W]
            eps_sum[b : b + 1] = eps.mean(0, keepdim=True)  # average over local samples

        return eps_sum

    def p_sample(
        self, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs
    ) -> torch.Tensor:
        """Sample x_{t-1} from p_{theta}(x_{t-1}|x_t)"""
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t, t, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0
        return mean + torch.sqrt(var) * noise

    def sample(self, shape: tuple, **model_kwargs) -> torch.Tensor:
        """Sample images from p_{theta}"""
        local_rank = _safe_rank()
        if local_rank == 0:
            print("Start generating...")
        if model_kwargs is None:
            model_kwargs = {}
        x_t = torch.randn(shape, device=self.device, dtype=self.dtype)
        tlist = torch.full((x_t.shape[0],), self.T - 1, device=self.device)
        for _ in tqdm(
            range(self.T),
            dynamic_ncols=True,
            disable=(_safe_rank() != 0),
        ):
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
            tlist -= 1
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print("ending sampling process...")
        return x_t

    def ddim_p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prevt: torch.Tensor,
        eta: float,
        **model_kwargs,
    ) -> torch.Tensor:
        """Calculate the parameters of p_{theta}(x_{t-1}|x_t)"""
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        full_cemb = model_kwargs.get("full_cemb", None)  # pass your FULL cemb layer
        cemb = model_kwargs.get("cemb", None)  # current conditional embedding if any
        subset_info = model_kwargs.get("subset_prior", None)  # dict or None
        K = model_kwargs.get("K", 1)
        uncond_cemb = model_kwargs.get("uncond_cemb", None)
        discrete = bool(model_kwargs.get("discrete_label", False))
        z_t = model_kwargs.get("labels", None)

        if "cemb" in model_kwargs:
            cemb = model_kwargs["cemb"]
            pred_eps_cond = self.model(x_t, t.long(), cemb)
            pred_eps_uncond = self.model(x_t, t.long(), uncond_cemb)
        else:
            pred_eps_uncond = pred_eps_cond = self.model(x_t, t.long(), **model_kwargs)

        if subset_info is not None:
            support = (
                subset_info  # tensor of class ids (discrete) or embeddings (cont.)
            )
            pred_eps_cond_2 = self._mc_full_cond_score(
                x_t, t, K, support, full_cemb, discrete, z_t
            )
            pred_eps = pred_eps_uncond + self.w * (
                0.5 * (pred_eps_cond_2 + pred_eps_cond) - pred_eps_uncond
            )
        else:
            pred_eps = pred_eps_uncond + self.w * (pred_eps_cond - pred_eps_uncond)

        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert (
            torch.isnan(pred_eps).int().sum() == 0
        ), f"nan in tensor pred_eps when t = {t[0]}"

        alphas_bar_t = self._extract(coef=self.alphas_bar, t=t, x_shape=x_t.shape)
        needs_one = (prevt < 0).view(-1, *([1] * (x_t.dim() - 1)))
        alphas_bar_prev = torch.where(
            needs_one,
            torch.ones_like(alphas_bar_t),
            self._extract(self.alphas_bar_prev, prevt + 1, x_t.shape),
        )
        sigma = eta * torch.sqrt(
            (1 - alphas_bar_prev)
            / (1 - alphas_bar_t)
            * (1 - alphas_bar_t / alphas_bar_prev)
        )
        p_var = sigma ** 2
        coef_eps = 1 - alphas_bar_prev - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps = torch.sqrt(coef_eps)
        p_mean = (
            torch.sqrt(alphas_bar_prev)
            * (x_t - torch.sqrt(1 - alphas_bar_t) * pred_eps)
            / torch.sqrt(alphas_bar_t)
            + coef_eps * pred_eps
        )
        return p_mean, p_var

    def ddim_p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prevt: torch.Tensor,
        eta: float,
        **model_kwargs,
    ) -> torch.Tensor:
        """Sample x_{t-1} from p_{theta}(x_{t-1}|x_t)"""

        if model_kwargs is None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.ddim_p_mean_variance(
            x_t,
            t.type(dtype=torch.long),
            prevt.type(dtype=torch.long),
            eta,
            **model_kwargs,
        )
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0
        return mean + torch.sqrt(var) * noise

    def ddim_sample(
        self,
        shape: tuple,
        num_steps: int,
        eta: float,
        select: str,
        generator: torch.Generator | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Sample images from p_{theta} using ddim"""

        local_rank = _safe_rank()
        if local_rank == 0:
            print("Start generating(ddim)...")
        if model_kwargs is None:
            model_kwargs = {}
        # a subsequence of range(0,1000)
        if select == "linear":
            tseq = list(np.linspace(0, self.T - 1, num_steps).astype(int))
        elif select == "quadratic":
            tseq = list(
                (np.linspace(0, np.sqrt(self.T), num_steps - 1) ** 2).astype(int)
            )
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(
                f'There is no ddim discretization method called "{select}"'
            )
        # deterministic init: prefer provided noise, else generate with given generator
        if generator is not None:
            x_t = torch.randn(
                shape, device=self.device, dtype=self.dtype, generator=generator
            )
        else:
            x_t = torch.randn(shape, device=self.device)

        tlist = torch.zeros([x_t.shape[0]], device=self.device, dtype=self.dtype)
        for i in tqdm(
            range(num_steps),
            dynamic_ncols=True,
            disable=(_safe_rank() != 0),
        ):
            with torch.no_grad():
                tlist = tlist * 0 + tseq[-1 - i]
                if i != num_steps - 1:
                    prevt = torch.ones_like(tlist, device=self.device) * tseq[-2 - i]
                else:
                    prevt = -torch.ones_like(tlist, device=self.device)
                x_t = self.ddim_p_sample(x_t, tlist, prevt, eta, **model_kwargs)
                torch.cuda.empty_cache()
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print("ending sampling process(ddim)...")
        return x_t

    def trainloss(self, x_0: torch.Tensor, **model_kwargs) -> torch.Tensor:
        """Calculate the loss of denoising diffusion probabilistic model"""
        if model_kwargs is None:
            model_kwargs = {}
        t = torch.randint(self.T, size=(x_0.shape[0],), device=self.device)
        x_t, eps = self.q_sample(x_0, t)
        pred_eps = self.model(x_t, t.long(), **model_kwargs)
        loss = F.mse_loss(pred_eps, eps, reduction="mean")
        return loss
