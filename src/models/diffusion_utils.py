"""Utility functions for diffusion models."""
import torch
from diffusers.training_utils import EMAModel

from src.models.embedding import ConditionalDINOEmbedding, ConditionalEmbedding
from src.models.unet import Unet


def _load_model(ckpt_path, device, local_cfg, clsnum, discrete_label=True):
    """Load model from checkpoint."""
    net = Unet(
        in_ch=local_cfg.get("in_ch"),
        mod_ch=local_cfg.get("mod_ch"),
        out_ch=local_cfg.get("out_ch"),
        ch_mul=local_cfg.get("ch_mul"),
        num_res_blocks=local_cfg.get("num_res_blocks"),
        cdim=local_cfg.get("cdim"),
        use_conv=local_cfg.get("use_conv"),
        droprate=local_cfg.get("droprate"),
        dtype=local_cfg.get("dtype"),
    ).to(device)

    if discrete_label:
        cemblayer = ConditionalEmbedding(
            clsnum, local_cfg.get("cdim"), local_cfg.get("cdim")
        ).to(device)
    else:
        cemblayer = ConditionalDINOEmbedding(
            dim=local_cfg.get("cdim"),
            d_model=local_cfg.get("cdim"),
            dino_dim=1024,
        ).to(device)

    if ckpt_path is not None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            net.load_state_dict(ckpt["net"])
            cemblayer.load_state_dict(ckpt["cemblayer"])
            ema_state = ckpt.get("net_ema")

            if ema_state is not None:
                ema = EMAModel(
                    net.parameters(),
                    model_cls=type(net),
                    model_config=getattr(net, "config", None),
                )
                ema.load_state_dict(ema_state)
                ema.copy_to(net.parameters())
        except Exception as e:
            print(f"Error loading checkpoint from {ckpt_path}: {e}")
            raise e

    return net, cemblayer


def _generate_samples(
    args,
    cemblayer,
    diffusion,
    labels,
    in_ch,
    H,
    W,
    device,
    batch_lim=1024,
):
    """Generate samples from the diffusion model."""
    total = labels.shape[0]
    out = []

    if args.discrete_label:
        unconds = torch.zeros_like(labels)
    else:
        unconds = cemblayer.null.detach().unsqueeze(0).expand(len(labels), -1)

    with torch.inference_mode():
        for s in range(0, total, batch_lim):
            e = min(s + batch_lim, total)
            uncond = unconds[s:e]
            uncond_cemb = cemblayer(uncond, drop_prob=0.0)

            lab = labels[s:e]
            cemb = cemblayer(lab, drop_prob=0.0)
            genshape = (e - s, in_ch, H, W)
            gen = torch.Generator(device=device)
            gen.manual_seed(int(args.opt_seed) + int(s))

            if args.ddim:
                x = diffusion.ddim_sample(
                    genshape,
                    args.num_steps,
                    args.eta,
                    args.select,
                    cemb=cemb,
                    uncond_cemb=uncond_cemb,
                    generator=gen,
                )
            else:
                x = diffusion.sample(genshape, cemb=cemb, uncond_cemb=uncond_cemb)
            out.append((x / 2 + 0.5).clamp(0, 1).to("cpu"))
            del x, cemb, lab
            torch.cuda.empty_cache()
    return torch.cat(out, 0)
