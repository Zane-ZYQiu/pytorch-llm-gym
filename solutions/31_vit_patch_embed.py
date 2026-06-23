HINTS = [
    "patchify: B,C,H,W = images.shape; p = patch; nh,nw = H//p, W//p.",
    "x = images.reshape(B,C,nh,p,nw,p).permute(0,2,4,1,3,5)  # (B,nh,nw,C,p,p), gathers the pixels of each patch together.",
    "x = x.reshape(B, nh*nw, C*p*p). Row-major: nh comes before nw.",
    "vit_embed: x = patchify(...) @ proj; cls_tok = cls.expand(B,-1,-1); x = torch.cat([cls_tok, x], dim=1) + pos.",
]

# ===== reference solution =====
import torch


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    B, C, H, W = images.shape
    p = patch
    nh, nw = H // p, W // p
    x = images.reshape(B, C, nh, p, nw, p)
    x = x.permute(0, 2, 4, 1, 3, 5)          # (B, nh, nw, C, p, p)
    x = x.reshape(B, nh * nw, C * p * p)
    return x


def vit_embed(images: torch.Tensor, patch: int, proj: torch.Tensor,
              cls: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    B = images.shape[0]
    x = patchify(images, patch) @ proj       # (B, N, D)
    cls_tok = cls.expand(B, -1, -1)          # (B, 1, D)
    x = torch.cat([cls_tok, x], dim=1)       # (B, N+1, D)
    return x + pos
