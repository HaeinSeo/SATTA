"""SATTA: Scale-Aware Tiled Test-Time Adjustment (paper Sec. 3, Algorithm 1, Supp. Table S1)."""
from dataclasses import dataclass
import time

import numpy as np


@dataclass
class SattaConfig:
    xs_max: float = 16.0            # s < 16  -> XS
    s_max: float = 32.0             # 16..32  -> S
    m_max: float = 96.0             # 32..96  -> M, >=96 -> L
    boost_small_tile: float = 1.10  # B(XS/S, tile)
    boost_large_global: float = 1.10  # B(M/L, global)
    penalty_large_tile: float = 0.85  # B(M/L, tile)
    gamma: float = 0.65             # min boundary factor (Eq. 7)
    margin: float = 0.08            # boundary margin m
    tau_xs: float = 0.055           # tau(XS) = tau(S)
    tau_s: float = 0.055
    tau_m: float = 0.10
    tau_l: float = 0.12
    cand_conf: float = 0.055        # candidate-generation threshold
    nms_iou: float = 0.50           # final class-wise NMS
    # component switches for the ablation study (all True = full SATTA, the paper configuration)
    use_source_size: bool = True    # source/size reweighting B(size, source)               (Eq. 5)
    use_boundary: bool = True       # tile-boundary penalty beta                            (Eqs. 6-8)
    use_threshold: bool = True      # size-dependent thresholds tau + candidate threshold   (Eq. 9)
    base_conf: float = 0.03         # uniform threshold used instead of tau / cand_conf when use_threshold=False (= SAHI's)

    def validate(self):
        numeric = [self.xs_max, self.s_max, self.m_max, self.boost_small_tile,
                   self.boost_large_global, self.penalty_large_tile, self.gamma,
                   self.margin, self.tau_xs, self.tau_s, self.tau_m, self.tau_l,
                   self.cand_conf, self.nms_iou, self.base_conf]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and np.isfinite(v) for v in numeric):
            raise ValueError("SATTA numeric parameters must be finite numbers")
        if not 0 < self.xs_max < self.s_max < self.m_max:
            raise ValueError("SATTA size boundaries must be positive and increasing")
        if self.margin <= 0 or not 0 <= self.gamma <= 1 or not 0 < self.nms_iou <= 1:
            raise ValueError("SATTA requires margin>0, gamma in [0,1], NMS IoU in (0,1]")
        if any(v < 0 for v in [self.boost_small_tile, self.boost_large_global,
                              self.penalty_large_tile, self.tau_xs, self.tau_s,
                              self.tau_m, self.tau_l, self.cand_conf, self.base_conf]):
            raise ValueError("SATTA weights and thresholds must be nonnegative")
        if not all(isinstance(v, bool) for v in [self.use_source_size, self.use_boundary, self.use_threshold]):
            raise ValueError("SATTA component switches must be booleans")


def bucket(s, c):
    return np.digitize(s, [c.xs_max, c.s_max, c.m_max])  # 0=XS 1=S 2=M 3=L


def satta_select(rec, c: SattaConfig = SattaConfig(), timers=None):
    """rec: cache record with 'g' (Nx6), 't' (Mx6) and 'tl' (Mx2 normalised in-tile centre).

    timers: optional dict; if given, wall-clock seconds of each stage are accumulated in it (latency breakdown).
    """
    try:
        import torch
        from torchvision.ops import batched_nms
    except ImportError as exc:
        raise ImportError("SATTA selection needs requirements-merge.txt") from exc
    c.validate()
    T = time.perf_counter
    t0 = T()
    g, t, tl = rec["g"], rec["t"], rec["tl"]
    boxes = np.concatenate([g[:, :4], t[:, :4]])
    conf = np.concatenate([g[:, 4], t[:, 4]]).astype(np.float64)
    cls = np.concatenate([g[:, 5], t[:, 5]])
    is_tile = np.concatenate([np.zeros(len(g), bool), np.ones(len(t), bool)])
    keep = conf >= (c.cand_conf if c.use_threshold else c.base_conf)
    boxes, conf, cls, is_tile = boxes[keep], conf[keep], cls[keep], is_tile[keep]
    cx = np.concatenate([np.zeros((len(g), 2)), tl])[keep]
    if len(conf) == 0:
        return np.zeros((0, 6), np.float32)
    s = np.sqrt(np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0))
    q = bucket(s, c)
    small = q <= 1
    t1 = T()
    B = np.ones_like(conf)
    if c.use_source_size:
        B[small & is_tile] = c.boost_small_tile
        B[~small & ~is_tile] = c.boost_large_global
        B[~small & is_tile] = c.penalty_large_tile
    p = conf * B
    t2 = T()
    if c.use_boundary:
        d = np.minimum(cx, 1 - cx).min(1)                                   # Eq. 6
        beta = c.gamma + (1 - c.gamma) * np.minimum(d / c.margin, 1.0)      # Eq. 7
        p = np.where(is_tile, p * beta, p)                                  # Eq. 8
    t3 = T()
    tau = np.array([c.tau_xs, c.tau_s, c.tau_m, c.tau_l])[q] if c.use_threshold else c.base_conf
    sel = p >= tau                                                      # Eq. 9
    t4 = T()
    if not sel.any():
        return np.zeros((0, 6), np.float32)
    b, p, cl = boxes[sel], p[sel], cls[sel]
    k = batched_nms(torch.from_numpy(b).float(), torch.from_numpy(p).float(),
                    torch.from_numpy(cl).long(), c.nms_iou).numpy()     # Eq. 10
    out = np.concatenate([b[k], p[k, None], cl[k, None]], 1).astype(np.float32)
    if timers is not None:
        t5 = T()
        for key, v in (("candidates+size", t1 - t0), ("source/size reweight", t2 - t1), ("boundary penalty", t3 - t2),
                       ("size thresholds", t4 - t3), ("class-wise NMS", t5 - t4)):
            timers[key] = timers.get(key, 0.0) + v
    return out
