"""Run a production-shape CUDA forward/backward smoke for UniGIR V3.

This diagnostic intentionally bypasses the data loader and linear probing. It
checks the geodesic profile stages that were absent from the existing CPU-only
tests, with an explicit CUDA synchronization after each stage.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.geodesic_profile import GeodesicPrototypeAlignment


DEFAULTS = {
    "seed": 7,
    "batch_size": 64,
    "num_masks": 6,
    "dim": 256,
    "num_prototypes": 128,
    "queue_size": 1024,
    "graph_k": 8,
    "graph_anchors": 4,
}


def synchronize(stage: str) -> None:
    torch.cuda.synchronize()
    print(f"PASS {stage}", flush=True)


def finite_scalar(name: str, value: torch.Tensor) -> None:
    if value.numel() != 1 or not torch.isfinite(value).all():
        raise RuntimeError(f"{name} is not a finite scalar: {value}")


def build_module(device: torch.device) -> GeodesicPrototypeAlignment:
    module = GeodesicPrototypeAlignment(
        dim=DEFAULTS["dim"],
        num_prototypes=DEFAULTS["num_prototypes"],
        queue_size=DEFAULTS["queue_size"],
        graph_k=DEFAULTS["graph_k"],
        graph_anchors=DEFAULTS["graph_anchors"],
        graph_update_frequency=1,
        geodesic_temperature=0.1,
        loss_weight=0.25,
        init_seed=DEFAULTS["seed"],
    ).to(device)
    module.eval()
    module.set_epoch(0)
    return module


def fill_queue(module: GeodesicPrototypeAlignment, device: torch.device) -> None:
    queue = F.normalize(
        torch.randn(DEFAULTS["queue_size"], DEFAULTS["dim"], device=device),
        p=2,
        dim=-1,
    )
    module.feature_queue.copy_(queue)
    module.queue_ptr.zero_()
    module.queue_len.fill_(DEFAULTS["queue_size"])


def run_diagnostic() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    torch.manual_seed(DEFAULTS["seed"])
    device = torch.device("cuda")
    batch_size = DEFAULTS["batch_size"]
    dim = DEFAULTS["dim"]
    num_masks = DEFAULTS["num_masks"]

    print(
        "CONFIG "
        f"seed={DEFAULTS['seed']} batch={batch_size} masks={num_masks} "
        f"dim={dim} prototypes={DEFAULTS['num_prototypes']} "
        f"queue={DEFAULTS['queue_size']} graph_k={DEFAULTS['graph_k']} "
        f"anchors={DEFAULTS['graph_anchors']}",
        flush=True,
    )
    print(
        "ENV "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(device)} amp=False",
        flush=True,
    )

    module = build_module(device)
    fill_queue(module, device)
    torch.cuda.synchronize()
    print(f"QUEUE_PREP queue_len={int(module.queue_len.item())}", flush=True)
    prototypes = module.prototypes.detach().clone()

    shortest, stats = module._build_graph(prototypes)
    if shortest.shape != (DEFAULTS["num_prototypes"], DEFAULTS["num_prototypes"]):
        raise RuntimeError(f"unexpected graph shape: {tuple(shortest.shape)}")
    if not torch.isfinite(shortest).all():
        raise RuntimeError("graph shortest paths contain non-finite values")
    synchronize("graph_build")
    print(f"GRAPH_STATS {stats}", flush=True)

    z_full_a = torch.randn(batch_size, dim, device=device)
    z_full_b = torch.randn(batch_size, dim, device=device)
    masks_a = [
        torch.randn(batch_size, dim, device=device, requires_grad=True)
        for _ in range(num_masks)
    ]
    masks_b = [
        torch.randn(batch_size, dim, device=device, requires_grad=True)
        for _ in range(num_masks)
    ]

    prot = module.prototypes.detach().clone()
    q_a = module._codes(module._norm(z_full_a), prot)
    q_b = module._codes(module._norm(z_full_b), prot)
    if q_a.shape != (batch_size, DEFAULTS["num_prototypes"]):
        raise RuntimeError(f"unexpected code shape: {tuple(q_a.shape)}")
    if not torch.isfinite(q_a).all() or not torch.isfinite(q_b).all():
        raise RuntimeError("full-view codes contain non-finite values")
    if not torch.allclose(
            q_a.sum(dim=-1),
            torch.ones(batch_size, device=device),
            atol=1e-5,
            rtol=1e-5,
    ):
        raise RuntimeError("full-view code rows do not sum to one")
    uniform = torch.full_like(q_a, 1.0 / DEFAULTS["num_prototypes"])
    if torch.allclose(q_a, uniform, atol=1e-7, rtol=1e-5):
        raise RuntimeError("full-view codes fell back to uniform assignment")
    synchronize("full_view_codes")

    pred = module._predict(
        [module._norm(mask) for mask in masks_a], q_a, prot)
    finite_scalar("masked_profile_loss", pred["loss"])
    finite_scalar("masked_profile_kl", pred["kl"])
    synchronize("masked_profile_forward")

    pred["loss"].backward()
    if any(mask.grad is None for mask in masks_a):
        raise RuntimeError("masked views did not receive gradients")
    if not all(torch.isfinite(mask.grad).all() for mask in masks_a):
        raise RuntimeError("masked-view gradients contain non-finite values")
    synchronize("masked_profile_backward")

    end_to_end = build_module(device)
    fill_queue(end_to_end, device)
    e2e_masks_a = [mask.detach().clone().requires_grad_() for mask in masks_a]
    e2e_masks_b = [mask.detach().clone().requires_grad_() for mask in masks_b]
    output = end_to_end(z_full_a, e2e_masks_a, z_full_b, e2e_masks_b)
    finite_scalar("end_to_end_loss", output["loss"])
    synchronize("end_to_end_forward")
    output["loss"].backward()
    if any(mask.grad is None for mask in e2e_masks_a + e2e_masks_b):
        raise RuntimeError("end-to-end masked views did not receive gradients")
    synchronize("end_to_end_backward")
    print(
        "RESULT PASS "
        f"queue_len={int(end_to_end.queue_len.item())} "
        f"loss={float(output['loss'].detach().cpu())}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    try:
        run_diagnostic()
    except Exception as exc:  # noqa: BLE001 - diagnostic must preserve traceback.
        print(f"FAIL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
