import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class PrototypeAlignment(nn.Module):
    """Global relational-profile alignment for masked multi-modal SSL.

    Idea (UniGIR V2, self-contained version): every sample has a *global
    relational profile* -- its distribution over a set of K prototypes in the
    shared (post-projection) embedding space. We ask every masked view of a
    sample to predict the profile of the sample's complete (unmasked) view.

    Compared with InfMasking's batch-local InfoNCE, the prototype codes are
    computed with a Sinkhorn-Knopp equipartition step (SwAV-style), so the
    profile encodes *global* relative position within the batch, and the
    prototypes are kept on-line with an EMA so that no frozen pre-trained
    encoder or off-line clustering is required.

    Components:
        - `_codes(z)`: balanced soft assignment (Sinkhorn-Knopp) of full
          embeddings onto the prototypes. Detached: used as pseudo-label.
        - `_predict(masks, codes)`: KL/CE loss from each masked view to the
          (detached) codes of the full view.
        - `_update(...)`: EMA update of the prototypes with the current batch
          codes (no gradient flows to the prototypes).

    `loss_weight` and `cross` (masked views of aug1 predicting codes of the
    full view of aug2 and vice-versa) are configurable.
    """

    def __init__(self,
                 dim: int = 256,
                 num_prototypes: int = 300,
                 temperature: float = 0.1,
                 sinkhorn_iters: int = 3,
                 sinkhorn_eps: float = 0.05,
                 update_rate: float = 0.05,
                 loss_weight: float = 1.0,
                 cross: bool = False,
                 shuffle_targets: bool = False,
                 normalize: bool = True,
                 queue_size: int = 0,
                 init_seed: int = 0):
        super().__init__()
        self.dim = dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps
        self.update_rate = update_rate
        self.loss_weight = loss_weight
        self.cross = cross
        self.shuffle_targets = bool(shuffle_targets)
        self.normalize = normalize
        self.queue_size = int(queue_size)
        self.init_seed = int(init_seed)
        if not 0.0 < self.update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1]")

        # Use an isolated generator so enabling UniGIR does not change the
        # global RNG stream used by model initialization and data augmentation.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.init_seed)
        prot = F.normalize(
            torch.randn(num_prototypes, dim, generator=generator), p=2, dim=1)
        self.register_buffer("prototypes", prot)
        if self.queue_size > 0:
            self.register_buffer("feature_queue", torch.zeros(self.queue_size, dim))
            self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
            self.register_buffer("queue_len", torch.zeros((), dtype=torch.long))

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def forward(self,
                z_full_a: torch.Tensor,
                masks_a: List[torch.Tensor],
                z_full_b: Optional[torch.Tensor] = None,
                masks_b: Optional[List[torch.Tensor]] = None,
                cross: Optional[bool] = None) -> Optional[dict]:
        """Compute the profile-alignment losses.

        Args:
            z_full_a / z_full_b: complete multimodal embeddings of the two
                augmentations of the batch, shape (B, dim).
            masks_a / masks_b: lists of T masked-view embeddings, (B, dim).
            cross: if True, additionally align masks_a -> codes(z_full_b) and
                masks_b -> codes(z_full_a) (augmentation-agnostic profiles).

        Returns:
            dict with scalar tensors {"loss", "acc"} averaged over all
            alignments, or None if inputs are unusable (empty masks, bad dims).
        """
        if masks_a is None or len(masks_a) == 0:
            return None
        if self.cross and cross is None:
            cross = True
        cross = bool(cross)

        zf_a = self._norm(z_full_a)
        masks_a = [self._norm(m) for m in masks_a]
        has_b = z_full_b is not None and masks_b is not None and len(masks_b) > 0
        if has_b:
            zf_b = self._norm(z_full_b)
            masks_b = [self._norm(m) for m in masks_b]

        # work on a detached snapshot: the EMA update below mutates the buffer
        # in-place *before* backward, which would otherwise corrupt the graph
        prot = self.prototypes.detach().clone()

        # balanced soft codes of the *complete* views (pseudo-labels)
        q_a = self._codes(zf_a, prot)
        if has_b:
            q_b = self._codes(zf_b, prot)

        # Keep the true codes for prototype/queue updates and diagnostics.
        # The shuffled condition only breaks the sample-to-profile pairing
        # seen by the masked-view predictor.
        pred_q_a = self._shuffle_target_codes(q_a) if self.shuffle_targets else q_a
        pred_q_b = self._shuffle_target_codes(q_b) if self.shuffle_targets and has_b else q_b

        losses = []
        accs = []
        kls = []
        prediction_usage_entropies = []
        # self-alignment: masked view -> code of its own complete view
        pred_out = self._predict(masks_a, pred_q_a, prot)
        losses.append(pred_out["loss"])
        accs.append(pred_out["acc"])
        kls.append(pred_out["kl"])
        prediction_usage_entropies.append(pred_out["usage_entropy"])
        if has_b:
            pred_out = self._predict(masks_b, pred_q_b, prot)
            losses.append(pred_out["loss"])
            accs.append(pred_out["acc"])
            kls.append(pred_out["kl"])
            prediction_usage_entropies.append(pred_out["usage_entropy"])
        # cross-alignment: profiles should be augmentation/modality-agnostic
        if cross and has_b:
            pred_out = self._predict(masks_a, pred_q_b, prot)
            losses.append(pred_out["loss"])
            accs.append(pred_out["acc"])
            kls.append(pred_out["kl"])
            prediction_usage_entropies.append(pred_out["usage_entropy"])
            pred_out = self._predict(masks_b, pred_q_a, prot)
            losses.append(pred_out["loss"])
            accs.append(pred_out["acc"])
            kls.append(pred_out["kl"])
            prediction_usage_entropies.append(pred_out["usage_entropy"])

        # EMA update of prototypes with the complete views of this batch
        # (only during training; validation must not pollute the prototype bank)
        with torch.no_grad():
            if self.training:
                update_features = zf_a if not has_b else torch.cat([zf_a, zf_b], dim=0)
                update_codes = q_a if not has_b else torch.cat([q_a, q_b], dim=0)
                self._update(update_features, update_codes)
                if self.queue_size > 0:
                    self._enqueue(update_features)

        loss = torch.stack(losses).mean()
        acc = torch.stack(accs).mean()
        kl = torch.stack(kls).mean()
        usage = torch.cat([q_a, q_b], dim=0) if has_b else q_a
        usage = usage.mean(dim=0).clamp_min(1e-12)
        usage_entropy = -(usage * usage.log()).sum() / torch.log(
            torch.tensor(float(self.num_prototypes), device=usage.device))
        active = (usage > (1.0 / self.num_prototypes) * 0.1).float().sum()
        target_codes = torch.cat([q_a, q_b], dim=0) if has_b else q_a
        target_entropy = -(target_codes.clamp_min(1e-12) *
                           target_codes.clamp_min(1e-12).log()).sum(dim=-1).mean()
        target_entropy = target_entropy / torch.log(
            torch.tensor(float(self.num_prototypes), device=target_codes.device))
        target_confidence = target_codes.max(dim=-1).values.mean()
        prediction_usage_entropy = torch.stack(prediction_usage_entropies).mean()
        prototype_similarity = prot @ prot.t()
        off_diagonal = ~torch.eye(
            self.num_prototypes, dtype=torch.bool, device=prototype_similarity.device)
        prototype_mean_abs_cosine = prototype_similarity[off_diagonal].abs().mean()
        return {"loss": loss, "acc": acc, "kl": kl,
                "usage_entropy": usage_entropy,
                "active_prototypes": active,
                "target_entropy": target_entropy,
                "target_confidence": target_confidence,
                "prediction_usage_entropy": prediction_usage_entropy,
                "prototype_mean_abs_cosine": prototype_mean_abs_cosine}

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _norm(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:  # (T, B, dim) -> reshape as (T*B, dim)
            z = z.reshape(-1, z.shape[-1])
        z = z.to(self.prototypes.device, self.prototypes.dtype)
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def _codes(self, z: torch.Tensor, prot: torch.Tensor) -> torch.Tensor:
        """Balanced (equipartition) soft codes in [0,1] with shape (B, K)."""
        B = z.shape[0]
        # Use recent complete views as additional anchors when available.
        # Only the last B rows are returned as pseudo-labels for this batch.
        source = z
        if self.queue_size > 0 and int(self.queue_len.item()) > 0:
            source = torch.cat([self.feature_queue[:int(self.queue_len.item())], z], dim=0)
        sims = source @ prot.t()                # (queue+B, K) cosine sims
        Q = torch.exp(sims / self.sinkhorn_eps)
        eps = 1e-9
        for _ in range(self.sinkhorn_iters):
            Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(eps)
            Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(eps)
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(eps)
        if not torch.isfinite(Q).all():
            # fall back to uniform codes instead of NaN-poisoning the loss
            Q = torch.full((source.shape[0], self.num_prototypes), 1.0 / self.num_prototypes,
                           device=z.device)
        return Q[-B:].detach()

    def _shuffle_target_codes(self, q: torch.Tensor) -> torch.Tensor:
        """Break sample-to-profile pairing without changing the profile set.

        A fixed cyclic shift avoids consuming another random stream, so the
        shuffled condition remains paired with the normal condition under
        deterministic training. Batch size one cannot be shuffled and is
        returned unchanged.
        """
        if q.shape[0] <= 1:
            return q
        return q.roll(shifts=1, dims=0)

    def _predict(self, masks: List[torch.Tensor],
                 q: torch.Tensor,
                 prot: torch.Tensor) -> dict:
        """KL( q || p(masked view over prototypes) ), averaged over views."""
        logits = torch.stack(masks, dim=0) @ prot.t()               # (T,B,K)
        log_p = F.log_softmax(logits / self.temperature, dim=-1)
        p = log_p.exp()
        kl = (q.unsqueeze(0) * (q.clamp_min(1e-12).log().unsqueeze(0) - log_p)).sum(dim=-1).mean()
        loss = -(q.unsqueeze(0) * log_p).sum(dim=-1).mean()         # KL + H(q)
        pred = logits.argmax(dim=-1)                                 # (T,B)
        acc = (pred == q.argmax(dim=-1).unsqueeze(0)).float().mean()
        prediction_usage = p.mean(dim=(0, 1)).clamp_min(1e-12)
        usage_entropy = -(prediction_usage * prediction_usage.log()).sum()
        usage_entropy = usage_entropy / torch.log(
            torch.tensor(float(self.num_prototypes), device=prediction_usage.device))
        return {"loss": loss, "kl": kl, "acc": acc,
                "usage_entropy": usage_entropy}

    @torch.no_grad()
    def _update(self, z: torch.Tensor, q: torch.Tensor) -> None:
        """Update prototypes with EMA of assignment-weighted batch centroids."""
        mass = q.sum(dim=0)
        active = mass > 1e-12
        centroids = self.prototypes.clone()
        if active.any():
            assigned_sum = q.t() @ z
            centroids[active] = assigned_sum[active] / mass[active].unsqueeze(1)
            centroids[active] = F.normalize(centroids[active], p=2, dim=1)
        new_prot = ((1.0 - self.update_rate) * self.prototypes +
                    self.update_rate * centroids)
        self.prototypes.copy_(F.normalize(new_prot, p=2, dim=1))

    @torch.no_grad()
    def _enqueue(self, z: torch.Tensor) -> None:
        """Append complete-view features to the cross-batch queue."""
        if self.queue_size <= 0 or z.numel() == 0:
            return
        z = z.detach()
        if z.shape[0] >= self.queue_size:
            self.feature_queue.copy_(z[-self.queue_size:])
            self.queue_ptr.zero_()
            self.queue_len.fill_(self.queue_size)
            return
        ptr = int(self.queue_ptr.item())
        n = z.shape[0]
        end = ptr + n
        if end <= self.queue_size:
            self.feature_queue[ptr:end].copy_(z)
        else:
            first = self.queue_size - ptr
            self.feature_queue[ptr:].copy_(z[:first])
            self.feature_queue[:end - self.queue_size].copy_(z[first:])
        self.queue_ptr.fill_(end % self.queue_size)
        self.queue_len.fill_(min(self.queue_size, int(self.queue_len.item()) + n))
