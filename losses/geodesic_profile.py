"""Prototype-graph geodesic profile for UniGIR V3.

The graph is built from a detached prototype snapshot and refreshed at epoch
boundaries.  It is used only to construct the complete-view target profile
and the masked-view prediction logits; no graph operation receives gradient.
The class subclasses the V2 alignment implementation so EMA, queue,
Sinkhorn, KL alignment, and diagnostics stay identical unless the distance
mode is changed explicitly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from losses.prototype_alignment import PrototypeAlignment


class GeodesicPrototypeAlignment(PrototypeAlignment):
    """UniGIR V3 profile alignment using a prototype k-NN graph."""

    def __init__(self, graph_k: int = 8, graph_anchors: int = 4,
                 graph_update_frequency: int = 1,
                 geodesic_temperature: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if graph_k < 1:
            raise ValueError("graph_k must be positive")
        if graph_anchors < 1:
            raise ValueError("graph_anchors must be positive")
        if graph_update_frequency < 1:
            raise ValueError("graph_update_frequency must be positive")
        if geodesic_temperature <= 0:
            raise ValueError("geodesic_temperature must be positive")

        self.graph_k = int(graph_k)
        self.graph_anchors = int(graph_anchors)
        self.graph_update_frequency = int(graph_update_frequency)
        self.geodesic_temperature = float(geodesic_temperature)
        self.current_epoch = 0
        self._graph_epoch = -1
        self.register_buffer(
            "graph_distances", torch.empty(0), persistent=False)
        self._graph_stats = {
            "graph_num_components": 0,
            "graph_unreachable_pairs": 0,
            "graph_mean_degree": 0.0,
            "graph_mean_edge_weight": 0.0,
        }

    def set_epoch(self, epoch: int) -> None:
        """Mark the epoch at which the next graph snapshot may be built."""
        epoch = int(epoch)
        if (self._graph_epoch < 0 or
                epoch - self._graph_epoch >= self.graph_update_frequency):
            self.current_epoch = epoch
            self._graph_epoch = -1
            self.graph_distances = torch.empty(
                0, device=self.prototypes.device,
                dtype=self.prototypes.dtype)

    @staticmethod
    @torch.no_grad()
    def _connected_components(adjacency: torch.Tensor) -> int:
        """Count components of a finite symmetric adjacency matrix."""
        finite = torch.isfinite(adjacency).cpu()
        count = 0
        unseen = set(range(finite.shape[0]))
        while unseen:
            count += 1
            stack = [unseen.pop()]
            while stack:
                node = stack.pop()
                neighbours = torch.nonzero(finite[node], as_tuple=False).flatten().tolist()
                for neighbour in neighbours:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        stack.append(neighbour)
        return count

    @torch.no_grad()
    def _build_graph(self, prototypes: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Build a symmetric k-NN graph and its all-pairs shortest paths."""
        prototypes = F.normalize(prototypes.detach(), p=2, dim=-1)
        num_prototypes = prototypes.shape[0]
        if num_prototypes < 2:
            raise ValueError("at least two prototypes are required")
        neighbours = min(self.graph_k, num_prototypes - 1)
        pairwise = torch.cdist(prototypes, prototypes, p=2)
        values, indices = torch.topk(
            pairwise, k=neighbours + 1, largest=False, dim=-1)
        values = values[:, 1:]
        indices = indices[:, 1:]

        inf = torch.tensor(float("inf"), device=pairwise.device, dtype=pairwise.dtype)
        adjacency = torch.full_like(pairwise, inf)
        row_ids = torch.arange(num_prototypes, device=pairwise.device).unsqueeze(1)
        adjacency[row_ids, indices] = values
        adjacency[indices, row_ids.expand_as(indices)] = values
        adjacency.fill_diagonal_(0.0)

        shortest = adjacency.clone()
        for pivot in range(num_prototypes):
            through_pivot = shortest[:, pivot:pivot + 1] + shortest[pivot:pivot + 1, :]
            shortest = torch.minimum(shortest, through_pivot)

        non_diagonal = ~torch.eye(
            num_prototypes, dtype=torch.bool, device=pairwise.device)
        finite_edges = torch.isfinite(adjacency) & non_diagonal
        finite_edge_weights = adjacency[finite_edges]
        stats = {
            "graph_num_components": self._connected_components(adjacency),
            "graph_unreachable_pairs": int(torch.isinf(shortest).sum().item()),
            "graph_mean_degree": float(finite_edges.float().sum(dim=1).mean().item()),
            "graph_mean_edge_weight": (
                float(finite_edge_weights.mean().item())
                if finite_edge_weights.numel() else 0.0
            ),
        }
        return shortest, stats

    @torch.no_grad()
    def _ensure_graph(self, prototypes: torch.Tensor) -> None:
        if self.graph_distances.numel() == 0 or self._graph_epoch < 0:
            self.graph_distances, self._graph_stats = self._build_graph(prototypes)
            self._graph_epoch = self.current_epoch

    def _geodesic_distances(self, values: torch.Tensor,
                            prototypes: torch.Tensor) -> torch.Tensor:
        self._ensure_graph(prototypes)
        anchors = min(self.graph_anchors, prototypes.shape[0])
        local_distances = torch.cdist(values, prototypes, p=2)
        anchor_distances, anchor_ids = torch.topk(
            local_distances, k=anchors, largest=False, dim=-1)
        graph_candidates = self.graph_distances[anchor_ids]
        return (anchor_distances.unsqueeze(-1) + graph_candidates).min(dim=1).values

    def _codes(self, z: torch.Tensor, prot: torch.Tensor) -> torch.Tensor:
        """Balanced codes from negative approximate geodesic distances."""
        batch_size = z.shape[0]
        source = z
        if self.queue_size > 0 and int(self.queue_len.item()) > 0:
            source = torch.cat([
                self.feature_queue[:int(self.queue_len.item())], z], dim=0)
        distances = self._geodesic_distances(source, prot)
        logits = -distances / self.geodesic_temperature
        Q = torch.exp(logits / self.sinkhorn_eps)
        eps = 1e-9
        for _ in range(self.sinkhorn_iters):
            Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(eps)
            Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(eps)
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(eps)
        if not torch.isfinite(Q).all():
            Q = torch.full(
                (source.shape[0], self.num_prototypes),
                1.0 / self.num_prototypes,
                device=z.device,
                dtype=z.dtype,
            )
        return Q[-batch_size:].detach()

    def _predict(self, masks, q: torch.Tensor, prot: torch.Tensor) -> dict:
        """Predict a geodesic profile from each masked view."""
        mask_values = torch.stack(masks, dim=0)
        num_masks, batch_size = mask_values.shape[:2]
        distances = self._geodesic_distances(
            mask_values.reshape(-1, mask_values.shape[-1]), prot)
        logits = -distances.reshape(num_masks, batch_size, -1)
        log_p = F.log_softmax(logits / self.geodesic_temperature, dim=-1)
        p = log_p.exp()
        kl = (
            q.unsqueeze(0)
            * (q.clamp_min(1e-12).log().unsqueeze(0) - log_p)
        ).sum(dim=-1).mean()
        loss = -(q.unsqueeze(0) * log_p).sum(dim=-1).mean()
        pred = logits.argmax(dim=-1)
        acc = (pred == q.argmax(dim=-1).unsqueeze(0)).float().mean()
        prediction_usage = p.mean(dim=(0, 1)).clamp_min(1e-12)
        usage_entropy = -(prediction_usage * prediction_usage.log()).sum()
        return {
            "loss": loss,
            "kl": kl,
            "acc": acc,
            "usage_entropy": usage_entropy / torch.log(
                torch.tensor(float(self.num_prototypes), device=p.device)),
        }

    def forward(self, *args, **kwargs) -> Optional[dict]:
        output = super().forward(*args, **kwargs)
        if output is not None:
            output.update({
                key: torch.tensor(value, dtype=self.prototypes.dtype,
                                  device=self.prototypes.device)
                for key, value in self._graph_stats.items()
            })
        return output
