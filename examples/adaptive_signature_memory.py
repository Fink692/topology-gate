"""Synthetic diagnostic for adaptive rough-path memory."""

from __future__ import annotations

import json

import numpy as np

from topology_gate.signatures import AdaptiveSignatureMemory, SignatureMemoryConfig


def main() -> None:
    rng = np.random.default_rng(7)
    memory = AdaptiveSignatureMemory(
        SignatureMemoryConfig(
            input_dim=2,
            candidate_depths=(1, 2, 3),
            window=8,
            forgetting_factor=0.98,
            switching_cost=0.002,
            loss_clip=25.0,
        )
    )
    depths = []
    for step in range(180):
        path = [
            (float(rng.normal(scale=0.3)), float(rng.normal(scale=0.3)))
            for _ in range(8)
        ]
        signed_area = sum(path[index][0] * path[index + 1][1] for index in range(7))
        target = (0.4 * path[-1][0] if step < 90 else 0.2 * signed_area)
        target += float(rng.normal(scale=0.05))
        update = memory.observe(path, target)
        depths.append(update.next_depth)
    print(
        json.dumps(
            {
                "kind": "adaptive_signature_memory_synthetic",
                "steps": 180,
                "change": 90,
                "candidate_depths": list(memory.candidate_depths),
                "initial_depth": depths[0],
                "final_depth": depths[-1],
                "depth_counts": {str(depth): depths.count(depth) for depth in memory.candidate_depths},
                "note": "rough-path memory control diagnostic; not market evidence",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

