"""Small deterministic offline demonstration of the control layer."""

from topology_gate import (
    RLS,
    OnlineRunConfig,
    RLSConfig,
    RollingTopologyDetector,
    TopologyConfig,
    generate_synthetic_regimes,
    run_recursive_rls,
)


def main() -> None:
    dataset = generate_synthetic_regimes(
        n_steps=128,
        n_features=2,
        change_points=(64,),
        seed=11,
        label_delay=1,
    )
    detector = RollingTopologyDetector(
        TopologyConfig(
            embedding_dim=1,
            cloud_window=16,
            graph_neighbors=3,
            n_eigenvalues=2,
            min_points=6,
            calibration_window=32,
            calibration_min_periods=8,
        )
    )
    learner = RLS(
        RLSConfig(
            n_features=dataset.features.n_features,
            lambda_min=0.90,
            lambda_max=1.0,
        )
    )
    result = run_recursive_rls(
        dataset.features.values,
        dataset.labels.values,
        realized_returns=dataset.realized_returns,
        detector=detector,
        learner=learner,
        config=OnlineRunConfig(label_delay=1, transaction_cost_bps=2.0),
        shift_points=dataset.change_points,
    )
    print(result.metrics)


if __name__ == "__main__":
    main()
