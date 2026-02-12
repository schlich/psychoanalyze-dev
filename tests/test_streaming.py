"""Tests for the streaming Bayesian update engine."""

import numpy as np

from psychoanalyze.analysis import streaming
from psychoanalyze.types import InferenceMethod, StreamingConfig, StreamingState


class TestInitialize:
    def test_smc_initialization_creates_particles(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.SMC,
            n_particles=50,
            random_seed=42,
        )
        state = streaming.initialize(config)
        assert state.particles.shape == (50, 2)
        assert len(state.log_weights) == 50
        assert state.step == 0

    def test_vi_initialization_sets_prior_moments(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.ADVI,
            random_seed=42,
        )
        state = streaming.initialize(config)
        assert state.vi_means is not None
        assert len(state.vi_means) == 2
        assert state.step == 0

    def test_initial_ess_equals_n_particles(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.SMC,
            n_particles=100,
            random_seed=0,
        )
        state = streaming.initialize(config)
        assert state.effective_sample_size == 100.0


class TestSMCUpdate:
    def test_single_update_increments_step(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.SMC,
            n_particles=50,
            random_seed=42,
        )
        state = streaming.initialize(config)
        new_state = streaming.update(state, intensity=0.5, response=1, config=config)
        assert new_state.step == 1

    def test_ess_tracked_across_updates(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.SMC,
            n_particles=100,
            random_seed=42,
        )
        state = streaming.initialize(config)
        state = streaming.update(state, intensity=0.5, response=1, config=config)
        state = streaming.update(state, intensity=-0.5, response=0, config=config)
        assert len(state.ess_history) >= 2

    def test_multiple_updates_shift_threshold_estimate(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.SMC,
            n_particles=500,
            random_seed=42,
        )
        state = streaming.initialize(config)
        initial_threshold = state.threshold_estimate

        # Feed consistent data: hits above 1.0, misses below 1.0
        for _ in range(20):
            state = streaming.update(state, intensity=2.0, response=1, config=config)
            state = streaming.update(state, intensity=-1.0, response=0, config=config)

        assert state.threshold_estimate != initial_threshold

    def test_particles_remain_finite(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.SMC,
            n_particles=50,
            random_seed=7,
        )
        state = streaming.initialize(config)
        for i in range(10):
            state = streaming.update(
                state, intensity=float(i) / 5, response=int(i > 4), config=config
            )
        assert np.all(np.isfinite(state.particles))


class TestVIUpdate:
    def test_vi_update_increments_step(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.ADVI,
            random_seed=42,
        )
        state = streaming.initialize(config)
        new_state = streaming.update(state, intensity=0.5, response=1, config=config)
        assert new_state.step == 1

    def test_vi_means_remain_finite(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.ADVI,
            random_seed=99,
            vi_learning_rate=0.001,
        )
        state = streaming.initialize(config)
        for i in range(10):
            state = streaming.update(
                state, intensity=float(i) / 5, response=int(i > 4), config=config
            )
        assert np.all(np.isfinite(state.vi_means))
        assert np.all(np.isfinite(state.vi_stds))


class TestSummarize:
    def test_summarize_returns_expected_keys(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.SMC,
            n_particles=50,
            random_seed=42,
        )
        state = streaming.initialize(config)
        summary = streaming.summarize(state)
        assert set(summary.keys()) == {"threshold", "slope", "ess", "step"}

    def test_summarize_values_are_finite(self) -> None:
        config = StreamingConfig(
            method=InferenceMethod.SMC,
            n_particles=50,
            random_seed=42,
        )
        state = streaming.initialize(config)
        state = streaming.update(state, intensity=0.0, response=1, config=config)
        summary = streaming.summarize(state)
        for v in summary.values():
            assert np.isfinite(v)


class TestStreamingState:
    def test_empty_state_ess_is_zero(self) -> None:
        state = StreamingState()
        assert state.effective_sample_size == 0.0

    def test_threshold_estimate_from_vi(self) -> None:
        state = StreamingState(
            method=InferenceMethod.ADVI,
            vi_means=np.array([2.5, 1.0]),
        )
        assert state.threshold_estimate == 2.5

    def test_slope_estimate_from_vi(self) -> None:
        state = StreamingState(
            method=InferenceMethod.ADVI,
            vi_means=np.array([2.5, 3.0]),
        )
        assert state.slope_estimate == 3.0
