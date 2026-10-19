"""R3 evaluator-only oracle: independent truth paths, matched reader, ceiling."""
import numpy as np
from data import train_set, held_out, damage
from conditional import ConditionalBridge
from oracle_gate import (
    FixedCoarseDiagnostic, affected_cells, oracle_hidden_coarse,
    oracle_target_codes, true_coarse,
)


def fixture():
    bridge = ConditionalBridge.fit(train_set(48), codes=16)
    truth = held_out(1, seed=287)[0]
    obs, mask = damage(truth, seed=2870, kind="gap")
    return bridge, truth, obs, mask


def test_hidden_cells_exist_and_only_affected_coarse_oracle_is_overwritten():
    bridge, truth, obs, mask = fixture()
    prior = bridge.conditional_predict(obs, mask).ravel()
    oracle = oracle_hidden_coarse(prior, truth, mask)
    affected = affected_cells(mask)
    np.testing.assert_array_equal(prior[~affected], oracle[~affected])
    np.testing.assert_allclose(oracle[affected], true_coarse(truth)[affected])
    fractions = mask.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    assert np.sum(fractions == 0) > 0
    assert np.sum((fractions > 0) & (fractions < 1)) > 0


def test_normal_arm_does_not_read_missing_pixels_or_oracle_target():
    bridge, truth, obs, mask = fixture()
    diagnostic = FixedCoarseDiagnostic(bridge)
    fake_obs = obs.copy()
    fake_obs[~mask] = 123456.
    a = diagnostic.infer(obs, mask, steps=3)
    b = diagnostic.infer(fake_obs, mask, steps=3)
    np.testing.assert_array_equal(a["z"], b["z"])
    np.testing.assert_allclose(a["image"], b["image"])
    np.testing.assert_array_equal(a["coarse"], b["coarse"])
    assert a["kind"] == "visible_only"


def test_oracle_reads_missing_target_only_in_labeled_branch():
    bridge, truth, obs, mask = fixture()
    diagnostic = FixedCoarseDiagnostic(bridge)
    alternate = truth.copy()
    alternate[~mask] = 1.0
    original = diagnostic.infer(obs, mask, steps=2, oracle_target=truth)
    changed = diagnostic.infer(obs, mask, steps=2, oracle_target=alternate)
    learned = diagnostic.infer(obs, mask, steps=2)
    # Changing withheld truth can change the oracle coarse message, but
    # has no path into the ordinary observation-only reader.
    assert np.linalg.norm(original["coarse"]-changed["coarse"]) > 1e-6
    assert original["kind"] == changed["kind"] == "ORACLE_GT_COARSE"
    assert learned["kind"] == "visible_only"


def test_reader_matches_original_r2_frozen_exactly_at_strength_one():
    bridge, truth, obs, mask = fixture()
    original = bridge.run(obs, mask, mode="frozen", steps=5)
    diagnostic = FixedCoarseDiagnostic(bridge).infer(obs, mask, steps=5, strength=1.)
    np.testing.assert_array_equal(original["z"], diagnostic["z"])
    np.testing.assert_allclose(original["image"], diagnostic["image"])
    np.testing.assert_allclose(original["frames"], diagnostic["frames"])


def test_oracle_and_normal_same_codes_before_inference_all_strengths():
    bridge, truth, obs, mask = fixture()
    diagnostic = FixedCoarseDiagnostic(bridge)
    for strength in (1, 4, 16):
        normal = diagnostic.infer(obs, mask, strength=strength, steps=3)
        oracle = diagnostic.infer(obs, mask, strength=strength,
                                  oracle_target=truth, steps=3)
        np.testing.assert_array_equal(normal["initial_codes"], oracle["initial_codes"])
        np.testing.assert_allclose(normal["coarse"][~affected_cells(mask)],
                                   oracle["coarse"][~affected_cells(mask)])
        np.testing.assert_allclose(normal["image"][mask], obs[mask])
        np.testing.assert_allclose(oracle["image"][mask], obs[mask])
        assert len(normal["changes"]) == len(oracle["changes"]) == 3


def test_oracle_codebook_target_projections_are_evaluation_only():
    bridge, truth, obs, mask = fixture()
    image, codes = oracle_target_codes(bridge, truth, obs, mask)
    assert image.shape == (32, 32)
    assert codes.shape == (64,)
    np.testing.assert_allclose(image[mask], obs[mask])
    assert 0 <= codes.min() <= codes.max() < bridge.base.k
    assert np.isfinite(image).all()
