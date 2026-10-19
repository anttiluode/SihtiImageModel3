"""R2 tests: information direction, exact energy, observation leakage and baselines."""
import numpy as np
from data import train_set, held_out, damage
from relational import RelationalMedium
from conditional import ConditionalBridge, block_observations


def medium():
    return ConditionalBridge.fit(train_set(48), codes=16)


def scene():
    target = held_out(1, 215)[0]
    return damage(target, seed=216, kind="gap")


def test_partial_block_observations_ignore_hidden_placeholders():
    obs, mask = scene()
    mean = np.full(64, .5)
    other = obs.copy()
    other[~mask] = 957.
    b, fractions = block_observations(obs, mask, mean)
    b2, fractions2 = block_observations(other, mask, mean)
    np.testing.assert_allclose(b, b2, atol=0, rtol=0)
    np.testing.assert_array_equal(fractions, fractions2)
    assert (fractions == 0).any()
    assert (fractions == 1).any()


def test_exact_gaussian_conditional_prediction_positive_control():
    # An analytically controlled correlation: site 1 normally follows site 0.
    cov = np.eye(64)
    cov[0, 1] = cov[1, 0] = .96
    mean = np.full(64, .5)
    fake = RelationalMedium(
        codes=np.zeros((2, 49)), cooc_h=np.zeros((2, 2)),
        cooc_v=np.zeros((2, 2)), coarse_mean=mean,
        coarse_precision=np.linalg.inv(cov),
        exemplar_images=np.zeros((20, 32, 32)),
    )
    bridge = ConditionalBridge(fake)
    mask = np.zeros((32, 32), bool)
    mask[:4, :4] = True
    obs = np.full((32, 32), -100., float)
    obs[:4, :4] = .95
    predicted = bridge.conditional_predict(obs, mask).ravel()
    assert predicted[1] > .80, f"Correlated hidden cell failed to respond: {predicted[1]}"
    assert np.abs(predicted[2]-.5) < 1e-10
    corrupted = obs.copy()
    corrupted[~mask] = 999999.
    np.testing.assert_allclose(
        bridge.conditional_predict(corrupted, mask),
        bridge.conditional_predict(obs, mask),
        atol=1e-12,
    )


def test_masked_inference_is_invariant_to_unobserved_pixels():
    model = medium()
    obs, mask = scene()
    corrupted = obs.copy()
    corrupted[~mask] = 20.
    for mode in ("mean", "frozen", "live", "shuffled", "no_coarse"):
        x = model.run(obs, mask, mode=mode, steps=3)
        y = model.run(corrupted, mask, mode=mode, steps=3)
        np.testing.assert_array_equal(x["z"], y["z"])
        np.testing.assert_allclose(x["image"], y["image"])
        np.testing.assert_allclose(x["coarse_prediction"], y["coarse_prediction"])


def test_local_energy_matches_implemented_site_choice():
    model = medium()
    obs, mask = scene()
    ctx = model.context(obs, mask)
    factors = model._factors(obs, mask)
    rng = np.random.default_rng(7)
    z = rng.integers(model.base.k, size=64)
    c = model._solve_coarse(z, ctx)
    idx = 27
    costs = model._site_cost(idx, z, c, factors, ctx)
    energies = []
    for k in range(model.base.k):
        candidate = z.copy()
        candidate[idx] = k
        energies.append(model.energy(candidate, c, factors, ctx))
    np.testing.assert_allclose(np.asarray(energies)-energies[0],
                               costs-costs[0], atol=1e-8)


def test_coarse_solve_exact_stationarity_and_response_to_fine_message():
    model = medium()
    obs, mask = scene()
    ctx = model.context(obs, mask)
    z = np.zeros(64, int)
    c = model._solve_coarse(z, ctx)
    wup = model.coupling_strength*ctx["reliability"]
    grad = ctx["P"]@(c-model.coarse_mean) + ctx["w"]*(c-ctx["b"]) + wup*(c-model._q_codes()[z])
    assert np.linalg.norm(grad) < 1e-8
    z2 = z.copy()
    q = model._q_codes()
    z2[33] = int(np.argmax(np.abs(q-q[z[33]])))
    assert np.linalg.norm(model._solve_coarse(z2, ctx)-c) > 1e-7


def test_frozen_live_same_initialization_and_monotonic_energy():
    model = medium()
    obs, mask = scene()
    frozen = model.run(obs, mask, mode="frozen", steps=4)
    live = model.run(obs, mask, mode="live", steps=4)
    np.testing.assert_array_equal(frozen["frames"][0], live["frames"][0])
    np.testing.assert_array_equal(frozen["coarse_frames"][0], live["coarse_frames"][0])
    assert max(np.diff(frozen["energy"])) <= 1e-7
    assert max(np.diff(live["energy"])) <= 1e-7
    np.testing.assert_allclose(frozen["image"][mask], obs[mask])
    np.testing.assert_allclose(live["image"][mask], obs[mask])
    assert len(live["changes"]) == 4


def test_observation_conditioned_message_differs_from_unconditional():
    model = medium()
    obs, mask = scene()
    prior = model.coarse_mean
    conditioned = model.conditional_predict(obs, mask).ravel()
    assert np.max(np.abs(conditioned-prior)) > 1e-4
