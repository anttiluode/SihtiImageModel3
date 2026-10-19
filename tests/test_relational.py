"""Energy consistency, two-level coupling, and observation-only inference."""
import numpy as np
from data import train_set, held_out, damage
from relational import RelationalMedium


def sample():
    model = RelationalMedium.fit(train_set(48), codes=16)
    target = held_out(1, 71)[0]
    obs, mask = damage(target, 721)
    return model, obs, mask


def test_learned_directional_adjacency():
    model, _, _ = sample()
    assert model.cooc_h.shape == (16, 16)
    assert np.var(model.cooc_h) > 0
    assert np.var(model.cooc_v) > 0
    assert not np.array_equal(model.cooc_h, model.cooc_v)


def test_local_update_matches_exact_joint_energy():
    model, obs, mask = sample()
    z = np.random.default_rng(10).integers(model.k, size=64)
    c = model._optimal_coarse(model._coarse_shape(z))
    factors = model._factors(obs, mask)
    ix = 33
    costs = model._site_cost(ix, z, c, factors)
    energies = []
    for k in range(model.k):
        trial = z.copy()
        trial[ix] = k
        energies.append(model.energy(trial, c, factors))
    np.testing.assert_allclose(np.asarray(energies)-energies[0], costs-costs[0], atol=1e-9)


def test_energy_nonincreasing_and_visible_pixels_clamped():
    model, obs, mask = sample()
    state = model.infer(obs, mask, sweeps=8)
    assert np.max(np.diff(state["energy"])) <= 1e-8
    assert state["frames"].shape == (9, 32, 32)
    np.testing.assert_allclose(state["image"][mask], obs[mask])


def test_hidden_data_never_used():
    model, obs, mask = sample()
    fake = obs.copy()
    fake[~mask] = 1e5
    a = model.infer(obs, mask, sweeps=4)
    b = model.infer(fake, mask, sweeps=4)
    np.testing.assert_allclose(a["image"], b["image"])


def test_frozen_and_live_share_initial_conditions():
    model, obs, mask = sample()
    live = model.infer(obs, mask, sweeps=0, feedback=True)
    frozen = model.infer(obs, mask, sweeps=0, feedback=False)
    np.testing.assert_allclose(live["image"], frozen["image"])
    np.testing.assert_allclose(live["energy"], frozen["energy"])


def test_ping_and_sham_same_budget_and_different_geometry():
    model, _, _ = sample()
    c = model.coarse_mean.copy()
    ping = model._probe_field(c, seed=17)
    sham = model._probe_field(c, seed=17, sham=True)
    np.testing.assert_allclose(np.linalg.norm(ping), np.linalg.norm(sham))
    assert np.linalg.norm(ping-sham) > 1e-4


def test_global_state_is_exact_minimizer():
    model, _, _ = sample()
    q = np.random.default_rng(77).random(64)
    c = model._optimal_coarse(q)
    grad = model.coarse_weight*(c-q) + model.prior_weight*model.coarse_precision@(c-model.coarse_mean)
    assert np.linalg.norm(grad) < 1e-6
