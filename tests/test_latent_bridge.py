"""R5 exact linear algebra, leakage and small end-to-end checks."""
import numpy as np
import pytest
from latent_bridge import LatentPCADecoder
from data import train_set, held_out, damage
from flux import FluxMedium
from flux_demo import evaluate


def toy():
    # Real orthonormal PCA-style rows, with four named, interpretable pixels.
    b = np.zeros((4,1024))
    b[np.arange(4), np.arange(4)] = 1.
    return LatentPCADecoder(np.zeros(1024), b)


def mask12():
    m = np.ones((32,32), bool)
    m.ravel()[:12] = False
    return m


def test_decoder_jacobian_matches_finite_difference():
    d = toy()
    z = np.array([.15, .26, .31, .42])
    direction = np.array([.5,-.2,.3,.1])
    eps=1e-6
    numeric = (d.decode(z + eps*direction)-d.decode(z))/eps
    predicted = (d.basis.T @ direction).reshape(32,32)
    assert np.allclose(numeric, predicted, atol=1e-9)


def test_one_pullback_step_reads_hidden_jacobian_transpose():
    d,mask = toy(),mask12()
    obs=np.zeros((32,32))
    target=np.zeros((32,32))
    target.ravel()[:4]=[.2,.3,.4,.5]
    out=d.correction(obs,mask,target,steps=1)
    assert np.allclose(out.ravel()[:4],target.ravel()[:4])
    assert np.array_equal(out[mask],obs[mask])


def test_unreachable_signal_has_zero_latent_pullback():
    d,m=toy(),mask12()
    obs=np.zeros((32,32))
    cue=np.zeros((32,32))
    cue.ravel()[8]=.8  # fifth hidden pixel, outside span of basis rows
    out=d.correction(obs,m,cue,steps=16)
    assert np.allclose(out,0)
    assert d.unreachable_fraction(cue,obs,m)==pytest.approx(1.)


def test_oracle_is_expressivity_ceiling_on_hidden_pixels():
    d,m=toy(),mask12()
    obs=np.zeros((32,32))
    truth=np.zeros((32,32))
    truth.ravel()[:12]=np.linspace(.05,.95,12)
    oracle=d.oracle_best(truth,obs,m)
    assert np.sum((oracle[~m]-truth[~m])**2) <= np.sum(truth[~m]**2)
    assert np.array_equal(oracle[m],obs[m])


def test_preconditioner_reduces_matching_loss_compared_with_one_step():
    d,m=toy(),mask12()
    obs=np.zeros((32,32))
    cue=np.zeros((32,32));cue.ravel()[:4]=.5
    a=d.correction(obs,m,cue,steps=1)
    b=d.correction(obs,m,cue,method="preconditioned",ridge=.02)
    assert np.sum((a[~m]-cue[~m])**2) <= 1e-12
    assert np.sum((b[~m]-cue[~m])**2) < .002


def test_all_ordinary_methods_ignore_hidden_placeholder_values():
    d,m=toy(),mask12()
    obs=np.zeros((32,32))
    poisoned=obs.copy()
    poisoned[~m]=999.99
    cue=np.zeros((32,32));cue.ravel()[:4]=.4
    for method in ("pullback","preconditioned"):
        a=d.correction(obs,m,cue,method=method,steps=16)
        b=d.correction(poisoned,m,cue,method=method,steps=16)
        assert np.array_equal(a,b)
    assert np.allclose(d.latent_from_visible(obs,m),
                       d.latent_from_visible(poisoned,m))


def test_shuffled_arm_deterministic_and_visible_clamped():
    d,m=toy(),mask12()
    obs=np.zeros((32,32));obs[m]=.12
    cue=np.full((32,32),.3)
    a=d.shuffled_correction(obs,m,cue,seed=12)
    b=d.shuffled_correction(obs,m,cue,seed=12)
    assert np.array_equal(a,b)
    assert np.array_equal(a[m],obs[m])


def test_small_real_training_and_flux_pipeline():
    train=train_set(36,seed=0)
    held=held_out(1,seed=569)[0]
    obs,m=damage(held,seed=5690,kind="gap")
    d=LatentPCADecoder.fit(train,rank=8)
    f=FluxMedium.fit(train)
    msg,_=f.predict(obs,m)
    cue=f.solve(obs,m,msg)
    z0=d.latent_from_visible(obs,m)
    arms=[
        d.compose(z0,obs,m),
        d.correction(obs,m,cue,steps=1),
        d.correction(obs,m,cue,steps=16),
        d.correction(obs,m,cue,method="preconditioned"),
        d.oracle_best(held,obs,m)
    ]
    for out in arms:
        assert out.shape==(32,32)
        assert np.all(np.isfinite(out))
        assert np.array_equal(out[m],obs[m])
    assert 0<=d.unreachable_fraction(held,obs,m)<=1.00000001
    assert evaluate(cue,held,m)["mse"]>=0
