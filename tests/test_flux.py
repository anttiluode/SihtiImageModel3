"""R4 tests: true directional channels, no hidden-data leakage and continuous solve."""
import numpy as np
from data import train_set, held_out, damage
from flux import (
    FluxMedium, F_OPERATOR, image_features, observed_features,
    sieve_residues,
)


def fitted():
    return FluxMedium.fit(train_set(48))


def test_forward_flux_operator_exact():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(32,32))
    np.testing.assert_allclose(F_OPERATOR@x.ravel(), image_features(x),atol=1e-12)


def test_same_mean_different_directions():
    vertical=np.zeros((32,32));vertical[:,16:]=1.
    horizontal=vertical.T.copy()
    a=image_features(vertical);b=image_features(horizontal)
    np.testing.assert_allclose(a[:64].mean(),b[:64].mean())
    assert np.linalg.norm(a[64:128])>np.linalg.norm(b[64:128])
    assert np.linalg.norm(b[128:])>np.linalg.norm(a[128:])


def test_anti_parallel_gradient_cancellation_is_explicit():
    x=np.zeros((32,32))
    x[:,1::4]=1.;x[:,2::4]=1.
    f=image_features(x)
    assert np.max(np.abs(f[64:]))<1e-12
    assert np.max(np.abs(f[:64]-.5))<1e-12


def test_mask_gradient_needs_two_visible_endpoints():
    x=np.tile(np.arange(32,dtype=float),(32,1))/31.
    mask=np.ones((32,32),bool)
    mask[:,14:19]=False
    a,cover=observed_features(x,mask)
    corrupted=x.copy();corrupted[~mask]=10000.
    b,other=observed_features(corrupted,mask)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(cover,other)
    assert (cover[64:] == 0).any()


def test_gaussian_prediction_cannot_read_masked_pixels():
    model=fitted()
    target=held_out(1,seed=431)[0]
    obs,mask=damage(target,seed=55)
    other=obs.copy();other[~mask]=-9999.
    a,cover=model.predict(obs,mask)
    b,cover2=model.predict(other,mask)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(cover,cover2)


def test_continuous_solver_clamps_and_nonzero_gradient_message_effect():
    model=fitted()
    img=held_out(1,seed=432)[0]
    obs,mask=damage(img,seed=45)
    p,_=model.predict(obs,mask)
    smooth=model.solve(obs,mask,p,"smooth")
    both=model.solve(obs,mask,p,"both")
    for out in (smooth,both):
        np.testing.assert_allclose(out[mask],obs[mask],atol=1e-12)
        assert np.isfinite(out).all()
    modified=p.copy()
    modified[64+27]+=.30
    modified[128+35]-=.20
    changed=model.solve(obs,mask,modified,"both")
    assert np.max(np.abs(changed[~mask]-both[~mask]))>1e-7


def test_gradient_pulse_has_separate_differential_echo():
    model=fitted()
    img=held_out(1,seed=433)[0]
    obs,mask=damage(img,seed=56)
    p,_=model.predict(obs,mask)
    echo=model.response(obs,mask,p)
    np.testing.assert_allclose(echo[mask],0,atol=1e-12)
    assert np.linalg.norm(echo[~mask])>0


def test_original_continuous_sihti_decomposition_reconstructs():
    x=held_out(1,seed=435)[0]
    bands=sieve_residues(x)
    np.testing.assert_allclose(sum(bands),x,atol=1e-12)
