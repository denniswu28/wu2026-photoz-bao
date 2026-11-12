"""Counting the random pairs once must give what counting them twice gave.

The random-random term is the most expensive part of a correlation-function measurement -- the
randoms outnumber the galaxies fifty to one and pair counting goes as their square -- and the pre-
and post-reconstruction measurements hand the counter identical positions and identical weights.
Reusing that count is an arithmetic identity.

It is not a *bitwise* identity, and that was measured rather than assumed. On a real cell the two
calls agree to 7.4e-16 relative -- about three machine epsilons on 3.8e10 pairs -- because the post
call additionally counts the shifted terms, which changes the order the sums accumulate in. A
repeated identical call *is* bitwise reproducible, at one thread and at sixteen; two differently
shaped calls are not. So these tests compare to floating-point tolerance where the two routes
genuinely differ in rounding, and demand exact equality only where exactness is available: the
reused counter is a copy of the pre-reconstruction one, so it equals it to the bit.

That also makes reuse the more reproducible of the two routes, not merely the cheaper one: it
replaces a differently-rounded recount with the same numbers.
"""

import numpy as np
import pytest

from pzbao.clustering import compute_xi

pytest.importorskip("pycorr")
pytest.importorskip("Corrfunc")

S_EDGES = np.arange(0.0, 60.0, 6.0)
MU_BINS = 10
IDENTITY = "SOURCE_ID"


def _catalog(rows, *, seed, shifted=True):
    """A small catalogue carrying both the original and the reconstructed positions."""
    generator = np.random.default_rng(seed)
    base = generator.uniform(-40.0, 40.0, size=(rows, 3)) + np.array([600.0, 0.0, 0.0])
    catalog = {
        IDENTITY: np.arange(rows, dtype=np.uint64) + 1,
        "X": base[:, 0],
        "Y": base[:, 1],
        "Z_CART": base[:, 2],
        "WEIGHT": np.ones(rows),
        "WEIGHT_FKP": np.ones(rows),
    }
    if shifted:
        nudge = generator.normal(0.0, 0.5, size=(rows, 3))
        catalog["X_SHIFTED"] = base[:, 0] + nudge[:, 0]
        catalog["Y_SHIFTED"] = base[:, 1] + nudge[:, 1]
        catalog["Z_CART_SHIFTED"] = base[:, 2] + nudge[:, 2]
    return catalog


def _measure(data, randoms, *, reconstructed, precomputed=None):
    return compute_xi(
        data,
        randoms,
        reconstructed=reconstructed,
        s_edges_mpc_h=S_EDGES,
        mu_bins=MU_BINS,
        line_of_sight="midpoint",
        nthreads=1,
        declared_estimator="shifted_random" if reconstructed else "landy_szalay",
        weight_expression="WEIGHT*WEIGHT_FKP",
        precomputed_random_counts=precomputed,
    )


@pytest.fixture(scope="module")
def sample():
    return _catalog(300, seed=5), _catalog(3000, seed=11)


def _counter_state(counter):
    return {
        "wcounts": np.asarray(counter.wcounts, dtype=float).copy(),
        "wnorm": float(counter.wnorm),
        "sedges": np.asarray(counter.edges[0], dtype=float).copy(),
        "muedges": np.asarray(counter.edges[1], dtype=float).copy(),
    }


def test_freshly_counted_pre_and_post_random_counts_agree_as_arithmetic(sample):
    """The premise. If these disagreed beyond rounding, reuse would be wrong, not merely faster.

    Compared to tolerance rather than bitwise: the two calls have different shapes -- the post one
    also counts the shifted terms -- so their sums accumulate in different orders. Measured on a
    real cell the disagreement is 7.4e-16 relative, about three machine epsilons.
    """
    data, randoms = sample
    pre = _measure(data, randoms, reconstructed=False)
    post = _measure(data, randoms, reconstructed=True)
    before, after = _counter_state(pre.R1R2), _counter_state(post.R1R2)
    np.testing.assert_allclose(before["wcounts"], after["wcounts"], rtol=1e-12, atol=0.0)
    assert before["wnorm"] == after["wnorm"]
    # The grids are not arithmetic and must match exactly.
    np.testing.assert_array_equal(before["sedges"], after["sedges"])
    np.testing.assert_array_equal(before["muedges"], after["muedges"])


def test_the_reused_counts_equal_a_separately_recomputed_post_count(sample):
    """Equal to the recount as arithmetic, and to the count it was taken from exactly."""
    data, randoms = sample
    pre = _measure(data, randoms, reconstructed=False)
    recomputed = _measure(data, randoms, reconstructed=True)
    reused = _measure(data, randoms, reconstructed=True, precomputed=pre.R1R2)
    np.testing.assert_allclose(
        np.asarray(reused.R1R2.wcounts, dtype=float),
        np.asarray(recomputed.R1R2.wcounts, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )
    assert float(reused.R1R2.wnorm) == float(recomputed.R1R2.wnorm)
    # Exactness is available here and is the point: reuse carries the pre-reconstruction numbers
    # through unchanged rather than producing a third rounding of the same sum.
    np.testing.assert_array_equal(
        np.asarray(reused.R1R2.wcounts, dtype=float),
        np.asarray(pre.R1R2.wcounts, dtype=float),
    )


def test_the_post_reconstruction_correlation_function_is_unchanged(sample):
    """The measurement, not merely the counts it was built from."""
    data, randoms = sample
    pre = _measure(data, randoms, reconstructed=False)
    recomputed = _measure(data, randoms, reconstructed=True)
    reused = _measure(data, randoms, reconstructed=True, precomputed=pre.R1R2)
    for ell in (0, 2):
        s_recomputed, xi_recomputed = recomputed(ells=ell, return_sep=True)
        s_reused, xi_reused = reused(ells=ell, return_sep=True)
        np.testing.assert_allclose(s_recomputed, s_reused, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            np.nan_to_num(xi_recomputed), np.nan_to_num(xi_reused), rtol=1e-10, atol=1e-12
        )


def test_the_pre_reconstruction_correlation_function_is_unchanged(sample):
    """Reuse must not reach backwards into the measurement it borrowed from."""
    data, randoms = sample
    pre = _measure(data, randoms, reconstructed=False)
    s_before, xi_before = pre(ells=0, return_sep=True)
    _measure(data, randoms, reconstructed=True, precomputed=pre.R1R2)
    s_after, xi_after = pre(ells=0, return_sep=True)
    np.testing.assert_array_equal(s_before, s_after)
    np.testing.assert_array_equal(xi_before, xi_after)


def test_every_other_counter_is_still_measured(sample):
    """Only the random-random term is reused; the rest of the estimator is untouched."""
    data, randoms = sample
    pre = _measure(data, randoms, reconstructed=False)
    recomputed = _measure(data, randoms, reconstructed=True)
    reused = _measure(data, randoms, reconstructed=True, precomputed=pre.R1R2)
    for name in recomputed.count_names:
        left = getattr(recomputed, name, None)
        right = getattr(reused, name, None)
        assert (left is None) == (right is None), name
        if left is None:
            continue
        if name == "R1R2":
            continue  # the one term reuse replaces; compared on its own above
        np.testing.assert_array_equal(
            np.asarray(left.wcounts, dtype=float),
            np.asarray(right.wcounts, dtype=float),
            err_msg=name,
        )


def test_reuse_does_not_mutate_the_counter_it_was_given(sample):
    """The estimator takes ownership of what it is handed, and provenance is bound onto it.

    Sharing one counter object between the two products would let the second provenance binding
    overwrite the first, so the pre-reconstruction product would carry the post-reconstruction
    record. The reused counts are copied for that reason, and this pins it.
    """
    data, randoms = sample
    pre = _measure(data, randoms, reconstructed=False)
    source = pre.R1R2
    before = _counter_state(source)
    reused = _measure(data, randoms, reconstructed=True, precomputed=source)

    assert reused.R1R2 is not source, "the reused counter is the same object as the source"
    after = _counter_state(source)
    np.testing.assert_array_equal(before["wcounts"], after["wcounts"])
    assert before["wnorm"] == after["wnorm"]

    # Binding an attribute on the copy must not reach the original.
    reused.R1R2.attrs = dict(reused.R1R2.attrs)
    reused.R1R2.attrs["analysis_provenance"] = {"product_kind": "xi", "state": "post"}
    assert "analysis_provenance" not in getattr(source, "attrs", {})


def test_measuring_without_reuse_still_works(sample):
    """The parameter is optional, and omitting it is the behaviour that existed before."""
    data, randoms = sample
    measured = _measure(data, randoms, reconstructed=True)
    assert measured.R1R2 is not None
    assert np.all(np.isfinite(np.asarray(measured.R1R2.wcounts, dtype=float)))
