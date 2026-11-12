"""Optional angular jackknife calculations retain cap-disjoint labels and unchanged RR reuse."""

import numpy as np
import pytest
from pzbao.clustering import assign_jackknife_regions, compute_xi

pytest.importorskip("pycorr")
pytest.importorskip("Corrfunc")

S_EDGES = np.arange(0.0, 60.0, 10.0)
MU_BINS = 8
IDENTITY = "SOURCE_ID"
REGIONS = 6


def _sky_catalog(rows, *, right_ascension, seed, first_identity, shifted=True):
    """A small catalogue on a patch of sky, in the Cartesian columns the pipeline uses."""
    generator = np.random.default_rng(seed)
    ra = np.radians(right_ascension + generator.uniform(-8.0, 8.0, rows))
    dec = np.radians(generator.uniform(-8.0, 8.0, rows))
    radius = generator.uniform(900.0, 1000.0, rows)
    catalog = {
        IDENTITY: np.arange(rows, dtype=np.uint64) + first_identity,
        "X": radius * np.cos(dec) * np.cos(ra),
        "Y": radius * np.cos(dec) * np.sin(ra),
        "Z_CART": radius * np.sin(dec),
        "WEIGHT": np.ones(rows),
        "WEIGHT_FKP": np.ones(rows),
    }
    if shifted:
        nudge = generator.normal(0.0, 0.4, size=(rows, 3))
        for index, name in enumerate(("X", "Y", "Z_CART")):
            catalog[f"{name}_SHIFTED"] = catalog[name] + nudge[:, index]
    return catalog


@pytest.fixture(scope="module")
def two_caps():
    """Two disjoint patches of sky, as NGC and SGC are."""
    return {
        "NGC": (
            _sky_catalog(400, right_ascension=150.0, seed=3, first_identity=1),
            _sky_catalog(2000, right_ascension=150.0, seed=4, first_identity=100_001),
        ),
        "SGC": (
            _sky_catalog(200, right_ascension=20.0, seed=5, first_identity=200_001),
            _sky_catalog(1000, right_ascension=20.0, seed=6, first_identity=300_001),
        ),
    }


def _assign(two_caps, region_count=REGIONS):
    return assign_jackknife_regions(
        {cap: data for cap, (data, _) in two_caps.items()},
        {cap: randoms for cap, (_, randoms) in two_caps.items()},
        region_count=region_count,
        position_columns=("X", "Y", "Z_CART"),
    )


def test_one_subsampler_over_both_caps_gives_each_cap_its_own_regions(two_caps):
    """The property that makes cap combination safe, asserted rather than assumed.

    Each cap must receive a disjoint set of region identifiers. If both were labelled 0..59
    independently, adding the two estimators would merge two unrelated patches of sky into every
    region-omitted realization -- a wrong jackknife covariance and therefore a wrong rescaling,
    with nothing raised anywhere.
    """
    assignment = _assign(two_caps)
    by_cap = assignment["regions_by_cap_index"]
    assert set(by_cap) == {"NGC", "SGC"}
    assert not set(by_cap["NGC"]) & set(by_cap["SGC"])
    assert sorted(by_cap["NGC"] + by_cap["SGC"]) == list(range(REGIONS))
    assert sum(assignment["occupancy"]) == 600
    assert assignment["empty_regions"] == []
    # Every random is labelled too, and inside the range.
    for cap, (_data, randoms) in two_caps.items():
        labels = assignment["regions_by_cap"][cap]["randoms"]
        assert labels.shape == (len(randoms[IDENTITY]),)
        assert labels.min() >= 0 and labels.max() < REGIONS
        assert assignment["regions_by_cap"][cap]["region_count"] == REGIONS


def test_the_assignment_is_angular_and_therefore_survives_reconstruction(two_caps):
    """Regions must not depend on radial position, or the two states would differ by more than
    the displacement.

    A radial dependence would let one realization's pre- and post-reconstruction jackknives omit
    different objects, which is not a difference reconstruction made.
    """
    assignment = _assign(two_caps)
    generator = np.random.default_rng(17)
    rescaled = {}
    for cap, (data, randoms) in two_caps.items():
        moved = dict(data)
        factor = generator.uniform(0.6, 1.5, len(data[IDENTITY]))
        for name in ("X", "Y", "Z_CART"):
            moved[name] = data[name] * factor
        rescaled[cap] = (moved, randoms)
    moved_assignment = _assign(rescaled)
    for cap in two_caps:
        assert np.array_equal(
            assignment["regions_by_cap"][cap]["data"],
            moved_assignment["regions_by_cap"][cap]["data"],
        )


def test_the_construction_is_deterministic_across_invocations(two_caps):
    """Two runs of the registered construction must label identically; the seed is RascalC's."""
    first, second = _assign(two_caps), _assign(two_caps)
    for cap in two_caps:
        for role in ("data", "randoms"):
            assert np.array_equal(
                first["regions_by_cap"][cap][role], second["regions_by_cap"][cap][role]
            )


def test_fewer_galaxies_than_regions_is_refused(two_caps):
    with pytest.raises(ValueError, match="fewer galaxies than jackknife regions"):
        _assign(two_caps, region_count=10_000)
    with pytest.raises(ValueError, match="at least two regions"):
        _assign(two_caps, region_count=1)


def _measure(data, randoms, regions, *, reconstructed, precomputed=None):
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
        jackknife_regions=regions,
    )


def test_each_cap_carries_every_region_so_the_caps_can_be_added(two_caps):
    """A cap holds only some regions, but its estimator must carry entries for all of them.

    Left to itself pycorr builds realizations only for the regions a cap actually contains, and
    the two caps' estimators then cannot be added at all -- their realization keys differ and the
    addition raises. The cap that lacks a region carries it zero-counted instead, which is also
    what makes the combined arithmetic right.
    """
    assignment = _assign(two_caps)
    data, randoms = two_caps["NGC"]
    regions = assignment["regions_by_cap"]["NGC"]
    estimator = _measure(data, randoms, regions, reconstructed=False)
    realizations = sorted(int(value) for value in estimator.realizations)
    assert realizations == list(range(REGIONS))
    absent = sorted(set(range(REGIONS)) - set(assignment["regions_by_cap_index"]["NGC"]))
    assert absent, "the fixture must give NGC only some of the regions"
    for index in absent:
        assert np.all(np.asarray(estimator.D1D2.auto[index].wcounts) == 0)
    # The correlation function itself is still the measurement, not a jackknife of it.
    plain = compute_xi(
        data,
        randoms,
        reconstructed=False,
        s_edges_mpc_h=S_EDGES,
        mu_bins=MU_BINS,
        line_of_sight="midpoint",
        nthreads=1,
        declared_estimator="landy_szalay",
        weight_expression="WEIGHT*WEIGHT_FKP",
    )
    assert np.allclose(
        np.asarray(estimator.D1D2.wcounts, dtype=float),
        np.asarray(plain.D1D2.wcounts, dtype=float),
    )


def test_omitting_a_region_removes_it_from_its_own_cap_and_leaves_the_other_whole(two_caps):
    """The definition a survey-wide jackknife needs, checked on the counts themselves.

    The measured correlation function is cap-combined, so a jackknife realization of it must omit
    one angular region of the union. Because the caps are disjoint on the sky, a region belongs to
    exactly one of them, and the realization that omits it is that cap without the region plus the
    other cap entire. This asserts that identity exactly rather than trusting that the addition
    did something sensible -- a wrong combination here is a wrong covariance with no symptom.
    """
    from pzbao.clustering import combine_cap_pair_counts

    assignment = _assign(two_caps)
    per_cap = {
        cap: _measure(data, randoms, assignment["regions_by_cap"][cap], reconstructed=False)
        for cap, (data, randoms) in two_caps.items()
    }
    combined = combine_cap_pair_counts(per_cap)
    assert sorted(int(value) for value in combined.realizations) == list(range(REGIONS))

    def _omitting(estimator, region):
        counts = estimator.D1D2
        return (
            np.asarray(counts.wcounts, dtype=float)
            - np.asarray(counts.auto[region].wcounts, dtype=float)
            - np.asarray(counts.cross12[region].wcounts, dtype=float)
            - np.asarray(counts.cross21[region].wcounts, dtype=float)
        )

    owner = {region: cap for cap in per_cap for region in assignment["regions_by_cap_index"][cap]}
    assert len(owner) == REGIONS
    for region, cap in owner.items():
        other = next(name for name in per_cap if name != cap)
        expected = _omitting(per_cap[cap], region) + np.asarray(
            per_cap[other].D1D2.wcounts, dtype=float
        )
        assert np.array_equal(_omitting(combined, region), expected)


def test_region_labels_that_do_not_cover_the_catalogue_are_refused(two_caps):
    assignment = _assign(two_caps)
    data, randoms = two_caps["NGC"]
    regions = dict(assignment["regions_by_cap"]["NGC"])
    with pytest.raises(ValueError, match="do not label every measured galaxy"):
        _measure(data, randoms, {**regions, "data": regions["data"][:-1]}, reconstructed=False)
    with pytest.raises(ValueError, match="do not label every measured random"):
        _measure(
            data, randoms, {**regions, "randoms": regions["randoms"][:-1]}, reconstructed=False
        )
    with pytest.raises(ValueError, match="outside the declared region count"):
        _measure(data, randoms, {**regions, "region_count": 2}, reconstructed=False)


def test_the_fit_vector_is_derived_the_same_way_from_either_estimator(two_caps):
    """A jackknife estimator must yield the same fit vector a plain one does.

    pycorr resolves ``return_std or return_std is None and hasattr(self, 'cov')``, so an estimator
    carrying a covariance -- which a jackknife one does -- silently returns a third value from the
    same call. ``xi_fit_data`` unpacks two, so the shot-noise route failed on real data at the
    point where it derives the vector it is about to integrate: after the catalogue was rebuilt and
    the jackknife correlation function measured.

    The vector is a property of the counts, not of which flavour of estimator carries them, so the
    two must agree exactly.
    """
    from pzbao.likelihood import xi_fit_data

    assignment = _assign(two_caps)
    data, randoms = two_caps["NGC"]
    jackknife = _measure(data, randoms, assignment["regions_by_cap"]["NGC"], reconstructed=False)
    plain = compute_xi(
        data,
        randoms,
        reconstructed=False,
        s_edges_mpc_h=S_EDGES,
        mu_bins=MU_BINS,
        line_of_sight="midpoint",
        nthreads=1,
        declared_estimator="landy_szalay",
        weight_expression="WEIGHT*WEIGHT_FKP",
    )
    assert hasattr(jackknife, "cov") and not hasattr(
        plain, "cov"
    ), "the fixture must contrast an estimator that carries a covariance with one that does not"

    arguments = dict(
        multipoles=(0,),
        rebin_factor=1,
        expected_bin_width_mpc_h=float(S_EDGES[1] - S_EDGES[0]),
        fit_edge_limits_mpc_h=(float(S_EDGES[0]), float(S_EDGES[-1])),
    )
    from_jack = xi_fit_data(jackknife, **arguments)
    from_plain = xi_fit_data(plain, **arguments)
    assert len(from_jack) == len(from_plain)
    for mine, theirs in zip(from_jack, from_plain):
        mine, theirs = np.asarray(mine, dtype=float), np.asarray(theirs, dtype=float)
        assert mine.shape == theirs.shape
        # Not exact equality. The separations are pair-weighted averages, so they carry the same
        # few-epsilon irreproducibility the campaign's own counts do -- measured at 2.2e-15 on BGS1
        # mock 6. A tolerance twelve orders of magnitude above that still separates this vector
        # from any other one.
        assert np.allclose(mine, theirs, rtol=1e-12, atol=0.0, equal_nan=True)
