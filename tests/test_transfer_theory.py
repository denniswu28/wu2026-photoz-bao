"""Analytic limits of the proposed local independent-source displacement model."""

import copy
import itertools

import numpy as np
import pytest

from pzbao.theory import (
    exclusive_class_source_cross_covariance,
    fixed_count_normalized_source_kernel,
    fixed_count_source_moments,
    independent_source_pair_covariance,
    independent_source_pair_kernel,
)


@pytest.mark.parametrize(
    "count,forced,excluded",
    [(0, (), ()), (4, (), ()), (2, (), ()), (3, (0, 2), ()), (1, (), (0, 2)), (2, (0,), (3,))],
)
def test_fixed_count_moments_match_all_assignments_and_independent_error_marks(
    count, forced, excluded
):
    means = np.array([[1, 2], [-3, 1], [2, -4], [5, 3]], dtype=float)
    # Unequal per-parent marks include a fixed contribution, as for a target
    # with its error conditioned on. Enumerate every assignment and +/- mark.
    errors = np.array([[0, 0], [0.5, 2], [1, -1], [-2, 0.25]])
    mark_covariance = np.einsum("ni,nj->nij", errors, errors)
    available = sorted(set(range(len(means))) - set(forced) - set(excluded))
    outcomes = []
    for selected in itertools.combinations(available, count - len(forced)):
        indices = list(forced) + list(selected)
        for signs in itertools.product((-1, 1), repeat=count):
            outcomes.append(
                (means[indices] + np.asarray(signs)[:, None] * errors[indices]).sum(axis=0)
            )
    outcomes = np.asarray(outcomes)
    expected_mean = outcomes.mean(axis=0)
    residuals = outcomes - expected_mean
    expected_covariance = residuals.T @ residuals / len(outcomes)
    before = means.copy(), mark_covariance.copy()
    actual_mean, actual_covariance = fixed_count_source_moments(
        means,
        count,
        contribution_covariances=mark_covariance,
        forced_indices=forced,
        excluded_indices=excluded,
    )
    np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-14)
    np.testing.assert_allclose(actual_covariance, expected_covariance, atol=1e-14)
    np.testing.assert_array_equal(means, before[0])
    np.testing.assert_array_equal(mark_covariance, before[1])


@pytest.mark.parametrize("parents,count", [(0, 0), (1, 0), (1, 1)])
def test_fixed_count_deterministic_empty_and_singleton_limits(parents, count):
    mean, covariance = fixed_count_source_moments(np.ones((parents, 3)), count)
    np.testing.assert_array_equal(mean, np.full(3, count))
    np.testing.assert_array_equal(covariance, np.zeros((3, 3)))


@pytest.mark.parametrize(
    "count,kwargs",
    [
        (True, {}),
        (1.5, {}),
        (-1, {}),
        (3, {}),
        (0, {"forced_indices": [0]}),
        (2, {"excluded_indices": [0]}),
        (1, {"forced_indices": [0], "excluded_indices": [0]}),
        (1, {"forced_indices": [0, 0]}),
        (1, {"excluded_indices": [2]}),
        (1, {"excluded_indices": [0.0]}),
        (1, {"contribution_covariances": np.full((2, 2, 2), np.nan)}),
        (1, {"contribution_covariances": -np.tile(np.eye(2), (2, 1, 1))}),
        (1, {"contribution_covariances": np.tile([[1, 1], [0, 1]], (2, 1, 1))}),
    ],
)
def test_fixed_count_rejects_invalid_conditioning_and_mark_covariances(count, kwargs):
    with pytest.raises(ValueError):
        fixed_count_source_moments(np.ones((2, 2)), count, **kwargs)


def inputs():
    # Three perpendicular modes give independent, explicitly soluble directions.
    return dict(
        wavevectors=np.eye(3),
        integration_weights=np.ones(3),
        linear_power=np.array([2.0, 3.0, 5.0]),
        source_noise_power=np.array([7.0, 11.0, 13.0]),
        source_response=np.ones(3),
        source_noise_filter=np.ones(3),
        pair_separation=np.full(3, np.pi),
        line_of_sight=np.array([0.0, 0.0, 1.0]),
        target_growth=0.5,
        reconstruction_growth=0.5,
    )


def test_perfect_signal_cancellation_leaves_source_noise_with_rsd_boost():
    values = inputs()
    result = independent_source_pair_covariance(**values, target_errors=[0, 0])
    np.testing.assert_allclose(result, np.diag([28, 44, 117]), atol=1e-12)
    values["source_noise_power"][:] = 0
    np.testing.assert_allclose(
        independent_source_pair_covariance(**values, target_errors=[0, 0]), 0, atol=1e-12
    )
    assert independent_source_pair_kernel([0.1, 0.2, 0.3], [0], [1], covariance_inputs=values) == 1


def test_no_shift_recovers_full_signal_gaussian_photoz_response():
    values = inputs()
    values["source_response"][:] = 0
    values["source_noise_filter"][:] = 0
    expected = np.diag([8, 12, 45])
    np.testing.assert_allclose(
        independent_source_pair_covariance(**values, target_errors=[8, -3]), expected
    )
    x, weights = np.polynomial.hermite.hermgauss(24)
    sigma_r = 2.0
    k = np.array([0.1, 0.2, 0.3])
    result = independent_source_pair_kernel(
        k, np.sqrt(2) * sigma_r * x, weights / np.sqrt(np.pi), covariance_inputs=values
    )
    np.testing.assert_allclose(
        result, np.exp(-k @ expected @ k / 2) * np.exp(-((k[2] * sigma_r) ** 2)), atol=1e-12
    )


def test_uncertain_shift_positions_do_not_cancel_perfect_source_signal():
    values = inputs()
    values["source_noise_power"][:] = 0
    before = copy.deepcopy(values)
    result = independent_source_pair_covariance(**values, target_errors=[np.pi, np.pi])
    np.testing.assert_allclose(result, np.diag([0, 0, 180]), atol=1e-12)
    for name in values:
        np.testing.assert_array_equal(values[name], before[name])


def test_error_averaged_tensor_matches_analytic_moment_and_kernel_symmetries():
    values = inputs()
    values["source_response"] = np.array([0.2, 0.4, 0.6])
    values["source_noise_filter"] = np.array([0.3, 0.2, 0.1])
    values["pair_separation"] = np.array([1.0, 2.0, 0.7])
    sigma = 0.7
    x, weights = np.polynomial.hermite.hermgauss(16)
    nodes, weights = np.sqrt(2) * sigma * x, weights / np.sqrt(np.pi)
    averaged = sum(
        wl * wr * independent_source_pair_covariance(**values, target_errors=[left, right])
        for left, wl in zip(nodes, weights)
        for right, wr in zip(nodes, weights)
    )
    response = values["source_response"]
    photo = np.array([1, 1, np.exp(-(sigma**2) / 2)])
    cosine = np.cos(values["pair_separation"])
    expected = (
        2
        * np.array([1, 1, 1.5**2])
        * (
            values["linear_power"]
            * (
                1
                - cosine
                - 2 * response * photo * (1 - cosine)
                + response**2 * (1 - photo**2 * cosine)
            )
            + values["source_noise_power"]
            * values["source_noise_filter"] ** 2
            * (1 - photo**2 * cosine)
        )
    )
    np.testing.assert_allclose(averaged, np.diag(expected), atol=1e-10)
    k = np.array([0.1, 0.2, 0.3])
    result = independent_source_pair_kernel(k, nodes, weights, covariance_inputs=values)
    reverse = independent_source_pair_kernel(-k, nodes, weights, covariance_inputs=values)
    np.testing.assert_allclose(reverse, result.conjugate(), atol=1e-12)
    assert abs(result) <= 1
    assert not np.isclose(result, np.exp(-k @ averaged @ k / 2 - (k[2] * sigma) ** 2))


@pytest.mark.parametrize(
    "name,value",
    [
        ("wavevectors", np.zeros((3, 3))),
        ("linear_power", np.array([-1, 1, 1])),
        ("source_noise_power", np.full(3, np.nan)),
        ("integration_weights", np.zeros(3)),
        ("source_response", np.ones((3, 1))),
        ("line_of_sight", np.array([0, 0, 2])),
        ("target_growth", -1),
    ],
)
def test_invalid_physical_inputs_are_rejected(name, value):
    values = inputs()
    values[name] = value
    with pytest.raises(ValueError):
        independent_source_pair_covariance(**values, target_errors=[0, 0])


def test_error_quadrature_does_not_silently_renormalize():
    with pytest.raises(ValueError, match="normalized"):
        independent_source_pair_kernel([0, 0, 1], [-1, 1], [1, 1], covariance_inputs=inputs())


def test_native_window_pair_shift_matches_fixed_count_normalized_contributions():
    """Check the proposed conditional window identity against real native solves.

    The random window and observed target pair stay fixed. Enumerate both forced
    self membership and exclusive transfer exclusions. This tests the normalized
    discrete operator, not an unconditional survey window or BAO damping centres.
    """
    pyrecon = pytest.importorskip("pyrecon")
    from pzbao.config import load_analysis_config

    settings = load_analysis_config()["measurement"]["reconstruction"]
    radius = settings["exploratory_transfer"]["common_smoothing_mpc_h"]
    bias, growth = 2.1, 0.8  # Synthetic operator fixture, not adopted tracer values.
    centre = np.array([1000.0, 0.0, 0.0])
    positions = centre + np.array(
        [
            [-70, 40, -20],
            [55, -30, 35],
            [35, 55, -45],
            [-45, -60, 15],
            [90, 15, 60],
            [-10, -15, -70],
        ],
        dtype=float,
    )
    weights = np.array([0.5, 2.0, 1.3, 3.0, 0.8, 1.7])
    pair = positions[:2].copy()
    axis = np.linspace(-108.0, 108.0, 10)
    randoms = centre + np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    random_weights = 1 + 0.2 * (randoms[:, 0] - centre[0]) / 108 + 0.1 * np.sin(randoms[:, 1] / 50)

    def reconstruction():
        result = pyrecon.IterativeFFTReconstruction(
            f=growth,
            bias=bias,
            los=None,
            nmesh=[16] * 3,
            boxsize=[512.0] * 3,
            boxcenter=centre,
            resampler=settings["resampler"],
            fft_plan=settings["fft_plan"],
            dtype=settings["mesh_dtype"],
            threshold_randoms=settings["threshold_randoms"]["value"],
            wrap=False,
            position_type="pos",
        )
        assert result.mpicomm.size == 1
        return result

    def pair_readout(result):
        result.run(niterations=settings["iterations"])
        shifted = result.read_shifts(pair, field="disp+rsd")
        return shifted[0] - shifted[1]

    def linear_pair_response(density):
        result = reconstruction()
        result.mesh_delta = result.pm.create(type="real", value=0.0)
        result.mesh_delta[...] = density
        return pair_readout(result)

    window = reconstruction()
    window.assign_randoms(randoms, random_weights)
    window.smoothing_radius = radius
    random_mesh = np.asarray(window._smooth_gaussian(window.mesh_randoms)).copy()
    total_random_mass = random_mesh.sum()
    threshold = settings["threshold_randoms"]["value"] * total_random_mass / len(randoms)
    mask = random_mesh > threshold
    assert mask.any() and not mask.all()  # Exercise a finite, nontrivial survey mask.
    offset = linear_pair_response(mask.astype(float) / bias)
    assert np.linalg.norm(offset) > 1e-8  # Dropping the window term must be detectable.
    numerator_fields, numerators, masses = [], [], []
    for position, weight in zip(positions, weights):
        painted = reconstruction()
        painted.assign_data(position[None, :], np.array([weight]))
        painted.smoothing_radius = radius
        contribution = np.asarray(painted._smooth_gaussian(painted.mesh_data)).copy()
        masses.append(contribution.sum())
        np.testing.assert_allclose(masses[-1], weight, rtol=1e-12, atol=1e-12)
        density = np.zeros_like(random_mesh)
        density[mask] = total_random_mass * contribution[mask] / (bias * random_mesh[mask])
        numerator_fields.append(density)
        numerators.append(linear_pair_response(density))
    numerators, masses = np.asarray(numerators), np.asarray(masses)
    numerator_fields = np.asarray(numerator_fields)
    joint = np.column_stack([numerators, masses])
    for count, forced, excluded in [(3, (0, 1), ()), (2, (), (0, 1))]:
        available = sorted(set(range(len(positions))) - set(forced) - set(excluded))
        assignments = [
            list(forced) + list(drawn)
            for drawn in itertools.combinations(available, count - len(forced))
        ]
        joint_sums, native_shifts = [], []
        for selected in assignments:
            direct = reconstruction()
            direct.assign_data(positions[selected], weights[selected])
            direct.assign_randoms(randoms, random_weights)
            direct.set_density_contrast(smoothing_radius=radius, kw_weights=None)
            mass = masses[selected].sum()
            expected_density = numerator_fields[selected].sum(axis=0) / mass - mask / bias
            np.testing.assert_allclose(
                np.asarray(direct.mesh_delta), expected_density, rtol=1e-11, atol=1e-11
            )
            expected_shift = numerators[selected].sum(axis=0) / mass - offset
            native_shifts.append(pair_readout(direct))
            np.testing.assert_allclose(native_shifts[-1], expected_shift, rtol=1e-10, atol=1e-10)
            joint_sums.append(joint[selected].sum(axis=0))
        joint_sums = np.asarray(joint_sums)
        mean, covariance = fixed_count_source_moments(
            joint, count, forced_indices=forced, excluded_indices=excluded
        )
        np.testing.assert_allclose(mean, joint_sums.mean(axis=0), rtol=1e-12, atol=1e-12)
        residual = joint_sums - joint_sums.mean(axis=0)
        np.testing.assert_allclose(
            covariance, residual.T @ residual / len(assignments), rtol=1e-12, atol=1e-12
        )
        assert covariance[-1, -1] > 0
        assert not np.allclose(
            np.mean(native_shifts, axis=0),
            mean[:3] / mean[-1] - offset,
            rtol=1e-6,
            atol=1e-8,
        )  # The ratio of means is a different approximation, even for fixed source counts.

        modes = np.array([[0.0, 0, 0], [0.08, 0.02, 0.1], [-0.08, -0.02, -0.1]])
        exact = fixed_count_normalized_source_kernel(
            numerators,
            masses,
            count,
            modes,
            window_offset=offset,
            max_assignments=10,
            forced_indices=forced,
            excluded_indices=excluded,
        )
        assert exact["empty_source_count"] == 0
        assert exact["assignment_count"] == len(assignments)
        expected_kernel = np.exp(-1j * (np.asarray(native_shifts) @ modes.T)).mean(axis=0)
        np.testing.assert_allclose(exact["kernel"], expected_kernel, rtol=1e-9, atol=1e-10)


def test_normalized_kernel_exact_unequal_mass_phase_and_scaling():
    numerators = np.array([[2.0, 0, 0], [12.0, 0, 0]])
    masses = np.array([1.0, 3.0])
    modes = np.array([[0.0, 0, 0], [0.7, 0, 0], [-0.7, 0, 0]])
    offset = np.array([0.5, 0, 0])
    result = fixed_count_normalized_source_kernel(
        numerators, masses, 1, modes, window_offset=offset, max_assignments=2
    )
    expected = (np.exp(-1j * modes[:, 0] * 1.5) + np.exp(-1j * modes[:, 0] * 3.5)) / 2
    np.testing.assert_allclose(result["kernel"], expected, rtol=1e-14, atol=1e-14)
    assert result["assignment_count"] == 2
    assert result["empty_source_count"] == 0
    assert result["empty_source_probability"] == 0
    assert result["kernel"][0] == 1
    assert result["kernel"][2] == pytest.approx(result["kernel"][1].conjugate())
    assert np.all(np.abs(result["kernel"]) <= 1 + 1e-14)
    wrong = (np.exp(-1j * modes[:, 0] * 0.5) + np.exp(-1j * modes[:, 0] * 5.5)) / 2
    assert abs(result["kernel"][1] - wrong[1]) > 0.01
    scaled = fixed_count_normalized_source_kernel(
        7 * numerators, 7 * masses, 1, modes, window_offset=offset, max_assignments=2
    )
    np.testing.assert_allclose(scaled["kernel"], expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(numerators, [[2.0, 0, 0], [12.0, 0, 0]])
    np.testing.assert_array_equal(masses, [1.0, 3.0])


@pytest.mark.parametrize("count,forced,excluded", [(2, (0,), ()), (1, (), (0,)), (2, (), ())])
def test_normalized_kernel_forced_excluded_and_full_draw(count, forced, excluded):
    numerators = np.array([[2.0, 0, 0], [12.0, 0, 0]])
    masses = np.array([1.0, 3.0])
    mode = np.array([[0.7, 0, 0]])
    result = fixed_count_normalized_source_kernel(
        numerators,
        masses,
        count,
        mode,
        window_offset=[0.5, 0, 0],
        max_assignments=1,
        forced_indices=forced,
        excluded_indices=excluded,
    )
    shift = 3.5 if count == 1 else 3.0
    np.testing.assert_allclose(result["kernel"], [np.exp(-1j * 0.7 * shift)], atol=1e-14)
    assert result["assignment_count"] == 1


@pytest.mark.parametrize(
    "count,forced,excluded,total,empty",
    [
        (0, (), (), 1, 1),
        (1, (), (), 3, 2),
        (2, (), (), 3, 1),
        (1, (2,), (), 1, 0),
        (1, (), (2,), 2, 2),
    ],
)
def test_normalized_kernel_keeps_empty_source_failure_mass(count, forced, excluded, total, empty):
    result = fixed_count_normalized_source_kernel(
        [[0.0, 0, 0]] * 3,
        [0.0, 0, 1.0],
        count,
        [[0.0, 0, 0], [1.0, 0, 0]],
        window_offset=[0.0, 0, 0],
        max_assignments=3,
        forced_indices=forced,
        excluded_indices=excluded,
    )
    assert result["assignment_count"] == total
    assert result["empty_source_count"] == empty
    assert result["empty_source_probability"] == empty / total
    if empty:
        assert result["kernel"] is None  # No conditional renormalization over successful draws.
    else:
        np.testing.assert_array_equal(result["kernel"], [1, 1])


def test_normalized_kernel_refuses_unbounded_enumeration():
    with pytest.raises(ValueError, match="above 10"):
        fixed_count_normalized_source_kernel(
            np.zeros((32, 3)),
            np.ones(32),
            16,
            [[0, 0, 0]],
            window_offset=[0, 0, 0],
            max_assignments=10,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"contribution_masses": [-1, 1]},
        {"contribution_masses": [1]},
        {"contribution_numerators": [[1, 2], [3, 4]]},
        {"contribution_numerators": [[1, 0, 0], [0, 0, 0]], "contribution_masses": [0, 1]},
        {"wavevectors": [[np.nan, 0, 0]]},
        {"wavevectors": []},
        {"window_offset": [0, 0]},
        {"max_assignments": True},
        {"max_assignments": 0},
        {"max_assignments": 1.5},
        {"source_count": 3},
        {"source_count": 0, "forced_indices": (0,)},
        {"forced_indices": (0,), "excluded_indices": (0,)},
    ],
)
def test_normalized_kernel_rejects_invalid_inputs(changes):
    kwargs = dict(
        contribution_numerators=np.zeros((2, 3)),
        contribution_masses=np.ones(2),
        source_count=1,
        wavevectors=[[0, 0, 0]],
        window_offset=[0, 0, 0],
        max_assignments=2,
    )
    kwargs.update(changes)
    with pytest.raises(ValueError):
        fixed_count_normalized_source_kernel(**kwargs)


@pytest.mark.parametrize(
    "parents,left_count,right_count,forced_left,forced_right,excluded",
    [
        (4, 1, 1, (), (), ()),
        (4, 2, 2, (), (), ()),
        (5, 2, 2, (0,), (1,), ()),
        (5, 2, 1, (0,), (), (4,)),
        (4, 0, 3, (), (), ()),
        (4, 4, 0, (), (), ()),
        (2, 1, 1, (), (), ()),
        (1, 0, 1, (), (), ()),
        (0, 0, 0, (), (), ()),
        (4, 1, 2, (0,), (1,), (3,)),
    ],
)
def test_exclusive_cross_covariance_matches_joint_assignment_enumeration(
    parents, left_count, right_count, forced_left, forced_right, excluded
):
    index = np.arange(parents, dtype=float)
    left = np.column_stack((index**2 - 2, 2 * index + 1))
    right = np.column_stack((index + 3, -(index**2), (-1.0) ** index))
    remaining = sorted(set(range(parents)) - set(forced_left + forced_right + excluded))
    left_sums, right_sums = [], []
    for left_draw in itertools.combinations(remaining, left_count - len(forced_left)):
        right_pool = sorted(set(remaining) - set(left_draw))
        for right_draw in itertools.combinations(right_pool, right_count - len(forced_right)):
            left_sums.append(left[list(forced_left + left_draw)].sum(axis=0))
            right_sums.append(right[list(forced_right + right_draw)].sum(axis=0))
    left_sums, right_sums = np.asarray(left_sums), np.asarray(right_sums)
    expected = (
        (left_sums - left_sums.mean(axis=0)).T
        @ (right_sums - right_sums.mean(axis=0))
        / len(left_sums)
    )
    actual = exclusive_class_source_cross_covariance(
        left,
        right,
        left_count,
        right_count,
        forced_left=forced_left,
        forced_right=forced_right,
        excluded_indices=excluded,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
    reverse = exclusive_class_source_cross_covariance(
        right,
        left,
        right_count,
        left_count,
        forced_left=forced_right,
        forced_right=forced_left,
        excluded_indices=excluded,
    )
    np.testing.assert_allclose(reverse, actual.T, rtol=1e-13, atol=1e-12)


def test_exclusive_cross_covariance_complement_cancels_assignment_variance():
    values = np.array([[2, -1], [3, 4], [-2, 5], [8, -3]], dtype=float)
    _, variance = fixed_count_source_moments(values, 2)
    cross = exclusive_class_source_cross_covariance(values, values, 2, 2)
    np.testing.assert_allclose(cross, -variance, rtol=0, atol=1e-14)
    assert np.linalg.norm(cross) > 0  # Exclusive classes are not independent samples.


def test_exclusive_cross_covariance_independent_parent_marks_need_no_extra_noise():
    left = np.array([[1, 2], [3, -1], [-2, 4]], dtype=float)
    right = np.array([[4], [-2], [1]], dtype=float)
    left_noise = np.array([[1, 3], [2, -1], [-1, 2]], dtype=float)
    right_noise = np.array([[2], [-1], [3]], dtype=float)
    left_sums, right_sums = [], []
    # The two possible marks of each parent are correlated across classes, but
    # that same parent can never occur in both sums in an exclusive assignment.
    for marks in itertools.product((-1.0, 1.0), repeat=3):
        marks = np.asarray(marks)[:, None]
        marked_left, marked_right = left + marks * left_noise, right + marks * right_noise
        for i, j in itertools.permutations(range(3), 2):
            left_sums.append(marked_left[i])
            right_sums.append(marked_right[j])
    left_sums, right_sums = np.asarray(left_sums), np.asarray(right_sums)
    expected = (
        (left_sums - left_sums.mean(axis=0)).T
        @ (right_sums - right_sums.mean(axis=0))
        / len(left_sums)
    )
    actual = exclusive_class_source_cross_covariance(left, right, 1, 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize(
    "changes",
    [
        {"right_contributions": np.ones((3, 2))},
        {"left_contributions": np.ones(4)},
        {"right_contributions": np.empty((4, 0))},
        {"left_contributions": np.full((4, 2), np.nan)},
        {"left_count": 1.5},
        {"right_count": True},
        {"left_count": 3, "right_count": 2},
        {"forced_left": (0,), "forced_right": (0,)},
        {"forced_left": (0, 1)},
        {"forced_right": (4,)},
        {"excluded_indices": (1, 1)},
        {"forced_left": (0,), "excluded_indices": (0,)},
        {"excluded_indices": (-1,)},
        {"forced_right": ((1,),)},
        {"forced_left": (0.5,)},
    ],
)
def test_exclusive_cross_covariance_rejects_incompatible_joint_conditioning(changes):
    arguments = {
        "left_contributions": np.ones((4, 2)),
        "right_contributions": np.ones((4, 3)),
        "left_count": 1,
        "right_count": 1,
    }
    arguments.update(changes)
    with pytest.raises(ValueError):
        exclusive_class_source_cross_covariance(**arguments)
