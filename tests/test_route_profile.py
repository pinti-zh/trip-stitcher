"""
Tests comparing RouteProfile against TravelItinerary (the reference implementation).

Run with:  python -m pytest tests/test_route_profile.py -v -s
"""

import timeit

import numpy as np
import pytest

from trip_stitcher.models import RouteProfile, TripGeometry

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
# The geometry below models three OSRM legs with 3 / 2 / 4 annotation points.
# distance[3] = 0.0 exercises the d<=0 skip logic in both implementations.
#
#  waypoint  0  1      2      3   (4 skipped, d=0)  5      6      7      8      9
#  is_stop   T  F      F      T        F             T      F      F      F      T
#  distance    150   200   180      0.0   120       90    110     85    130
#  speed_km/h  50    50    50      30.0    30       70     70     70     70
#  elevation 400  402    405    403    403         401    407    406    404    402
_GEOM = TripGeometry(
    lon=[0.0] * 10,
    lat=[0.0] * 10,
    distance=[150.0, 200.0, 180.0, 0.0, 120.0, 90.0, 110.0, 85.0, 130.0],
    elevation=[400.0, 402.0, 405.0, 403.0, 403.0, 401.0, 407.0, 406.0, 404.0, 402.0],
    is_stop=[True, False, False, True, False, True, False, False, False, True],
    speed_limit=[50.0, 50.0, 50.0, 30.0, 30.0, 70.0, 70.0, 70.0, 70.0],
)


@pytest.fixture(scope="module")
def travel_itinerary():
    return _GEOM.create_itinerary()


@pytest.fixture(scope="module")
def route_profile():
    return RouteProfile.from_trip_geometry(_GEOM)


# ---------------------------------------------------------------------------
# Helper: get the shared sample-distance array used by both objects
# ---------------------------------------------------------------------------
def _shared_samples(travel_itinerary, route_profile):
    """
    Return a numpy array of sample distances that both objects include as
    waypoint positions (so stop overrides fire for both implementations).
    We ask RouteProfile for its grid — the two should produce the same set
    of waypoint positions.
    """
    from optool.uom import Quantity

    s_new = route_profile.get_sample_distances(5.0)
    s_qty = Quantity(s_new, "m")
    return s_new, s_qty


# ---------------------------------------------------------------------------
# Correctness: sample distances
# ---------------------------------------------------------------------------
def test_sample_distances(travel_itinerary, route_profile):
    s_new = route_profile.get_sample_distances(5.0)
    s_ref = travel_itinerary.get_sample_distances("5 m").m_as("m")
    np.testing.assert_allclose(s_new, s_ref, rtol=1e-9, err_msg="Sample distance grids differ")


# ---------------------------------------------------------------------------
# Correctness: maximum speed limit
# ---------------------------------------------------------------------------
def test_max_speed_limit(travel_itinerary, route_profile):
    s_new, s_qty = _shared_samples(travel_itinerary, route_profile)
    v_max_new = route_profile.get_max_speed_limit(s_new)
    v_max_ref = travel_itinerary.get_speed_limit(s_qty, "maximum").m_as("m/s")
    np.testing.assert_allclose(
        v_max_new, v_max_ref, rtol=1e-9, err_msg="Maximum speed limits differ"
    )


# ---------------------------------------------------------------------------
# Correctness: minimum speed limit
# ---------------------------------------------------------------------------
def test_min_speed_limit(travel_itinerary, route_profile):
    s_new, s_qty = _shared_samples(travel_itinerary, route_profile)
    v_min_new = route_profile.get_min_speed_limit(s_new)
    v_min_ref = travel_itinerary.get_speed_limit(s_qty, "minimum").m_as("m/s")
    np.testing.assert_allclose(
        v_min_new, v_min_ref, rtol=1e-9, err_msg="Minimum speed limits differ"
    )


# ---------------------------------------------------------------------------
# Correctness: inclination
# ---------------------------------------------------------------------------
def test_inclination(travel_itinerary, route_profile):
    s_new, s_qty = _shared_samples(travel_itinerary, route_profile)
    alpha_new = route_profile.get_inclination(s_new)
    alpha_ref = travel_itinerary.get_inclination(s_qty).m_as("rad")
    np.testing.assert_allclose(alpha_new, alpha_ref, rtol=1e-9, err_msg="Inclinations differ")


# ---------------------------------------------------------------------------
# Correctness: stops have v_max = v_min = 0
# ---------------------------------------------------------------------------
def test_stops_have_zero_speed(route_profile):
    stop_dists = route_profile._waypoint_distances[route_profile._stop_mask]
    v_max_at_stops = route_profile.get_max_speed_limit(stop_dists)
    v_min_at_stops = route_profile.get_min_speed_limit(stop_dists)
    assert np.all(v_max_at_stops == 0.0), "v_max must be 0 at all stops"
    assert np.all(v_min_at_stops == 0.0), "v_min must be 0 at all stops"


# ---------------------------------------------------------------------------
# Correctness: zero-distance segment is skipped
# ---------------------------------------------------------------------------
def test_zero_distance_segment_skipped(route_profile):
    # The raw geometry has 9 segments; d[3]=0 is dropped, leaving 8 valid ones.
    assert len(route_profile._segment_max_speeds) == 8
    assert len(route_profile._waypoint_distances) == 9


# ---------------------------------------------------------------------------
# Timing comparison (printed to stdout with -s)
# ---------------------------------------------------------------------------
def test_timing(capsys):
    N_ITER = 200

    # --- Construction ---
    t_itinerary = timeit.timeit(
        lambda: _GEOM.create_itinerary(),
        number=N_ITER,
    )
    t_profile = timeit.timeit(
        lambda: RouteProfile.from_trip_geometry(_GEOM),
        number=N_ITER,
    )

    # --- Interpolation (construction + full query) ---
    def _itinerary_query():
        it = _GEOM.create_itinerary()
        s_qty = it.get_sample_distances("5 m")
        it.get_speed_limit(s_qty, "maximum")
        it.get_speed_limit(s_qty, "minimum")
        it.get_inclination(s_qty)

    def _profile_query():
        rp = RouteProfile.from_trip_geometry(_GEOM)
        s = rp.get_sample_distances(5.0)
        rp.get_max_speed_limit(s)
        rp.get_min_speed_limit(s)
        rp.get_inclination(s)

    t_itinerary_full = timeit.timeit(_itinerary_query, number=N_ITER)
    t_profile_full = timeit.timeit(_profile_query, number=N_ITER)

    with capsys.disabled():
        print(f"\n{'─' * 60}")
        print(f"  Timing over {N_ITER} iterations")
        print(f"{'─' * 60}")
        print("  Construction only:")
        print(
            f"    TravelItinerary  : {t_itinerary * 1e3:.1f} ms total  "
            f"({t_itinerary / N_ITER * 1e6:.1f} µs/call)"
        )
        print(
            f"    RouteProfile     : {t_profile * 1e3:.1f} ms total  "
            f"({t_profile / N_ITER * 1e6:.1f} µs/call)"
        )
        print(f"    Speed-up         : {t_itinerary / t_profile:.1f}×")
        print("  Construction + all queries:")
        print(
            f"    TravelItinerary  : {t_itinerary_full * 1e3:.1f} ms total  "
            f"({t_itinerary_full / N_ITER * 1e6:.1f} µs/call)"
        )
        print(
            f"    RouteProfile     : {t_profile_full * 1e3:.1f} ms total  "
            f"({t_profile_full / N_ITER * 1e6:.1f} µs/call)"
        )
        print(f"    Speed-up         : {t_itinerary_full / t_profile_full:.1f}×")
        print(f"{'─' * 60}")

    # sanity: new implementation is not slower
    assert t_profile_full < t_itinerary_full, (
        "RouteProfile should be faster than TravelItinerary, but was not"
    )
