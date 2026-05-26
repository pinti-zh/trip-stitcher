# compare_speed_profile_opt.py
#
# Standalone script that compares two speed-profile optimisation approaches
# for a hard-coded sample trip (route 220):
#
#   1. Reference  — ocsept's TimeOptimalStrategy (IPOPT via optool)
#   2. Local      — re-implemented CasADi NLP inside EnergyDemandEstimator
#
# The script reports travel-time accuracy and runtime speedup, then renders
# two Plotly figures: a speed-profile comparison and an acceleration-profile
# comparison with active constraint limits (torque, power, comfort, traction).
#
# Hard-coded parameters at the top of the file control the bus type
# (BUS_TYPE) and auxiliary power consumption (AUX_POWER).
#
# Usage:
#   python compare_speed_profile_opt.py
#
# Note: requires ocsept and optool packages in addition to the standard
#       trip_stitcher dependencies.

import time
from contextlib import contextmanager

import numpy as np
import plotly.graph_objects as go
from IPython.utils.io import capture_output
from loguru import logger
from ocsept.data.generation.speed.time_optimal import TimeOptimalStrategy
from ocsept.models.transport.comfort import RidingComfort
from ocsept.models.transport.mission import DrivingMission as OcseptDrivingMission
from ocsept.simulation.qss import LongitudinalVehicleDynamics
from optool.uom import Quantity

from trip_stitcher.elevation import ElevationOracle
from trip_stitcher.energy_demand_estimator import EnergyDemandEstimator
from trip_stitcher.models import Stop, Trip


@contextmanager
def suppress_output():
    with capture_output():
        yield


# ---------------------------------------------------------------------------
# Hard-coded parameters
# ---------------------------------------------------------------------------
BUS_TYPE = "mega"
AUX_POWER = 2.0e3  # W

# ---------------------------------------------------------------------------
# Hard-coded sample data (route 220, trip 2.TA.96-100-0-j25-1.8.H)
# ---------------------------------------------------------------------------
stop_dict = {
    "8571330": Stop(id="8571330", name="Reichenbach i. K., Bahnhof", lon=7.690208, lat=46.625525),
    "8571331": Stop(id="8571331", name="Reichenbach i. K., Bären", lon=7.694187, lat=46.625432),
    "8583254": Stop(id="8583254", name="Scharnachtal, Halten", lon=7.697223, lat=46.620608),
    "8507766": Stop(id="8507766", name="Scharnachtal, Viesen", lon=7.697807, lat=46.617979),
    "8571332": Stop(id="8571332", name="Scharnachtal, Schulhaus", lon=7.698158, lat=46.614709),
}

trip = Trip(
    id="2.TA.96-100-0-j25-1.8.H",
    route="96-100-0-j25-1",
    stops=["8571330", "8571331", "8583254", "8507766", "8571332"],
    arrival_times=["18:27:00", "18:28:00", "18:30:00", "18:32:00", "18:33:00"],
)


def main() -> None:

    logger.remove()  # suppress ocsept internal logs

    if BUS_TYPE == "mega":
        from trip_stitcher.vehicles.mega import bus
    elif BUS_TYPE == "maxi":
        from trip_stitcher.vehicles.maxi import bus
    else:
        from trip_stitcher.vehicles.mini import bus

    # bus.powertrain.motors[0][1].torque_limit = Quantity(1300, "N*m")

    elevation_oracle = ElevationOracle()
    trip_geometry = trip.download_geometry(stop_dict, elevation_oracle=elevation_oracle)

    itinerary = trip_geometry.create_itinerary()
    comfort = RidingComfort()

    # -------------------------------------------------------------------------
    # Reference: TimeOptimalStrategy (ocsept / IPOPT via optool)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    with suppress_output():
        sp_ref = TimeOptimalStrategy().process(itinerary, bus, comfort)
    rt_ref = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # New: direct CasADi NLP
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    sp_cas = EnergyDemandEstimator._compute_speed_profile(itinerary, bus, comfort)
    rt_cas = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # Energy calculation (reference profile)
    # -------------------------------------------------------------------------
    mission = OcseptDrivingMission(
        name="Fastest possible travel speed",
        time=sp_ref.time,
        speed=sp_ref.speed,
        inclination=itinerary.get_inclination(sp_ref.distance),
        payload="0 kg",
    )
    with suppress_output():
        vehicle_dynamics = LongitudinalVehicleDynamics.of(bus, mission)

    average_velocity = sp_ref.speed[:-1] + sp_ref.speed[1:] / 2
    p_mech = average_velocity * vehicle_dynamics.traction_force[:-1]

    efficiency = 0.9
    propulsion_power = np.where(p_mech >= 0, p_mech / efficiency, p_mech * efficiency)
    propulsion_energy = np.sum(np.diff(sp_ref.time) * propulsion_power).to("J")
    print(f"Propulsion energy (reference): {propulsion_energy}")

    t_ref_end = float(sp_ref.time[-1].m_as("s"))
    t_cas_end = float(sp_cas.time[-1].m_as("s"))
    print(f"Travel time — reference : {t_ref_end:.2f} s")
    print(f"Travel time — CasADi   : {t_cas_end:.2f} s")
    print(
        f"Difference              : {abs(t_cas_end - t_ref_end):.3f} s  "
        f"({abs(t_cas_end - t_ref_end) / t_ref_end * 100:.2f} %)"
    )
    print()
    print(f"Runtime — reference : {rt_ref:.2f} s")
    print(f"Runtime — CasADi   : {rt_cas:.2f} s")
    print(f"Speedup             : {rt_ref / rt_cas:.2f}×")

    # -------------------------------------------------------------------------
    # Comparison plot
    # -------------------------------------------------------------------------
    d_ref = sp_ref.distance.m_as("m")
    v_ref = sp_ref.speed.m_as("m/s") * 3.6  # convert to km/h for display
    d_cas = sp_cas.distance.m_as("m")
    v_cas = sp_cas.speed.m_as("m/s") * 3.6

    # Elevation along the route (sampled at CasADi distances for the secondary axis)
    elev = itinerary.get_inclination(sp_cas.distance).m_as("rad")
    cum_elev = np.degrees(elev)  # inclination in degrees

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d_ref,
            y=v_ref,
            mode="lines",
            name="TimeOptimalStrategy (reference)",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_cas,
            y=v_cas,
            mode="lines",
            name="CasADi NLP (new)",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_cas,
            y=cum_elev,
            mode="lines",
            name="Road inclination (°)",
            line=dict(color="#2ca02c", width=1),
            yaxis="y2",
            opacity=0.6,
        )
    )

    fig.update_layout(
        title=dict(text=f"Speed profile comparison — trip {trip.id}", x=0.5),
        xaxis=dict(title="Distance (m)"),
        yaxis=dict(title="Speed (km/h)", rangemode="tozero"),
        yaxis2=dict(
            title="Inclination (°)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=True,
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        template="plotly_white",
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.01,
                xanchor="right",
                yanchor="bottom",
                text=(
                    f"Travel time — Ref: {t_ref_end:.1f} s &nbsp;|&nbsp; "
                    f"CasADi: {t_cas_end:.1f} s &nbsp;|&nbsp; "
                    f"Δ = {abs(t_cas_end - t_ref_end):.2f} s "
                    f"({abs(t_cas_end - t_ref_end) / t_ref_end * 100:.2f} %)"
                    f"<br>"
                    f"Runtime &nbsp;&nbsp;&nbsp;— Ref: {rt_ref:.2f} s &nbsp;|&nbsp; "
                    f"CasADi: {rt_cas:.2f} s &nbsp;|&nbsp; "
                    f"Speedup: {rt_ref / rt_cas:.2f}×"
                ),
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.7)",
            )
        ],
    )

    fig.show()

    # -------------------------------------------------------------------------
    # Figure 2: Acceleration profiles + active limits (verification)
    # Limits are evaluated at the CasADi solution's speed/distance profile.
    # -------------------------------------------------------------------------
    s2 = sp_cas.distance.m_as("m")
    N2 = len(s2)
    s2_mid = 0.5 * (s2[:-1] + s2[1:])

    # Bus parameters (mirror those inside compute_speed_profile_casadi)
    _G = 9.81
    _RHO = 1.225
    r_w = float(bus.chassis.wheel_radius.m_as("m"))
    fd_ratio = float(bus.powertrain.final_drive_transmission_ratio)
    fd_eff = float(bus.powertrain.final_drive_efficiency)
    Cd_Af = float(bus.chassis.aerodynamic_drag_area.m_as("m**2"))
    cr = float(bus.chassis.rolling_friction)
    m_curb = float(bus.curb_weight.m_as("kg"))
    I_rot = float(bus.rotational_inertia_at_wheels.m_as("kg*m**2"))

    n_m, motor_f2 = bus.powertrain.motors[0]
    n_m = int(n_m)
    T_shaft_lim = float(motor_f2.torque_limit.m_as("N*m")) * float(motor_f2.transmission_ratio)
    P_motor_lim = float(motor_f2.power_limit.m_as("W"))
    motor_eff = float(motor_f2.constant_efficiency)
    P_bat_max = float(bus.battery.get_maximum_power_output(soc=0.7, soh=1.0).m_as("W"))

    # Itinerary quantities at CasADi sample points
    alpha2 = itinerary.get_inclination(sp_cas.distance).m_as("rad")
    m_payload2 = itinerary.get_payload(sp_cas.distance).m_as("kg")
    p_aux2 = itinerary.get_peak_auxiliary_power(sp_cas.distance).m_as("W")

    m_eff2 = m_curb + m_payload2 + I_rot / r_w**2
    F_grav2 = m_eff2 * _G * np.sin(alpha2)
    F_roll2 = cr * m_eff2 * _G * np.cos(alpha2)
    P_mech_max2 = np.minimum(P_motor_lim, (P_bat_max - p_aux2) / n_m * motor_eff)

    # Mid-interval quantities
    v2 = sp_cas.speed.m_as("m/s")
    v2_mid = 0.5 * (v2[:-1] + v2[1:])
    v2_safe = np.maximum(v2_mid, 1e-3)
    m_eff2_mid = 0.5 * (m_eff2[:-1] + m_eff2[1:])
    F_grav2_mid = 0.5 * (F_grav2[:-1] + F_grav2[1:])
    F_roll2_mid = 0.5 * (F_roll2[:-1] + F_roll2[1:])
    F_aero2_mid = 0.5 * _RHO * Cd_Af * v2_mid**2
    P_mech_max2_mid = 0.5 * (P_mech_max2[:-1] + P_mech_max2[1:])
    shaft_spd2 = v2_safe * fd_ratio / r_w

    # Net resistance contribution to acceleration (negative = opposing motion)
    a_resist = (-F_grav2_mid - F_aero2_mid - F_roll2_mid) / m_eff2_mid

    # Acceleration limits from torque and power individually
    a_lim_torque = n_m * T_shaft_lim * fd_ratio * fd_eff / r_w / m_eff2_mid + a_resist
    a_lim_power = (
        n_m * (P_mech_max2_mid / shaft_spd2) * fd_ratio * fd_eff / r_w / m_eff2_mid + a_resist
    )

    F_trac_max = np.array(
        [
            bus.get_maximum_traction_force(
                speed=Quantity(float(v), "m/s"),
                auxiliary_power=Quantity(float(p), "W"),
                soc=0.7,
                soh=1.0,
            ).m_as("N")
            for v, p in zip(v2_mid, 0.5 * (p_aux2[:-1] + p_aux2[1:]))
        ]
    )
    a_max_trac = (F_trac_max - F_grav2_mid - F_aero2_mid - F_roll2_mid) / m_eff2_mid

    # Comfort limits
    a_comfort_decel = float(comfort.constant_deceleration_limit.m_as("m/s**2"))
    c_acc = float(comfort.constant_acceleration_limit.m_as("m/s**2"))
    if comfort.is_speed_dependent():
        a0_acc = float(comfort.max_acceleration_standstill.m_as("m/s**2"))
        b_acc = float(comfort.acceleration_shrink_rate.m_as("s/m"))
        a_lim_comfort = c_acc + (a0_acc - c_acc) * 2 / (1 + np.exp(b_acc * v2_mid))
    else:
        a_lim_comfort = np.full(N2 - 1, c_acc)

    # Actual accelerations
    a_cas2 = sp_cas.acceleration.m_as("m/s**2")

    # Reference acceleration re-sampled to CasADi midpoints
    d_ref_mid = 0.5 * (sp_ref.distance.m_as("m")[:-1] + sp_ref.distance.m_as("m")[1:])
    a_ref_interp = np.interp(s2_mid, d_ref_mid, sp_ref.acceleration.m_as("m/s**2"))

    # Y-axis range: focus on the physically meaningful region
    y_lo = a_comfort_decel - 0.5
    y_hi = float(np.nanmax(a_lim_comfort)) + 0.5

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=s2_mid,
            y=a_ref_interp,
            mode="lines",
            name="TimeOptimalStrategy (reference)",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=s2_mid,
            y=a_cas2,
            mode="lines",
            name="CasADi NLP",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=s2_mid,
            y=a_lim_comfort,
            mode="lines",
            name="Comfort accel limit (speed-dep.)",
            line=dict(color="#2ca02c", width=1.5, dash="dot"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[s2_mid[0], s2_mid[-1]],
            y=[a_comfort_decel, a_comfort_decel],
            mode="lines",
            name="Comfort decel limit",
            line=dict(color="#d62728", width=1.5, dash="dot"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=s2_mid,
            y=np.clip(a_lim_torque, y_lo - 2, y_hi + 2),
            mode="lines",
            name="Torque limit",
            line=dict(color="#9467bd", width=1.5, dash="dashdot"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=s2_mid,
            y=np.clip(a_lim_power, y_lo - 2, y_hi + 2),
            mode="lines",
            name="Power limit",
            line=dict(color="#8c564b", width=1.5, dash="dashdot"),
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=s2_mid,
            y=np.clip(a_max_trac, y_lo - 2, y_hi + 2),
            mode="lines",
            name="Max traction limit",
            line=dict(color="#e377c2", width=1.5, dash="dashdot"),
        )
    )

    fig2.update_layout(
        title=dict(text=f"Acceleration profile & limits — trip {trip.id}", x=0.5),
        xaxis=dict(title="Distance (m)"),
        yaxis=dict(
            title="Acceleration (m/s²)",
            range=[y_lo, y_hi],
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        template="plotly_white",
    )

    fig2.show()


if __name__ == "__main__":
    main()
