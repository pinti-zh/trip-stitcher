import casadi
import numpy as np
import pandas as pd
from ocsept.data.generation.speed import SpeedProfile
from ocsept.models.transport.comfort import RidingComfort
from ocsept.models.transport.mission import DrivingMission as OcseptDrivingMission
from ocsept.simulation.qss import LongitudinalVehicleDynamics
from optool.uom import Quantity

from trip_stitcher.elevation import ElevationOracle
from trip_stitcher.models import RouteProfile, Stop, Trip
from trip_stitcher.utils import str_to_datetime, suppress_stdout
from trip_stitcher.vehicles import maxi, mega, mini


class _CasadiStrategyTag:
    """Sentinel satisfying the SpeedProfileStrategy protocol."""

    def process(self, itinerary, vehicle, comfort):
        return EnergyDemandEstimator._compute_speed_profile(itinerary, vehicle, comfort)


class EnergyDemandEstimator:
    def __init__(self, df: pd.DataFrame):
        self.stop_dict: dict[str, Stop] = dict(
            (stop.id, stop) for stop in Stop.list_from_dataframe(df)
        )
        self.elevation_oracle = ElevationOracle()
        self.cache: dict[tuple[str, ...], dict[str, float]] = {}

    @staticmethod
    def _compute_speed_profile(itinerary: RouteProfile, bus, comfort) -> SpeedProfile:
        """Compute a time-optimal speed profile via a direct CasADi/IPOPT NLP."""
        # ------------------------------------------------------------------
        # 1. Sample distances (SI: metres)
        # ------------------------------------------------------------------
        s = itinerary.get_sample_distances(5.0)  # np.ndarray, shape (N,)
        N = len(s)

        # ------------------------------------------------------------------
        # 2. Per-sample constraints from itinerary
        # ------------------------------------------------------------------
        v_max = itinerary.get_max_speed_limit(s)  # (N,) m/s
        v_min = itinerary.get_min_speed_limit(s)  # (N,) m/s
        alpha = itinerary.get_inclination(s)  # (N,) rad
        # payload and auxiliary power are zero for all modelled routes
        m_payload = np.zeros(N)  # (N,) kg
        p_aux = np.zeros(N)  # (N,) W

        # ------------------------------------------------------------------
        # 3. Bus parameters
        # ------------------------------------------------------------------
        _G = 9.81  # m/s²
        _RHO = 1.225  # kg/m³  (air density at sea level)

        r_w = float(bus.chassis.wheel_radius.m_as("m"))
        fd_ratio = float(bus.powertrain.final_drive_transmission_ratio)
        fd_eff = float(bus.powertrain.final_drive_efficiency)
        Cd_Af = float(bus.chassis.aerodynamic_drag_area.m_as("m**2"))
        cr = float(bus.chassis.rolling_friction)
        m_curb = float(bus.curb_weight.m_as("kg"))
        I_rot = float(bus.rotational_inertia_at_wheels.m_as("kg*m**2"))

        assert len(bus.powertrain.motors) == 1, (
            "_compute_speed_profile only supports a single motor type "
            f"(got {len(bus.powertrain.motors)} entries in bus.powertrain.motors)"
        )
        n_motors, motor = bus.powertrain.motors[0]
        n_motors = int(n_motors)
        motor_tr = float(motor.transmission_ratio)
        # Maximum torque at shaft (per motor) — torque_limit is at motor axle
        T_shaft_limit = float(motor.torque_limit.m_as("N*m")) * motor_tr
        P_motor_limit = float(motor.power_limit.m_as("W"))  # mechanical power limit per motor
        motor_eff = float(motor.constant_efficiency)

        P_bat_max = float(bus.battery.get_maximum_power_output(soc=0.7, soh=1.0).m_as("W"))

        # ------------------------------------------------------------------
        # 4. Comfort parameters
        # ------------------------------------------------------------------
        a_decel = float(comfort.constant_deceleration_limit.m_as("m/s**2"))  # negative
        c_acc = float(comfort.constant_acceleration_limit.m_as("m/s**2"))  # positive

        speed_dependent = comfort.is_speed_dependent()
        if speed_dependent:
            a0_acc = float(comfort.max_acceleration_standstill.m_as("m/s**2"))
            b_acc = float(comfort.acceleration_shrink_rate.m_as("s/m"))

        # ------------------------------------------------------------------
        # 5. Pre-computed per-sample arrays (numpy, SI)
        # ------------------------------------------------------------------
        m_eff = m_curb + m_payload + I_rot / r_w**2  # (N,) effective translational mass

        F_grav = m_eff * _G * np.sin(alpha)  # (N,) gravitational resistance
        F_roll = cr * m_eff * _G * np.cos(alpha)  # (N,) rolling resistance

        # Per-motor available mechanical power, clamped to motor's own power limit
        P_elec_per_motor = (P_bat_max - p_aux) / n_motors  # (N,)
        P_mech_max = np.minimum(P_motor_limit, P_elec_per_motor * motor_eff)  # (N,)

        ds = np.diff(s)  # (N-1,) segment lengths

        # ------------------------------------------------------------------
        # 6. CasADi decision variables
        # ------------------------------------------------------------------
        v = casadi.MX.sym("v", N)  # speed at each sample point (m/s)
        a = casadi.MX.sym("a", N - 1)  # acceleration in each interval (m/s²)

        # Trapezoidal time steps (CasADi symbolic)
        tau = 2 * ds / (v[:-1] + v[1:])  # (N-1,)

        # Objective: minimise total travel time
        f_obj = casadi.sum1(tau)

        g_eq = []  # equality constraints g = 0
        g_ineq = []  # inequality constraints g >= 0

        for i in range(N - 1):
            # (a) Velocity continuity
            g_eq.append(v[i + 1] - v[i] - tau[i] * a[i])

            # (b) Traction force limit (vehicle physics)
            #     fmax guards against shaft-speed singularity at v = 0
            v_safe = casadi.fmax(v[i], 1e-3)
            shaft_speed = v_safe * fd_ratio / r_w  # rad/s at shaft
            T_one = casadi.fmin(P_mech_max[i] / shaft_speed, T_shaft_limit)
            F_trac_max = n_motors * T_one * fd_ratio * fd_eff / r_w
            F_aero_i = 0.5 * _RHO * Cd_Af * v[i] ** 2
            a_trac_max = (F_trac_max - F_grav[i] - F_aero_i - F_roll[i]) / m_eff[i]
            g_ineq.append(a_trac_max - a[i])

            # (c) Comfort acceleration limit (speed-dependent sigmoid)
            if speed_dependent:
                a_comfort_max = c_acc + (a0_acc - c_acc) * 2 / (1 + casadi.exp(b_acc * v[i]))
            else:
                a_comfort_max = c_acc
            g_ineq.append(a_comfort_max - a[i])

        # ------------------------------------------------------------------
        # 7. Assemble and solve NLP
        # ------------------------------------------------------------------
        g_all = casadi.vertcat(*g_eq, *g_ineq)
        n_eq = len(g_eq)
        n_ineq = len(g_ineq)

        nlp = {"x": casadi.vertcat(v, a), "f": f_obj, "g": g_all}
        ipopt_opts = {"ipopt.print_level": 0}
        solver = casadi.nlpsol("speed_profile_solver", "ipopt", nlp, ipopt_opts)

        v0_init = np.clip(0.5 * (v_min + v_max), v_min, v_max)
        a0_init = np.zeros(N - 1)
        x0 = np.concatenate([v0_init, a0_init])

        lbx = np.concatenate([v_min, np.full(N - 1, a_decel)])
        ubx = np.concatenate([v_max, np.full(N - 1, a0_acc if speed_dependent else c_acc)])

        lbg = np.concatenate([np.zeros(n_eq), np.zeros(n_ineq)])
        ubg = np.concatenate([np.zeros(n_eq), np.full(n_ineq, np.inf)])

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        # ------------------------------------------------------------------
        # 8. Reconstruct time array and return SpeedProfile
        # ------------------------------------------------------------------
        x_sol = np.array(sol["x"]).flatten()
        v_sol = x_sol[:N]
        a_sol = x_sol[N:]

        tau_sol = 2 * np.diff(s) / (v_sol[:-1] + v_sol[1:])
        t_sol = np.cumsum(np.insert(tau_sol, 0, 0.0))

        return SpeedProfile(
            time=Quantity(t_sol, "s"),
            distance=Quantity(s, "m"),
            speed=Quantity(v_sol, "m/s"),
            acceleration=Quantity(a_sol, "m/s**2"),
            strategy_used=_CasadiStrategyTag(),
        )

    def calculate_energy_demand(
        self, trip: Trip, bus_type: str | None = None, aux_power: float = 0.0
    ) -> float:
        cache_key = tuple(trip.stops + [str(aux_power), str(bus_type)])
        if cache_key in self.cache.keys():
            cached = self.cache[cache_key]
            trip.estimated_energy_demand = cached["energy"]
            trip.covered_distance = cached["distance"]
            return cached["energy"]

        trip_geometry = trip.download_geometry(
            self.stop_dict, elevation_oracle=self.elevation_oracle
        )

        max_inclination = max(
            abs(100 * (e2 - e1) / d)
            for d, e1, e2 in zip(
                trip_geometry.distance,
                trip_geometry.elevation[:-1],
                trip_geometry.elevation[1:],
            )
            if d > 0
        )
        assert max_inclination < 20

        itinerary = RouteProfile.from_trip_geometry(trip_geometry)

        bus = None
        match bus_type:
            case "mini":
                bus = mini.bus
            case "maxi":
                bus = maxi.bus
            case "mega":
                bus = mega.bus
            case _:
                raise ValueError(f"Unknown bus type: {bus_type}")
        assert bus is not None

        comfort = RidingComfort()

        speed_profile = self._compute_speed_profile(itinerary, bus, comfort)

        assert speed_profile.distance is not None
        assert speed_profile.speed is not None
        assert speed_profile.time is not None

        mission = OcseptDrivingMission(
            name="Fastest possible travel speed",
            time=speed_profile.time,
            speed=speed_profile.speed,
            inclination=Quantity(
                itinerary.get_inclination(speed_profile.distance.m_as("m")), "rad"
            ),
            payload="0 kg",
        )

        with suppress_stdout():
            vehicle_dynamics = LongitudinalVehicleDynamics.of(bus, mission)

        average_velocity = speed_profile.speed[:-1] + speed_profile.speed[1:] / 2
        p_mech = average_velocity * vehicle_dynamics.traction_force[:-1]
        efficiency = 0.9
        propulsion_power = np.where(p_mech >= 0, p_mech / efficiency, p_mech * efficiency)
        propulsion_energy = np.sum(np.diff(speed_profile.time) * propulsion_power).to("J")

        trip_duration = str_to_datetime(trip.arrival_times[-1]) - str_to_datetime(
            trip.arrival_times[0]
        )
        aux_energy = (Quantity(aux_power, "W") * Quantity(trip_duration.total_seconds(), "s")).to(
            "J"
        )

        total_energy = propulsion_energy.magnitude + aux_energy.magnitude
        distance = sum(trip_geometry.distance)
        self.cache[cache_key] = {"energy": total_energy, "distance": distance}
        trip.estimated_energy_demand = total_energy
        trip.covered_distance = distance
        return total_energy
