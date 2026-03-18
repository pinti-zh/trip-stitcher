from ocsept.models.components.battery import Battery, BatteryChemistry
from ocsept.models.components.chassis import Chassis
from ocsept.models.components.propulsion import ElectricMotor, ElectricPowertrain
from ocsept.models.components.vehicles import BatteryBus

# MAXI
chassis = Chassis(
    length="10.633m",
    weight="13.5t",
    gvm="18t",
    aerodynamic_drag_area="5.0m^2",
    rolling_friction=0.012,
    wheel_inertia=[(10, "20kg*m^2")],
    wheel_radius="0.45m",
)
motor = ElectricMotor(
    weight="360 kg",
    inertia="0.6 kg*m^2",
    transmission_ratio=1,
    torque_limit="2500 N*m", # apparently unrealistic
    power_limit="180 kW",
    speed_limit="3200 rpm",
    constant_efficiency=0.93,
)
powertrain = ElectricPowertrain(
    additional_weight="0kg",
    additional_inertia_at_shaft=[(2, "0.1kg*m^2")],
    final_drive_efficiency=0.95,
    motors=[(2, motor)],
    final_drive_transmission_ratio=6.2,
)
battery = Battery(
    weight="1800kg",
    chemistry=BatteryChemistry.LTO,
    capacity="120kWh",
    c_rate_limit_continuous_charging="4.1/h",
    c_rate_limit_continuous_discharging="4.1/h",
    c_rate_limit_peak_charging="6/h",
    c_rate_limit_peak_discharging="7/h",
)

bus = BatteryBus(chassis=chassis, powertrain=powertrain, battery=battery)
