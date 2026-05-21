from ocsept.models.components.battery import Battery, BatteryChemistry
from ocsept.models.components.chassis import Chassis
from ocsept.models.components.propulsion import ElectricMotor, ElectricPowertrain
from ocsept.models.components.vehicles import BatteryBus

# MINI
chassis = Chassis(
    length="5.94m",
    weight="2.7t",
    gvm="4.5t",
    aerodynamic_drag_area="2.5m^2",
    rolling_friction=0.01,
    wheel_inertia=[(4, "8kg*m^2")],
    wheel_radius="0.45m",
)
motor = ElectricMotor(
    weight="100kg",
    inertia="0.6 kg*m^2",
    transmission_ratio=1,
    torque_limit="1200 N*m",  # apparently unrealistic
    power_limit="100 kW",
    speed_limit="7000 rpm",
    constant_efficiency=0.93,
)
powertrain = ElectricPowertrain(
    additional_weight="0kg",
    additional_inertia_at_shaft=[(1, "0.1kg*m^2")],
    final_drive_efficiency=0.95,
    motors=[(1, motor)],
    final_drive_transmission_ratio=9.5,
)
battery = Battery(
    weight="1078kg",
    chemistry=BatteryChemistry.NMC,
    capacity="154kWh",
    c_rate_limit_continuous_charging="1/h",
    c_rate_limit_continuous_discharging="1/h",
    c_rate_limit_peak_charging="2/h",
    c_rate_limit_peak_discharging="2/h",
    soc_min=0.2,
    soc_max=0.8,
)

bus = BatteryBus(chassis=chassis, powertrain=powertrain, battery=battery)
