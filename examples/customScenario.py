from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from simglucose.analysis.report import report
from datetime import datetime
import numpy as np
import pandas as pd
import os

now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())
folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
path = os.path.join(os.path.abspath('./results/'), folder_name)
os.makedirs(path, exist_ok=True)

# Create a simulation environment
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')
# custom scenario is a list of tuples (time, meal_size)
base_scen = [(7, 50), (12, 60), (18.5, 80), (23, 15)]
scen = []
for meal, vals in enumerate(base_scen):
    time, CHO = vals
    time += np.random.normal(0.0,0.25)
    if not meal == 3:
        CHO += np.random.normal(0.0, 10)
    else:
        CHO += np.random.normal(0.0, 5)
    scen.append((time, CHO))


scenario = CustomScenario(start_time=start_time, scenario=scen)
# scenario = CustomScenario(start_time=datetime.combine(datetime.now().date(), datetime.min.time()), scenario=[(7, 45), (12, 70), (16, 15), (18, 80), (23, 10)])
env = T1DSimEnv(patient, sensor, pump, scenario)

# Create a controller
# controller = PIDController(P=-0.0001, I=-0.000000275, D=-0.1)
controller = BBController()

log = [ 'patient name: ', patient.name,
        'sensor: ', sensor.name,
        'pump: ', str(type(pump)),
        'base_scen: ', str(base_scen),
        'scen: ', str(scen),
        'controller: ', str(type(controller))]

with open(os.path.join(path, 'scen.txt'), 'w') as f:
    f.write('\n'.join(log))

# Put them together to create a simulation object
s = SimObj(env, controller, timedelta(days=2), animate=False, path=path)

results = sim(s)
df = pd.concat([results], keys=[s.env.patient.name])
results, ri_per_hour, zone_stats, figs, axes = report(df, path)