from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, batch_sim
from datetime import timedelta
from simglucose.analysis.report import report
from datetime import datetime
import pkg_resources
import numpy as np
import pandas as pd
import os
import copy

# import patient parameters
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

# define start date as hour 0 of the current day
now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())


def create_result_folder():
    """create results folder"""
    _folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _path = os.path.join(os.path.abspath('./results/'), _folder_name)
    os.makedirs(_path, exist_ok=True)
    return _path


def build_envs(name, _scenario):
    """create environments for simulation
    name: patient name, e.g. 'adolescent#001
    scenario: simglucose.simulation.scenario.CustomScenario"""
    patient = T1DPatient.withName(name)
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    scen = copy.deepcopy(_scenario)
    env = T1DSimEnv(patient, sensor, pump, scen)
    return env


def create_scenario(_base_scen, _sim_days):
    repeat_scen = []
    vary_scen = []
    for simDay in range(_sim_days):
        for time, mealsize in _base_scen:
            repeat_scen.append((24 * simDay + time, mealsize))

    for meal, vals in enumerate(repeat_scen):
        time, CHO = vals
        time += np.random.normal(0.0, 0.25)
        if not (meal - 3) % 4 == 0:
            CHO += np.random.normal(0.0, 10)
        else:
            CHO += np.random.normal(0.0, 5)
        vary_scen.append((float(f'{time:.2f}'), float(f'{CHO:.2f}')))
    return CustomScenario(start_time=start_time, scenario=vary_scen), vary_scen


def write_log(_envs):
    """Generate log file containing infos to simulation. Contains patient names, sensor type, pump type, base scenario
    modified scenario, and controllers"""
    log = ['patient name: ', str(patient_names),
           'sensor: ', _envs[0].sensor.name,
           'pump: ', _envs[0].pump._params[0],
           'base scen: ', str(base_scen),
           'scen: ', str(scen_list),
           'controllers: ', str(controllers)]

    with open(os.path.join(path, 'scen.txt'), 'w') as f:
        f.write('\n'.join(log))


def select_patients(_patient_group='All'):
    """Select patients to run simulation for.
    Valid choices: 'All' (default), 'Adolescents', 'Adults', 'Children'"""
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    all_patients = list(patient_params['Name'].values)
    if _patient_group == 'All':
        return all_patients
    elif _patient_group == 'Adolescents':
        return all_patients[:10]
    elif _patient_group == 'Adults':
        return all_patients[10:20]
    elif _patient_group == 'Children':
        return all_patients[20:30]


def create_ctrllers(_ctrllers):
    """Enable to run same scenario with multiple controllers"""
    _controllers = []
    for _controller in _ctrllers:
        as_list = [copy.deepcopy(_controller) for _ in range(len(envs))]
        _controllers.extend(as_list)
    return _controllers


if __name__ == '__main__':
    path = create_result_folder()

    # select controller to run simulation with
    controller_names = ['BBController', 'PIDCtrller_0.0001_0.00000275_0.1']
    controllers = [BBController(), PIDController(P=-0.0001, I=-0.000000275, D=-0.1)]

    # Select parameters to run simulation for
    patient_group = 'Adolescents'
    sim_days = 3
    patient_names = select_patients(patient_group)
    # patient_names = ['adolescent#001', 'adolescent#002']

    # set base scenario and add variability
    base_scen = [(7, 50), (12, 60), (18.5, 80), (23, 15)]
    scenario, scen_list = create_scenario(base_scen, sim_days)

    for num, controller in enumerate(controllers):
        folder_name = controller_names[num]
        path_ctrl = os.path.join(path, folder_name)
        os.makedirs(path_ctrl, exist_ok=True)

        # create environment for each patient
        envs = [build_envs(patient, scenario) for patient in patient_names]

        # copy controller for each environment
        ctrllers = [copy.deepcopy(controller) for _ in range(len(envs))]

        # create simulation objects
        sim_instances = [
            SimObj(env, ctrl, timedelta(days=sim_days), animate=False, path=path_ctrl)
            for (env, ctrl) in zip(envs, ctrllers)
        ]

        write_log(envs)
        results = batch_sim(sim_instances, parallel=True)
        df = pd.concat(results, keys=[s.env.patient.name for s in sim_instances])
        results, ri_per_hour, zone_stats, figs, axes = report(df, path_ctrl)
