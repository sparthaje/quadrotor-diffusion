import numpy as np
import pandas as pd
from competition.data_gen_costs import *
from competition.test_case import TestCase
from competition.data_collection_ctrls import Controller, yaw_rot, INITIAL_GATE_EXIT

def cost_function(row):
  g2 = np.arctan2(np.sin(row["theta1"] + row["theta2"]), np.cos(row["theta1"] + row["theta2"]))
  
  # add gates to the map convert local frame data in test_case to global frame
  second_gate_pos = np.array([0, 0, row["z0"]]) + row["d1"] * (yaw_rot(0) @ INITIAL_GATE_EXIT)
  second_gate_pos[2] = row["z1"]
  third_gate_pos = second_gate_pos + row["d2"] * (yaw_rot(row["theta1"]) @ INITIAL_GATE_EXIT)
  third_gate_pos[2] = row["z2"]
  
  initial_info = {
    "ctrl_timestep": 30,
    "ctrl_freq": 30,
    "nominal_gates_pos_and_type": [
      [0, 0, 0, 0, 0, 0, 1 if row["z0"] == 0.3 else 0],
      [second_gate_pos[0], second_gate_pos[1], 0, 0, 0, row["theta1"], 1 if row["z1"] == 0.3 else 0],
      [third_gate_pos[0], third_gate_pos[1], 0, 0, 0, g2, 1 if row["z2"] == 0.3 else 0]
    ],
    "nominal_obstacles_pos": [],
    "quadrotor_kf": 0
  }

  ctrl = Controller(None, initial_info)
  tc = TestCase(row["z0"], row["v0"], row["d1"], row["theta1"], row["z1"], row["d2"], row["theta2"], row["z2"])
  ctrl.build_traj(tc, row["v1"], row["t1"])
  
  dynamics_cost = 0
  
  if ACCEL_VECTOR:
    for a1, a2, a3, v1, v2, v3 in zip(ctrl.ref_acc[0], ctrl.ref_acc[1], ctrl.ref_acc[2], ctrl.ref_vel[0], ctrl.ref_vel[1], ctrl.ref_vel[2]):
      a = np.linalg.norm(np.array([a1, a2, a3]))
      v = np.linalg.norm(np.array([v1, v2, v3]))
      if v > VEL_LIMIT_COST:
        dynamics_cost += VEL_COST
      
      if abs(a) > PARABOLA_SHAPE * (v**2) + PARABOLA_INTERCEPT:
        dynamics_cost += ACCEL_COST
  else:  
    for xyz in range(3):
      for a, v in zip(ctrl.ref_acc[xyz], ctrl.ref_vel[xyz]):
        if v > VEL_LIMIT_COST:
          dynamics_cost += VEL_COST
        
        if abs(a) > PARABOLA_SHAPE * (v**2) + PARABOLA_INTERCEPT:
          dynamics_cost += ACCEL_COST

  #  Negatives because it returns a reward not a cost
  return -CRASH_COST * row["crashes"] - \
          SKIPPED_WAYPOINT_COST * row["skipped_waypoints"] - \
          TRACKING_COST * row["avg_err"] - \
          ACCEL_RANGE_COST * row["accl_rng"] - \
          TIME_COST * row["t1"] - \
          dynamics_cost

def find_best_v_t(group):
  group['cost'] = group.apply(lambda row: cost_function(row), axis=1)
  best_row = group.loc[group['cost'].idxmax()]
  return pd.Series([best_row["v1"], best_row["t1"]])

# Load initial data frame
df = pd.read_csv("data.csv")
grouped = df.groupby(list(df.columns[:8])).apply(lambda x: find_best_v_t(x))
grouped = grouped.reset_index()
grouped = grouped.rename(columns={0: 'best_v', 1: 'best_t'})
print(grouped)

