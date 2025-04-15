import time
import pickle
import requests
import numpy as np

COURSE = np.array([
    [-0.4520511, -1.27241438, 0.3, 0.],
    [-0.622861, -0.6179413, 0.3, 0.78143325],
    [-0.44599258, 0.65827566, 0.525, -0.97531991],
    [0.77898342, -0.41174775, 0.525, 2.65535065]
])

current_traj = None
gate_idx = 0
while True:
    global_context = COURSE
    if current_traj is not None:
        global_context = np.vstack((COURSE[gate_idx + 1:], COURSE[1:gate_idx]))
    data = pickle.dumps((current_traj, global_context))
    r = requests.post("http://localhost:5000/plan", data=data)
    next_traj = pickle.loads(r.content)
    print(next_traj)
    gate_idx += 1
    current_traj = next_traj
    if gate_idx == len(COURSE):
        gate_idx = 1
    time.sleep(1.0)
