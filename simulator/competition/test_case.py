import numpy as np

class TestCase:
  def __init__(self, z, v, dist, g_theta, g1z, end_dist, end_theta, g2z):
    # first gate position (z is the only one that really matters xy are there from legacy cocde)
    self.x = 0
    self.y = -1.5
    self.z = z
    
    # first gate velocity
    self.v = v
    
    # theta relative to each previous gate, in this implementation theta should always be zero (relative to nothing)
    self.theta_rel = [0, g_theta, end_theta]
    
    self.theta = 0
    
    # (dist, g_theta) is the polar coordinates from first gate to second gate
    self.dist = dist
    # here we modify g_theta to be second gate angle in global frame
    # note that this is sort of legacy since self.theta is fixed to be zero
    g_theta += self.theta
    g_theta = np.arctan2(np.sin(g_theta), np.cos(g_theta)) 
    self.g_theta = g_theta
    # z position of second gate
    self.g1z = g1z
    
    # (end_dst, end_theta) is the polar coordinates from second gate to third gate
    self.end_dist = end_dist
    # modify end_theta to be third gate angle in global frame
    end_theta += g_theta
    end_theta = np.arctan2(np.sin(end_theta), np.cos(end_theta))
    self.end_theta = end_theta
    # z position of third gate
    self.g2z = g2z
    
    # optimal values that needed to be calculated for this test_case
    self.optimal_velocity = None
    self.optimal_time = None
    self.crashes = None
    self.skipped_waypoints = None
    self.average_error = None
    self.accel_range = None
  
  @staticmethod
  def get_header():
    # header for csv file
    # v0: velocity @ gate 1
    # z0: z position @ gate 1
    # d1: distance from gate 1 to gate 2
    # theta1: angle from gate 1 to gate 2
    # z1: z position @ gate 2
    # d2: distance from gate 2 to gate 3
    # theta2: angle from gate 2 to gate 3
    # z2: z position @ gate 3
    # v1: velocity for gate 1
    # t1: time for gate 1
    # crashes: number of crashes
    # skipped_waypoints: number of waypoints not visited
    # avg_err: average deviation from trajectory
    # accl_rng: range of acceleration normalized by max absolute acceleration
    return "v0,z0,d1,theta1,z1,d2,theta2,z2,v1,t1,crashes,skipped_waypoints,avg_err,accl_rng"
  
  def __str__(self):
    # returns all fields in class as a comma separated string ordered by the get_header output
    return f"{self.v},{self.z},{self.dist},{self.theta_rel[1]},{self.g1z},{self.end_dist},{self.theta_rel[2]},{self.g2z},{self.optimal_velocity},{self.optimal_time},{self.crashes},{self.skipped_waypoints},{self.average_error},{self.accel_range}"
