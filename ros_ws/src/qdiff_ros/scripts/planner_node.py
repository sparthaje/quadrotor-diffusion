#!/usr/bin/env python3
import pickle
import numpy as np
import requests

import rospy
from crazyswarm.msg import FullState
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped


class Planner:
    def __init__(self):
        rospy.init_node('planner')

        self.course = np.array([
            [-0.4520511, -1.27241438, 0.3, 0.],
            [-0.622861, -0.6179413, 0.3, 0.78143325],
            [-0.44599258, 0.65827566, 0.525, -0.97531991],
            [0.77898342, -0.41174775, 0.525, 2.65535065]
        ])

        self.current_traj: list[np.ndarray] = None
        self.next_traj: list[np.ndarray] = None

        # Step along current trajectory
        self.step = 0

        # Gate current traj starts at
        self.gate_idx = 0

        self.update_next_traj()
        self.current_traj = self.next_traj
        self.next_traj = None
        self.update_next_traj()

        self.timer = rospy.Timer(rospy.Duration(1./30.), self.update_next_traj)
        self.vicon_sub = rospy.Subscriber('vicon/cf1/cf1', TransformStamped, self.cf_pose_callback)
        self.cf_publisher = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)
        self.rate = rospy.Rate(30)
        print("Initialized Planner")

    def update_next_traj(self, event=None):
        if self.next_traj is not None:
            return

        # If no trajectory exists, provide the full course so the 0th position can be used as the
        # local conditioning and the rest of it can be used for global conditioning
        global_context = self.course
        if self.current_traj is not None:
            global_context = np.vstack((self.course[self.gate_idx + 1:], self.course[1:self.gate_idx]))

        data = pickle.dumps((self.current_traj, global_context))
        r = requests.post("http://localhost:5000/plan", data=data)
        self.next_traj = pickle.loads(r.content)

        self.gate_idx += 1
        if self.gate_idx == len(self.course):
            self.gate_idx = 1

    def cf_pose_callback(self, msg: TransformStamped):
        self.course[0][0] = msg.transform.translation.x
        self.course[0][1] = msg.transform.translation.y
        self.course[0][2] = msg.transform.translation.z

    def run(self):
        while not rospy.is_shutdown():

            position = self.current_traj[0][self.step]
            velocity = self.current_traj[1][self.step]
            acceleration = self.current_traj[2][self.step]

            state = FullState()
            state.pose.position.x = position[0]
            state.pose.position.y = position[1]
            state.pose.position.z = position[2]

            state.twist.linear.x = velocity[0]
            state.twist.linear.y = velocity[1]
            state.twist.linear.z = velocity[2]

            state.acc.x = acceleration[0]
            state.acc.y = acceleration[1]
            state.acc.z = acceleration[2]

            self.cf_publisher.publish(state)

            self.step += 1
            # Finished current trajectory
            if self.step == self.current_traj[0].shape[0]:
                self.current_traj = self.next_traj
                self.next_traj = None
                self.step = 0

            self.rate.sleep()


if __name__ == '__main__':
    planner = Planner()
    planner.run()
