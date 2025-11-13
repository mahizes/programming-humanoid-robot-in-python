'''In this exercise you need to implement the PID controller for joints of robot.

* Task:
    1. complete the control function in PIDController with prediction
    2. adjust PID parameters for NAO in simulation

* Hints:
    1. the motor in simulation can simple modelled by angle(t) = angle(t-1) + speed * dt
    2. use self.y to buffer model prediction
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'software_installation'))

import numpy as np
from collections import deque
from spark_agent import SparkAgent, JOINT_CMD_NAMES


class PIDController(object):
    '''a discretized PID controller, it controls an array of servos,
       e.g. input is an array and output is also an array
    '''
    def __init__(self, dt, size):
        '''
        @param dt: step time
        @param size: number of control values
        @param delay: delay in number of steps
        '''
        self.dt = dt
        # control signal (joint speeds)
        self.u = np.zeros(size)
        # previous errors e[k-1], e[k-2] for incremental PID
        self.e1 = np.zeros(size)
        self.e2 = np.zeros(size)
        # ADJUST PARAMETERS BELOW (reasonable defaults, can be tuned in pid_test.ipynb)
        delay = 0
        self.delay = delay
        # PID gains tuned via pid_test.ipynb
        self.Kp = 10.0
        self.Ki = 0.0
        self.Kd = 0.2
        # buffer for predicted joint angles; store one numpy array per time step
        self.y = deque([np.zeros(size)], maxlen=delay + 1)

    def set_delay(self, delay):
        '''
        @param delay: delay in number of steps
        '''
        self.delay = delay
        # keep the latest prediction (or zeros) and resize the buffer
        last = self.y[-1] if len(self.y) > 0 else np.zeros_like(self.u)
        self.y = deque([last] * (delay + 1), maxlen=delay + 1)

    def control(self, target, sensor):
        '''apply PID control
        @param target: reference values
        @param sensor: current values from sensor
        @return control signal (joint speeds)
        '''
        target = np.asarray(target, dtype=float)
        sensor = np.asarray(sensor, dtype=float)

        # initialise prediction buffer with the current sensor reading
        if len(self.y) == 0:
            self.y.append(sensor.copy())

        # predicted joint angle used for control (accounts for communication delay)
        # if delay == 0 this is simply the latest sensor reading
        y_pred = self.y[-1] if self.delay > 0 else sensor

        # current control error
        e = target - y_pred

        # incremental discrete PID:
        # u[k] = u[k-1] + Kp*(e[k]-e[k-1]) + Ki*dt*e[k] + Kd/dt*(e[k]-2e[k-1]+e[k-2])
        du = (
            self.Kp * (e - self.e1) +
            self.Ki * self.dt * e +
            self.Kd / self.dt * (e - 2.0 * self.e1 + self.e2)
        )
        self.u = self.u + du

        # update prediction buffer using simple motor model: angle(t)=angle(t-1)+speed*dt
        y_new = y_pred + self.u * self.dt
        self.y.append(y_new)

        # shift stored errors
        self.e2 = self.e1
        self.e1 = e

        return self.u


class PIDAgent(SparkAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PIDAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.joint_names = JOINT_CMD_NAMES.keys()
        number_of_joints = len(self.joint_names)
        self.joint_controller = PIDController(dt=0.01, size=number_of_joints)
        self.target_joints = {k: 0 for k in self.joint_names}

    def think(self, perception):
        action = super(PIDAgent, self).think(perception)
        '''calculate control vector (speeds) from
        perception.joint:   current joints' positions (dict: joint_id -> position (current))
        self.target_joints: target positions (dict: joint_id -> position (target)) '''
        joint_angles = np.asarray(
            [perception.joint[joint_id]  for joint_id in JOINT_CMD_NAMES])
        target_angles = np.asarray([self.target_joints.get(joint_id, 
            perception.joint[joint_id]) for joint_id in JOINT_CMD_NAMES])
        u = self.joint_controller.control(target_angles, joint_angles)
        action.speed = dict(zip(JOINT_CMD_NAMES.keys(), u))  # dict: joint_id -> speed
        return action


if __name__ == '__main__':
    agent = PIDAgent()
    agent.target_joints['HeadYaw'] = 1.0
    agent.run()
