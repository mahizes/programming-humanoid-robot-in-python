'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from numpy.matlib import matrix, identity
from math import cos, sin, sqrt
import numpy as np

from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'],
                       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'],
                       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
                       }

        # parameters: (x, y, z) translation from parent, and rotation axis
        # based on NAO H25 documentation
        self.params = {
            'HeadYaw': (0, 0, 0.1265, 'z'),
            'HeadPitch': (0, 0, 0, 'y'),
            
            'LShoulderPitch': (0, 0.098, 0.100, 'y'),
            'LShoulderRoll': (0, 0, 0, 'z'),
            'LElbowYaw': (0.105, 0.015, 0, 'x'),
            'LElbowRoll': (0, 0, 0, 'z'),
            
            'RShoulderPitch': (0, -0.098, 0.100, 'y'),
            'RShoulderRoll': (0, 0, 0, 'z'),
            'RElbowYaw': (0.105, -0.015, 0, 'x'),
            'RElbowRoll': (0, 0, 0, 'z'),
            
            'LHipYawPitch': (0, 0.050, -0.085, 'yp'), 
            'LHipRoll': (0, 0, 0, 'x'),
            'LHipPitch': (0, 0, 0, 'y'),
            'LKneePitch': (0, 0, -0.100, 'y'),
            'LAnklePitch': (0, 0, -0.1029, 'y'),
            'LAnkleRoll': (0, 0, 0, 'x'),
            
            'RHipYawPitch': (0, -0.050, -0.085, 'yp'), # Special axis
            'RHipRoll': (0, 0, 0, 'x'),
            'RHipPitch': (0, 0, 0, 'y'),
            'RKneePitch': (0, 0, -0.100, 'y'),
            'RAnklePitch': (0, 0, -0.1029, 'y'),
            'RAnkleRoll': (0, 0, 0, 'x')
        }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint
        
        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        x, y, z, axis = self.params[joint_name]
        
        # Translation matrix
        T_trans = identity(4)
        T_trans[0, 3] = x
        T_trans[1, 3] = y
        T_trans[2, 3] = z
        
        # Rotation matrix
        c = cos(joint_angle)
        s = sin(joint_angle)
        T_rot = identity(4)
        
        if axis == 'x':
            T_rot[1, 1] = c
            T_rot[1, 2] = -s
            T_rot[2, 1] = s
            T_rot[2, 2] = c
        elif axis == 'y':
            T_rot[0, 0] = c
            T_rot[0, 2] = s
            T_rot[2, 0] = -s
            T_rot[2, 2] = c
        elif axis == 'z':
            T_rot[0, 0] = c
            T_rot[0, 1] = -s
            T_rot[1, 0] = s
            T_rot[1, 1] = c
        elif axis == 'yp':
            # HipYawPitch: axis tilted 45° between +y and +z
            ux, uy, uz = 0.0, 1.0 / sqrt(2.0), 1.0 / sqrt(2.0)
            one_c = 1.0 - c
            # 3x3 rotation matrix from axis-angle (Rodrigues' formula)
            R00 = c + ux * ux * one_c
            R01 = ux * uy * one_c - uz * s
            R02 = ux * uz * one_c + uy * s
            R10 = uy * ux * one_c + uz * s
            R11 = c + uy * uy * one_c
            R12 = uy * uz * one_c - ux * s
            R20 = uz * ux * one_c - uy * s
            R21 = uz * uy * one_c + ux * s
            R22 = c + uz * uz * one_c

            T_rot[0, 0] = R00; T_rot[0, 1] = R01; T_rot[0, 2] = R02
            T_rot[1, 0] = R10; T_rot[1, 1] = R11; T_rot[1, 2] = R12
            T_rot[2, 0] = R20; T_rot[2, 1] = R21; T_rot[2, 2] = R22
            
        return T_trans * T_rot

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_name, chain_joints in self.chains.items():
            T = identity(4)
            for joint in chain_joints:
                angle = joints.get(joint, 0.0) # Use 0.0 if joint not in perception
                Tl = self.local_trans(joint, angle)
                T = T * Tl
                self.transforms[joint] = T

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
