'''In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'kinematics'))

from inverse_kinematics import InverseKinematicsAgent
from xmlrpc.server import SimpleXMLRPCServer
from socketserver import ThreadingMixIn
from threading import Thread
import numpy as np


class ThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    '''Multi-threaded XML-RPC server to handle concurrent requests'''
    pass


class ServerAgent(InverseKinematicsAgent):
    '''ServerAgent provides RPC service
    '''
    
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ServerAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        
        # Flag to track if keyframes are still executing
        self.keyframes_executing = False
        
        # Start RPC server in a separate thread (multi-threaded to handle concurrent requests)
        self.rpc_server = ThreadedXMLRPCServer(('localhost', 9559), allow_none=True, logRequests=False)
        self.rpc_server.register_function(self.get_angle, 'get_angle')
        self.rpc_server.register_function(self.set_angle, 'set_angle')
        self.rpc_server.register_function(self.get_posture, 'get_posture')
        self.rpc_server.register_function(self.execute_keyframes, 'execute_keyframes')
        self.rpc_server.register_function(self.get_transform, 'get_transform')
        self.rpc_server.register_function(self.set_transform, 'set_transform')
        
        # Start RPC server thread
        self.rpc_thread = Thread(target=self.rpc_server.serve_forever)
        self.rpc_thread.daemon = True
        self.rpc_thread.start()
        print("RPC Server started on localhost:9559")
    
    def think(self, perception):
        # Check if keyframes execution is complete
        if self.keyframes_executing:
            names, times, keys = self.keyframes
            if names and times:
                # Get max duration from all joints
                max_time = max(t[-1] for t in times if t)
                elapsed = perception.time - self.motion_start_time if self.motion_start_time else 0
                if elapsed >= max_time:
                    self.keyframes_executing = False
        
        return super(ServerAgent, self).think(perception)
    
    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        return self.perception.joint.get(joint_name, 0.0)
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        self.target_joints[joint_name] = angle
        return True

    def get_posture(self):
        '''return current posture of robot'''
        return self.posture

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        # Convert lists back to proper format (XML-RPC converts tuples to lists)
        names = keyframes[0]
        times = keyframes[1]
        keys = keyframes[2]
        
        self.keyframes = (names, times, keys)
        self.motion_start_time = None  # Reset to trigger new motion
        self.keyframes_executing = True
        
        # Block until keyframes are done
        while self.keyframes_executing:
            import time
            time.sleep(0.02)
        
        return True

    def get_transform(self, name):
        '''get transform with given name
        '''
        transform = self.transforms.get(name, np.identity(4))
        # Convert numpy matrix to list for XML-RPC serialization
        return np.asarray(transform).tolist()

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # Convert list back to numpy array
        transform_matrix = np.array(transform)
        self.set_transforms(effector_name, transform_matrix)
        
        # Wait for motion to complete (blocking)
        names, times, keys = self.keyframes
        if names and times:
            max_time = max(t[-1] for t in times if t)
            import time
            time.sleep(max_time + 0.5)  # Wait for motion duration plus buffer
        
        return True

if __name__ == '__main__':
    agent = ServerAgent()
    agent.run()
