'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
import numpy as np
from math import cos, sin, sqrt, pi


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        chain = self.chains.get(effector_name, [])
        if not chain:
            return []

        n_joints = len(chain)
        # Initial guess: slight bend to avoid singularity at 0
        q = np.full(n_joints, 0.1) 
        
        target_pos = np.array(transform[:3, 3]).flatten()
        target_rot = np.array(transform[:3, :3])
        
        max_iter = 50
        error_threshold = 1e-4
        
        for iter_idx in range(max_iter):
            # Forward Kinematics for the chain with current q
            T_current = np.identity(4)
            
            # Store joint axes and positions for Jacobian
            z_indices = []
            p_indices = []
            
            for i, joint in enumerate(chain):
                x, y, z, axis = self.params[joint]
                
                # Translation to joint frame
                T_trans = np.identity(4)
                T_trans[0, 3] = x
                T_trans[1, 3] = y
                T_trans[2, 3] = z
                
                # Transform before rotation (to get axis and position)
                T_pre = np.dot(T_current, T_trans)
                
                # Determine axis vector
                if isinstance(axis, str):
                    if axis == 'x': u = np.array([1, 0, 0])
                    elif axis == 'y': u = np.array([0, 1, 0])
                    elif axis == 'z': u = np.array([0, 0, 1])
                    elif axis == 'yp': u = np.array([0, 1/sqrt(2), 1/sqrt(2)]) # Consistent with FK
                    else: u = np.array([0, 0, 1])
                else:
                    u = np.array(axis)
                
                # Axis and position in base frame
                z_idx = np.dot(T_pre[:3, :3], u)
                p_idx = T_pre[:3, 3]
                
                z_indices.append(z_idx)
                p_indices.append(p_idx)
                
                # Apply rotation
                angle = q[i]
                c = cos(angle)
                s = sin(angle)
                
                ux, uy, uz = u
                # Skew-symmetric matrix K
                K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
                # Rodrigues' rotation formula
                R_rot = np.identity(3) + s * K + (1 - c) * np.dot(K, K)
                
                T_rot = np.identity(4)
                T_rot[:3, :3] = R_rot
                
                # Update current transform
                T_current = np.dot(T_pre, T_rot)
            
            T_end = T_current
            p_end = T_end[:3, 3]
            R_end = T_end[:3, :3]
            
            # Position error
            e_pos = target_pos - p_end
            
            # Orientation error
            n = R_end[:, 0]; s_vec = R_end[:, 1]; a = R_end[:, 2]
            nd = target_rot[:, 0]; sd = target_rot[:, 1]; ad = target_rot[:, 2]
            
            e_rot = 0.5 * (np.cross(n, nd) + np.cross(s_vec, sd) + np.cross(a, ad))
            
            error = np.concatenate([e_pos, e_rot])
            
            err_norm = np.linalg.norm(error)
            # Optional debug print for the first iteration; set self.debug = True to enable
            if iter_idx == 0 and getattr(self, "debug", False):
                print(f"IK start error for {effector_name}: {err_norm}")
            
            if err_norm < error_threshold:
                break
            
            # Jacobian
            J = np.zeros((6, n_joints))
            for i in range(n_joints):
                z_i = z_indices[i]
                p_i = p_indices[i]
                
                J[:3, i] = np.cross(z_i, p_end - p_i)
                J[3:, i] = z_i
            
            # Solve for joint update using the Jacobian pseudo-inverse
            dq = np.dot(np.linalg.pinv(J), error)
            
            q = q + 0.5 * dq # Gain
            
        # Normalize to [-pi, pi]
        q = (q + np.pi) % (2 * np.pi) - np.pi
            
        # Debug: Print computed position vs target
        # T_check = self.compute_fk_for_chain(chain, q.tolist())
        # print(f"IK Error Pos: {np.linalg.norm(target_pos - T_check[:3, 3])}")
            
        return q.tolist()

    def compute_fk_for_chain(self, chain, angles):
        T = np.identity(4)
        for i, joint in enumerate(chain):
            x, y, z, axis = self.params[joint]
            T_trans = np.identity(4)
            T_trans[0, 3] = x
            T_trans[1, 3] = y
            T_trans[2, 3] = z
            
            angle = angles[i]
            c = cos(angle)
            s = sin(angle)
            
            if isinstance(axis, str):
                 if axis == 'x':
                     u = np.array([1, 0, 0])
                 elif axis == 'y':
                     u = np.array([0, 1, 0])
                 elif axis == 'z':
                     u = np.array([0, 0, 1])
                 elif axis == 'yp':
                     # same tilted HipYawPitch axis as in forward_kinematics
                     u = np.array([0, 1/np.sqrt(2.0), 1/np.sqrt(2.0)])
                 else:
                     u = np.array([0, 0, 1])
            else:
                 u = np.array(axis)
            
            ux, uy, uz = u
            K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
            R_rot = np.identity(3) + s * K + (1 - c) * np.dot(K, K)
            T_rot = np.identity(4)
            T_rot[:3, :3] = R_rot
            
            T = np.dot(T, np.dot(T_trans, T_rot))
        return T

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        joint_angles = self.inverse_kinematics(effector_name, transform)
        if joint_angles:
            names = self.chains[effector_name]
            times = [[1.0] for _ in names]
            keys = [[angle] for angle in joint_angles]
            self.keyframes = (names, times, keys)

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[1, 3] = 0.05
    T[2, 3] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
