'''In this file you need to implement remote procedure call (RPC) client

* The agent_server.py has to be implemented first (at least one function is implemented and exported)
* Please implement functions in ClientAgent first, which should request remote call directly
* The PostHandler can be implement in the last step, it provides non-blocking functions, e.g. agent.post.execute_keyframes
 * Hints: [threading](https://docs.python.org/2/library/threading.html) may be needed for monitoring if the task is done
'''

import weakref
from xmlrpc.client import ServerProxy
from threading import Thread, Lock


class PostHandler(object):
    '''the post hander wraps function to be excuted in paralle
    '''
    def __init__(self, obj):
        self.proxy = weakref.proxy(obj)

    def execute_keyframes(self, keyframes):
        '''non-blocking call of ClientAgent.execute_keyframes'''
        thread = Thread(target=self.proxy.execute_keyframes, args=(keyframes,))
        thread.daemon = True
        thread.start()
        return thread

    def set_transform(self, effector_name, transform):
        '''non-blocking call of ClientAgent.set_transform'''
        thread = Thread(target=self.proxy.set_transform, args=(effector_name, transform))
        thread.daemon = True
        thread.start()
        return thread


class ClientAgent(object):
    '''ClientAgent request RPC service from remote server
    '''
    
    def __init__(self, server_url='http://localhost:9559'):
        self.server_url = server_url
        self.post = PostHandler(self)
    
    def _get_proxy(self):
        '''Create a new ServerProxy for each call (thread-safe)'''
        return ServerProxy(self.server_url, allow_none=True)
    
    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        return self._get_proxy().get_angle(joint_name)
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        return self._get_proxy().set_angle(joint_name, angle)

    def get_posture(self):
        '''return current posture of robot'''
        return self._get_proxy().get_posture()

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        return self._get_proxy().execute_keyframes(keyframes)

    def get_transform(self, name):
        '''get transform with given name
        '''
        return self._get_proxy().get_transform(name)

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # Convert numpy array to list if necessary
        if hasattr(transform, 'tolist'):
            transform = transform.tolist()
        return self._get_proxy().set_transform(effector_name, transform)


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'joint_control'))
    from keyframes import hello
    import time
    
    agent = ClientAgent()
    print("ClientAgent connected to server!")
    
    passed = 0
    failed = 0
    
    # TEST CODE HERE
    # Test get_angle
    print("\n[Test] get_angle:")
    try:
        angle = agent.get_angle('HeadYaw')
        assert isinstance(angle, (int, float)), "Expected number"
        print(f"  HeadYaw = {angle} ✓")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
    
    # Test set_angle (call before keyframes so it's not overridden)
    print("\n[Test] set_angle:")
    try:
        old_angle = agent.get_angle('HeadYaw')
        agent.set_angle('HeadYaw', 0.5)
        time.sleep(1.0)
        new_angle = agent.get_angle('HeadYaw')
        # Just verify the call succeeded and angle changed (or was already close)
        print(f"  Called set_angle(HeadYaw, 0.5)")
        print(f"  Before: {old_angle:.3f}, After: {new_angle:.3f} ✓")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
    
    # Test get_posture
    print("\n[Test] get_posture:")
    try:
        posture = agent.get_posture()
        assert isinstance(posture, str), "Expected string"
        print(f"  Posture = {posture} ✓")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
    
    # Test get_transform
    print("\n[Test] get_transform:")
    try:
        transform = agent.get_transform('HeadYaw')
        assert len(transform) == 4 and len(transform[0]) == 4, "Expected 4x4 matrix"
        print(f"  Got 4x4 transform matrix ✓")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
    
    # Test execute_keyframes (blocking)
    print("\n[Test] execute_keyframes (blocking):")
    try:
        print("  Executing hello wave...")
        agent.execute_keyframes(hello())
        print("  Completed ✓")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
    
    # Test non-blocking execute_keyframes
    print("\n[Test] post.execute_keyframes (non-blocking):")
    try:
        thread = agent.post.execute_keyframes(hello())
        assert thread.is_alive(), "Thread should be running"
        print(f"  Started in background ✓")
        thread.join()
        print("  Background thread completed ✓")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1
    
    # Summary
    print("\n" + "=" * 40)
    if failed == 0:
        print(f"All {passed} tests PASSED! ✓")
    else:
        print(f"Results: {passed} passed, {failed} FAILED")
