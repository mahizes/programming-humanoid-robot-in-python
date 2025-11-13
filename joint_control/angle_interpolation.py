'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.motion_start_time = None

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        # some keyframes (e.g. stand-up motions) provide LHipYawPitch but not RHipYawPitch.
        # copy it only if LHipYawPitch is present for this motion.
        if 'LHipYawPitch' in target_joints:
            target_joints['RHipYawPitch'] = target_joints['LHipYawPitch']
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        names, times, keys = keyframes

        # no keyframes configured → keep current targets
        if not names:
            return target_joints

        # initialise motion reference time on first call
        if self.motion_start_time is None:
            self.motion_start_time = perception.time

        t_now = perception.time - self.motion_start_time

        for idx, joint_name in enumerate(names):
            t_list = times[idx]
            k_list = keys[idx]
            # extract raw angles, ignoring Bezier handle data if present
            angles = [k if isinstance(k, (int, float)) else k[0] for k in k_list]

            if not t_list:
                continue

            # before first keyframe
            if t_now <= t_list[0]:
                angle = angles[0]
            # after last keyframe
            elif t_now >= t_list[-1]:
                angle = angles[-1]
            else:
                # find segment for linear interpolation
                angle = angles[-1]
                for j in range(1, len(t_list)):
                    if t_now <= t_list[j]:
                        t0, t1 = t_list[j - 1], t_list[j]
                        a0, a1 = angles[j - 1], angles[j]
                        ratio = (t_now - t0) / (t1 - t0) if t1 > t0 else 0.0
                        angle = a0 + ratio * (a1 - a0)
                        break

            target_joints[joint_name] = angle

        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
