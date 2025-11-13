'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''


from angle_interpolation import AngleInterpolationAgent
from keyframes import hello

import os
import pickle


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.posture = 'unknown'
        self.posture_classifier = None
        self.pose_classes = []

        # try to load the trained classifier (see learn_posture.ipynb)
        base_dir = os.path.abspath(os.path.dirname(__file__))
        clf_path = os.path.join(base_dir, 'robot_pose.pkl')
        data_dir = os.path.join(base_dir, 'robot_pose_data')

        if os.path.exists(clf_path):
            try:
                with open(clf_path, 'rb') as f:
                    self.posture_classifier = pickle.load(f)
            except Exception:
                # if loading fails we simply keep classifier as None
                self.posture_classifier = None

        if os.path.isdir(data_dir):
            # order must match training (see learn_posture.ipynb, classes = listdir(...))
            try:
                self.pose_classes = os.listdir(data_dir)
            except Exception:
                self.pose_classes = []

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'

        # if classifier is not available, we cannot recognize the posture
        if self.posture_classifier is None:
            return posture

        joint = perception.joint
        imu = getattr(perception, 'imu', [0.0, 0.0])

        # feature vector must match the training features in learn_posture.ipynb:
        # ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch',
        #  'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch',
        #  'AngleX', 'AngleY']
        features = [
            joint.get('LHipYawPitch', 0.0),
            joint.get('LHipRoll', 0.0),
            joint.get('LHipPitch', 0.0),
            joint.get('LKneePitch', 0.0),
            joint.get('RHipYawPitch', 0.0),
            joint.get('RHipRoll', 0.0),
            joint.get('RHipPitch', 0.0),
            joint.get('RKneePitch', 0.0),
            imu[0],
            imu[1],
        ]

        try:
            idx = int(self.posture_classifier.predict([features])[0])
            if 0 <= idx < len(self.pose_classes):
                posture = self.pose_classes[idx]
        except Exception:
            posture = 'unknown'

        return posture

if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
