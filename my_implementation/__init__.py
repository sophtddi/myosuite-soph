import os
from myosuite.utils import gym

register = gym.register
curr_dir = os.path.dirname(os.path.abspath(__file__))

register(
    id='myoOSLRunTrackImitation-v0',
    entry_point='my_implementation.envs.my_run_track_v0:RunTrackImitation',
    max_episode_steps=1000,
    kwargs={
        # reprend les kwargs de myoChallengeOslRunFixed-v0, en pointant vers TES assets
        'model_path': curr_dir + '/../myosuite/envs/myo/assets/leg/myoosl_runtrack.xml',
        'normalize_act': True,
        'reset_type': 'random',
        'terrain': 'flat',
        'end_pos': -15,
        'frame_skip': 5,
        'start_pos': 14,
        'init_pose_path': curr_dir + '/../myosuite/envs/myo/assets/leg/sample_gait_cycle.csv',
        'max_episode_steps': 1000,
    }
)