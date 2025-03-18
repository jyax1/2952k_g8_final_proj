# [ROBOSUITE] reasonable options of policy frequencies
POLICY_FREQS = [1, 2, 4, 5, 10, 20]

# [ROBOSUITE] This offset is empirically determined, and describes the positional relationship between
# the gripper hand's center and the position of the eef site, which is the coordinate system
# the robosuite OSC_POSE controller expects for goal end-effector positions.
POSITIONAL_OFFSET = [-0.002, 0, 0.078]

# [ROBOSUITE] Camera height and width
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640

# [ROBOSUITE] ameras to use for downstream policy and point cloud extraction
CAMERAS_FOR_POLICY = ['frontview', 'fronttableview', 'agentview', 'robot0_eye_in_hand']
ADDITIONAL_CAMERAS_FOR_POINT_CLOUD = ['frontview',
                                    'squared0viewfar','squared0view2far', 
                                    'squared0view3far', 'squared0view4far',
                                    'birdview', 'sideview', 'sideview2']