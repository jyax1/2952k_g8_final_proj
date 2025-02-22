POLICY_FREQS = [5, 10, 20] # Options of policy frequencies

# This offset is empirically determined, and describes the positional relationship between
# the gripper hand's center and the position of the eef site, which is the coordinate system
# the robosuite OSC_POSE controller expects for goal end-effector positions.
POSITIONAL_OFFSET = [-0.002, 0, 0.078]