from action_extractor.point_cloud.config import *
import random

def change_policy_freq(policy_freq):
    current_policy_freq = policy_freq

    # Filter out the old policy_freq
    valid_options = [opt for opt in POLICY_FREQS if opt != current_policy_freq]

    # Randomly pick a new policy_freq
    return random.choice(valid_options)