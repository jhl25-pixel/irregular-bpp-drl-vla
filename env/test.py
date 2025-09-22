from openpi.training import config as _config
print("OpenPI config has been imported successfully.")
from openpi.policies import policy_config
print("OpenPI policies version has been imported successfully.")
from openpi.shared import download
print("OpenPI shared version has been imported successfully.")

config = _config.get_config("pi05_droid")
print("Configuration for pi05_droid has been retrieved successfully.")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
print("Configuration and checkpoint directory have been downloaded successfully.")
# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print("Trained policy has been created successfully.")
# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "observation/wrist_image_left": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],

    "prompt": "pick up the fork"
}
print("Example input prepared successfully.")
action_chunk = policy.infer(example)["actions"]