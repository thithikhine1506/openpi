import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Kinova Gen3 (7-DoF) + Robotiq 2F-85.
#
# STATE  (8,) = [j0..j6 joint positions (rad), gripper]   gripper: 0.0 open -> 1.0 closed
# ACTION (8,) = [j0..j6 joint position targets (rad), gripper]
#
# Joint POSITION targets, not velocities: openpi recommends position actions
# (velocities are much harder to simulate and transfer poorly across embodiments).
#
# If you later switch to Cartesian delta actions, ACTION_DIM stays 7 (dx,dy,dz,drx,dry,drz,gripper)
# and you must change ACTION_DIM below AND the slice in KinovaOutputs.
ACTION_DIM = 8
STATE_DIM = 8


def make_kinova_example() -> dict:
    """Creates a random input example for the Kinova policy. Used for shape-checking
    the server without a robot attached."""
    return {
        "observation/state": np.random.rand(STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the red block and place it in the bowl",
    }


def _parse_image(image) -> np.ndarray:
    """LeRobot stores images as float32 (C,H,W); the model wants uint8 (H,W,C).
    Keep this verbatim -- it is skipped automatically during live inference."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class KinovaInputs(transforms.DataTransformFn):
    """Converts Kinova Gen3 dataset/observation dicts into the model's expected format.
    Used for BOTH training and inference -- they must match exactly, or the policy will
    train on one representation and be served another."""

    # Determines which model will be used. Do not change.
    model_type: _model.ModelType

    # Set False if you are running with only the fixed third-person camera.
    # The Gen3's built-in vision module gives you a wrist stream for free, but it is
    # optional -- a third-person view is the only mandatory one.
    has_wrist_image: bool = True

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        if self.has_wrist_image:
            wrist_image = _parse_image(data["observation/wrist_image"])
            wrist_mask = np.True_
        else:
            wrist_image = np.zeros_like(base_image)
            # Padded images are masked False for pi0 (flow matching) and True for pi0-FAST.
            wrist_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Gen3 is a single arm -- no right wrist camera exists. Always padded.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": wrist_mask,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class KinovaOutputs(transforms.DataTransformFn):
    """Converts model output back to the Kinova action space. Inference only.

    The model emits a padded action vector; slice out the real dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
