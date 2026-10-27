"""Scene splits and the day/night tag (ADR 0004, ADR 0005).

Splits are always by scene, never by frame: keyframes are 0.5 s apart, so neighbouring
frames are near-duplicates and a frame-level split leaks training data into validation.
"""

from typing import Any

from nuscenes.utils.splits import create_splits_scenes

# Dev-only validation scenes, one day and one night. The official `mini_val` has no night
# scene, so this split exists only to smoke-test the day/night evaluation code on mini.
# Never report results on it.
_MINI_DEV_VAL = ("scene-0103", "scene-1100")

_OFFICIAL = ("train", "val", "mini_train", "mini_val")


def scene_is_night(scene: dict[str, Any]) -> bool:
    """Whether a nuScenes `scene` record is a night scene (its description mentions night)."""
    return "night" in scene["description"].lower()


def get_split_scenes(split: str) -> list[str]:
    """Scene names (like `scene-0061`) of a split.

    Official splits: `train`, `val` (v1.0-trainval) and `mini_train`, `mini_val` (v1.0-mini).
    Dev-only splits on mini: `mini_dev_train`, `mini_dev_val`. Do not mix mini splits with
    trainval splits: two of the eight `mini_train` scenes are official validation scenes.
    """
    official = create_splits_scenes()
    splits = {name: list(official[name]) for name in _OFFICIAL}
    mini_scenes = official["mini_train"] + official["mini_val"]
    splits["mini_dev_train"] = [name for name in mini_scenes if name not in _MINI_DEV_VAL]
    splits["mini_dev_val"] = list(_MINI_DEV_VAL)
    if split not in splits:
        raise ValueError(f"unknown split {split!r}; choose one of {sorted(splits)}")
    return splits[split]
