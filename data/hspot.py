import os
from data.dancetrack import DanceTrack


class HSPOT(DanceTrack):
    def __init__(
        self,
        data_root: str = "./datasets/",
        sub_dir: str = "hspot",
        split: str = "train",
        load_annotation: bool = True,
    ):
        super().__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
        return

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        return os.path.join(
            sequence_dir, "img1", f"{frame_idx + 1:06d}.jpg"
        )  # different from DanceTrack
