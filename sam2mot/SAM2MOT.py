import os
from sam2.build_sam import build_sam2_video_predictor
from GroundingDINO.groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate,
)


class SAM2MOT:
    def __init__(
        self,
        video_path,
        detector_cfg,
        detector_checkpoint,
        sam2_cfg,
        sam2_checkpoint,
        device="cuda",
        TEXT_PROMPT="person",
        BOX_TRESHOLD=0.35,
        TEXT_TRESHOLD=0.25,
    ):
        self.device = device
        self.text_prompt = TEXT_PROMPT

        self.detector = load_model(
            model_checkpoint_path=detector_checkpoint,
            model_config_path=detector_cfg,
            device=device,
        )

        self.sam2_predictor = build_sam2_video_predictor(
            sam2_cfg, sam2_checkpoint, device=device
        )

        self.inference_state = self.sam2_predictor.init_state(video_path)
