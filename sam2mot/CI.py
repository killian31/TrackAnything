"""
Cross-object Interaction
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


class CI:
    def __init__(self):
        self.overlap_th = 0.8  # 遮挡发生阈值
        self.N = 10  # 计算方差的帧数
        self.similar_th = 0.2  # 判断两个logit是否相似的阈值

    def get_trackinfo(self, frame_idx, inference_state, device, upscale=True):  # TODO
        """
        从SAM2输出的inference_state中获取tracking当前帧获取的信息，包括masks、bboxes、logits、ids

        inputs:
        - frame_idx: 当前帧序号
        - inference_state: SAM2的memory bank
        - upscale: 是否上采样

        outputs:
        - tck_ids: ids of tracked objects from SAM2
        - tck_logits: logits of tracked objects from SAM2
        - tck_masks: masks of tracked objects from SAM2
        - tck_logits_history: N-frames-logits history of tracked objects from SAM2
        """
        tck_ids = []
        tck_masks = []
        tck_logits = []
        tck_bboxes = []
        tck_logits_history = []
        H_orig, W_orig = inference_state["video_height"], inference_state["video_width"]

        for obj_id in inference_state["obj_ids"]:
            obj_idx = inference_state["obj_id_to_idx"][obj_id]
            out_dict = inference_state["output_dict_per_obj"][obj_idx]

            # 合并所有已输出帧（包含 conditioned 和 non-conditioned）
            all_outputs = {}
            all_outputs.update(out_dict.get("cond_frame_outputs", {}))
            all_outputs.update(out_dict.get("non_cond_frame_outputs", {}))

            # 当前帧上该目标的输出（优先选择 conditioned 输出）
            if frame_idx in out_dict["cond_frame_outputs"]:
                out = out_dict["cond_frame_outputs"][frame_idx]
            elif frame_idx in out_dict["non_cond_frame_outputs"]:
                out = out_dict["non_cond_frame_outputs"][frame_idx]
            else:
                continue  # 该对象未被传播到这一帧

            mask = out["pred_masks"].to(device)

            # 上采样到原始分辨率
            if upscale:  # 上采样到原始视频尺寸
                mask = F.interpolate(
                    mask, size=(H_orig, W_orig), mode="bilinear", align_corners=False
                )[0, 0]

            # 二值化掩码
            mask = mask > 0

            logit = out["object_score_logits"]

            # 计算 bbox 等同前，只是 xs, ys 需转到 CPU 或在 GPU 上用 torch.nonzero
            ysx = mask.nonzero()  # Tensor [N,2] 在 GPU
            if ysx.shape[0] > 0:
                y1x1 = ysx.min(dim=0)[0]
                y2x2 = ysx.max(dim=0)[0]
                y1, x1 = y1x1[0].item(), y1x1[1].item()
                y2, x2 = y2x2[0].item(), y2x2[1].item()
                bbox = torch.tensor(
                    [[x1, y1, x2, y2]], dtype=torch.float32, device=mask.device
                )
            else:
                bbox = torch.tensor(
                    [[0, 0, 0, 0]], dtype=torch.float32, device=mask.device
                )

            # 获取最近 N 帧的 logits 历史（只选择 <= frame_idx）
            valid_frames = [f for f in all_outputs.keys() if f <= frame_idx]
            valid_frames.sort()
            recent_frames = valid_frames[-self.N :]

            history_logits = [
                all_outputs[frame_id]["object_score_logits"].item()
                for frame_id in recent_frames
            ]

            tck_ids.append(obj_id)
            tck_logits.append(logit)
            tck_masks.append(mask)
            tck_bboxes.append(bbox)
            tck_logits_history.append(history_logits)

        return tck_ids, tck_logits, tck_masks, tck_logits_history, tck_bboxes

    def remove_occlusion(self, inference_state, frame_id, device):
        """
        判定物体是否发生遮挡，确定被遮挡物体并删去

        inputs:
        - inference_state: memory bank from SAM2
        - frame_id: id of the current frame

        outputs:
        - removed: the list of ids of occluded objects, which will be removed from memory bank
        """
        tck_ids, tck_logits, tck_masks, tck_logits_history, tck_bboxes = (
            self.get_trackinfo(
                frame_idx=frame_id, inference_state=inference_state, device=device
            )
        )
        removed = []
        for i in range(len(tck_ids)):
            for j in range(i + 1, len(tck_ids)):
                id_i, id_j = tck_ids[i], tck_ids[j]
                mask_i, mask_j = tck_masks[i], tck_masks[j]
                bbox_i, bbox_j = tck_bboxes[i], tck_bboxes[j]
                logit_i, logit_j = tck_logits[i], tck_logits[j]
                logit_history_i, logit_history_j = (
                    tck_logits_history[i],
                    tck_logits_history[j],
                )
                miou = self.miou(
                    mask1=mask_i, mask2=mask_j
                )  # 为什么要用miou呢？？？？SAM2生成的miou几乎不可能重叠呀
                # biou = box_iou(bbox_i, bbox_j)[0, 0].item()
                if miou > self.overlap_th:  # 发生occlusion
                    if abs(logit_i - logit_j) > self.similar_th:  # 相差较大
                        removed.append(id_i if logit_i < logit_j else id_j)
                    else:  # 相差较小，需要靠方差来判断
                        removed.append(
                            id_i
                            if np.var(logit_history_i) > np.var(logit_history_j)
                            else id_j
                        )

        return removed

    def miou(self, mask1, mask2):
        """
        计算两个mask之间的mask iou
        """
        inter = torch.logical_and(mask1, mask2).sum().float()
        union = torch.logical_or(mask1, mask2).sum().float()
        return inter / union if union > 0 else 0.0
