"""
Trajectory Manager System
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import torch


class Tracklet:
    """
    每个追踪物体的信息，包括物体的id、bbox、logit、logits_history、status、last_seen
    """

    def __init__(self, track_id, logit, bbox, frame_id):
        self.id = track_id
        self.bbox = bbox  # tck_bbox
        self.logits_history = [logit]
        self.last_seen = frame_id
        # 设置状态的阈值
        self.reliable_th = 8.0
        self.pending_th = 6.0
        self.suspicious_th = 2.0
        self.lost_th = self.suspicious_th
        self.status = "reliable"

    def update_logit(self, bbox, score, frame_id):
        """
        更新tracklet，主要更新status(该逻辑下认定所有non-lost都需要tracking)
        """
        self.bbox = bbox
        current_status = self.get_status(logit=score)
        self.logits_history.append(score)
        if self.status != current_status:  # 状态改变
            if self.status == "lost":  # lost->non-lost
                self.last_seen = frame_id
                self.status = current_status
            elif current_status == "lost":  # non-lost->lost
                self.status = current_status
            else:  # nonlost->nonlost
                self.last_seen = frame_id
                self.status = current_status
        else:  # 状态不变
            if self.status != "lost":  # 依然可见
                self.last_seen = frame_id

    def get_status(self, logit):
        """
        根据当前的logit获取当前的状态
        """
        if logit > self.reliable_th:
            return "reliable"
        elif logit > self.pending_th and logit <= self.reliable_th:
            return "pending"
        elif logit > self.suspicious_th and logit <= self.pending_th:
            return "suspicious"
        else:
            return "lost"


class TMS:
    def __init__(self):
        self.tracklets = {}  # 为每个tck_object设定对应的Tracklet对象
        self.next_id = 1  # 新tck_object的id
        self.max_missing = 25  # lost的最大容忍帧数
        self.iou_thresh = 0.3  # 进行匈牙利算法匹配时所需的阈值
        self.confidence_th = 0.5  # 保留的detector生成的bbox的logits阈值
        self.r = 0.5  # 判断是否触发object addition
        self.obj_id_to_tck_id = {}  # obj_id->tck_id映射表

    def TrajectroyManage(
        self, frame_id, inference_state, det_bboxes, det_logits, device="device"
    ):
        """
        传入detector和inference_state进行轨迹管理
        """
        # 获取tracking的信息
        tck_ids, tck_logits, tck_bboxes, tracked_mask, _ = self.get_trackinfo(
            frame_idx=frame_id, inference_state=inference_state, device=device
        )
        # 根据detector返回的信息更新TMS的Tracklet
        self.update(
            tck_ids=tck_ids,
            tck_logits=tck_logits,
            tck_bboxes=tck_bboxes,
            frame_id=frame_id,
        )
        # 将det和tck匹配起来，保留大于0.5confidence的det_bboxes
        matched_pairs, unmatched_dets, det_bboxes, det_logits = self.get_bboxes(
            det_bboxes=det_bboxes, det_logits=det_logits
        )
        # object addition
        added = self.object_addition(
            unmatched_dets=unmatched_dets,
            det_bboxes=det_bboxes,
            det_logtis=det_logits,
            frame_id=frame_id,
            tracked_mask=tracked_mask,
            device=device,
        )
        # object removal
        removed = self.object_removal(frame_id=frame_id)
        # quality reconstruction
        reconstructed = self.quality_reconstruction(
            matched_pairs=matched_pairs, det_bboxes=det_bboxes
        )

        new_prompts = np.empty((0, 4))
        obj_ids = []
        if added:
            new_prompts = np.append(
                new_prompts,
                np.array(
                    [self.tracklets[obj_id].bbox for obj_id in added], dtype=np.float32
                ),
                axis=0,
            )
            obj_ids.extend(added)
        if reconstructed:
            new_prompts = np.append(
                new_prompts,
                np.array(
                    [self.tracklets[obj_id].bbox for obj_id in reconstructed],
                    dtype=np.float32,
                ),
                axis=0,
            )
            obj_ids.extend(reconstructed)
        return new_prompts, obj_ids, removed

    def get_trackinfo(
        self, frame_idx, inference_state, upscale=True, device="cuda"
    ):  # TODO
        """
        从 SAM2 输出的 inference_state 中获取当前帧的跟踪信息
        inputs
        - frame_idx: 当前帧序号
        - inference_state: SAM2的memory bank
        - upscale: 是否进行上采样

        outputs
        - tck_ids:   List[int]  每个目标的 obj_id
        - tck_logits: List[Tensor] 每个目标的 upscaled logits map ([H, W])
        - tck_bboxes: List[List[int]] 每个目标的 bbox [x1,y1,x2,y2]
        - tracked_mask: Tensor([H, W]) 全局已跟踪区域的二值掩码
        """
        tck_ids = []
        tck_logits = []
        tck_bboxes = []
        tck_masks = []

        self.H_orig, self.W_orig = (
            inference_state["video_height"],
            inference_state["video_width"],
        )

        # 初始化全局掩码
        tracked_mask = torch.zeros(
            (self.H_orig, self.W_orig), dtype=torch.bool, device=device
        )

        for obj_id in inference_state["obj_ids"]:
            obj_idx = inference_state["obj_id_to_idx"][obj_id]
            out_dict = inference_state["output_dict_per_obj"][obj_idx]

            # 获取当前帧的输出
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
                    mask,
                    size=(self.H_orig, self.W_orig),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

            # 二值化掩码
            mask = mask > 0

            logit = out["object_score_logits"]

            # 将该 mask 并入全局 tracked_mask
            tracked_mask = tracked_mask | mask

            # 计算 bbox 等同前，只是 xs, ys 需转到 CPU 或在 GPU 上用 torch.nonzero
            ysx = mask.nonzero()  # Tensor [N,2] 在 GPU
            if ysx.shape[0] > 0:
                y1x1 = ysx.min(dim=0)[0]
                y2x2 = ysx.max(dim=0)[0]
                y1, x1 = int(y1x1[0]), int(y1x1[1])
                y2, x2 = int(y2x2[0]), int(y2x2[1])
                bbox = [x1, y1, x2, y2]
            else:
                bbox = [0, 0, 0, 0]

            tck_ids.append(obj_id)
            tck_logits.append(logit)  # 保持在GPU
            tck_bboxes.append(bbox)
            tck_masks.append(mask)
        return tck_ids, tck_logits, tck_bboxes, tracked_mask, tck_masks

    def update(self, tck_ids, tck_logits, tck_bboxes, frame_id):
        """
        从SAM2输出的tracking物体数据，更新TSM中每个目标跟踪物体的tracklet

        inputs:
        - inference_state: SAM2的memory bank
        - frame_id: 当前帧序号
        """
        for id, logit, bbox in zip(tck_ids, tck_logits, tck_bboxes):
            if id not in self.tracklets:
                self.tracklets[id] = Tracklet(
                    track_id=id, logit=logit, bbox=bbox, frame_id=frame_id
                )
            self.tracklets[id].update_logit(bbox, logit, frame_id)

    def get_bboxes(self, det_bboxes, det_logits):
        """
        获取detectors生成的det_bboxes后，将tracking bbox和detection bbox进行配对，获取已配对的id pairs和未配对的det_bbox序号
        inputs:
        - det_bboxes: bboxes from detectors
        - det_logits: logits from detectors

        outputs:
        - matched_pairs: [(det_id, tck_id)]
        - unmatched_dets: list of unmatched det_idx
        - det_bboxes: high-confidence det_bboxes
        """
        det_logits = det_logits.tolist()
        det_bboxes = det_bboxes.tolist()
        det_satisfied_ids = [
            i for i in range(len(det_logits)) if det_logits[i] > self.confidence_th
        ]  # confidence_th=0.5,保留大于0.5的det_bboxes
        det_logits = [det_logits[i] for i in det_satisfied_ids]
        det_bboxes = [det_bboxes[i] for i in det_satisfied_ids]

        # det_bboxes去除归一化
        for det_idx in range(len(det_bboxes)):
            x1_norm, y1_norm, x2_norm, y2_norm = det_bboxes[det_idx]
            x1 = int(round(x1_norm * self.W_orig))
            y1 = int(round(y1_norm * self.H_orig))
            x2 = int(round(x2_norm * self.W_orig))
            y2 = int(round(y2_norm * self.H_orig))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.W_orig, x2), min(self.H_orig, y2)
            det_bboxes[det_idx] = [x1, y1, x2, y2]

        active_tracklets = [
            t for t in self.tracklets.values() if t.status != "lost"
        ]  # 活跃的tracking对象
        tck_bboxes = [t.bbox for t in active_tracklets]
        tck_ids = [t.id for t in active_tracklets]

        matches, unmatched_dets = self.HugarianMatch(
            det_bboxes=det_bboxes, tck_bboxes=tck_bboxes
        )
        # 构建匹配成功的tck_id与det_id的pair
        matched_pairs = []
        for i, j in matches:
            matched_pairs.append((i, tck_ids[j]))

        return matched_pairs, unmatched_dets, det_bboxes, det_logits

    def object_addition(
        self,
        unmatched_dets,
        det_bboxes,
        det_logtis,
        frame_id,
        tracked_mask,
        device="cuda",
    ):
        """
        Object Addition部分加入新目标

        inputs:
        - det_bboxes: detector生成的bbox
        - det_logtis: detector生成的logit
        - frame_id: 当前帧序号

        outputs:
        - added: 需要作为new prompts的tck_id序列
        """
        added = []  # 需要输入SAM2的bbox prompts
        # 当前帧所有跟踪框构成的掩码
        if unmatched_dets:
            for (
                det_idx
            ) in (
                unmatched_dets
            ):  # 对每一个边界框，只有其中超过0.5的部分在untracking中才可以作为prompt
                x1, y1, x2, y2 = det_bboxes[det_idx]
                # 构建该框的掩码
                box_mask = torch.zeros(
                    (self.H_orig, self.W_orig), dtype=torch.bool, device=device
                )
                box_mask[y1:y2, x1:x2] = True
                if box_mask.sum() == 0:  # det_bbox很小
                    continue
                r = (
                    box_mask & (~tracked_mask)
                ).sum() / box_mask.sum()  # r表示边界框内尚未跟踪的部分占边界框mask的比例

                if r > self.r:  # 边界框超过0.5的部分未跟踪，作为新prompt预备输入
                    unmatched_bbox = det_bboxes[det_idx]  # 未匹配的边界框

                    unmatched_bbox_id = self.next_id  # 给未匹配的边界框增加新的id
                    self.next_id += 1

                    unmatched_bbox_tracklet = Tracklet(
                        track_id=unmatched_bbox_id,
                        logit=det_logtis[det_idx],
                        bbox=unmatched_bbox,
                        frame_id=frame_id,
                    )
                    self.tracklets[unmatched_bbox_id] = unmatched_bbox_tracklet
                    self.obj_id_to_tck_id[unmatched_bbox_id] = unmatched_bbox_id

                    added.append(unmatched_bbox_id)
        return added

    def object_removal(self, frame_id):
        """
        object remove部分，将确定为lost的物体删去

        inputs:
        - frame_id: 当前帧序号

        outputs:
        - removed: 需要移去的物体的idx列表
        """
        removed = []
        for id, tracklet in self.tracklets.items():
            if (
                tracklet.status == "lost"
                and (frame_id - tracklet.last_seen) > self.max_missing
            ):
                removed.append(id)
        return removed

    def quality_reconstruction(self, matched_pairs, det_bboxes):
        """
        Quality Reconstruction部分，查看是否需要更新prompt

        inputs:
        - det_bboxes: detector生成的bbox
        - det_logtis: detector生成的logit

        outputs:
        - reconstructed: 需要更新prompt的物体tck_id
        """
        reconstructed = []  # 需要更新prompt的id
        if matched_pairs:
            for det_id, tck_id in matched_pairs:
                if self.tracklets[tck_id].status == "pending":
                    reconstructed.append(tck_id)
                    self.tracklets[tck_id].bbox = det_bboxes[det_id]

        return reconstructed

    def HugarianMatch(self, det_bboxes, tck_bboxes):
        """
        匈牙利算法匹配detector生成的bbox和tracking所得的bbox，一般来说det_bboxes>tck_bboxes

        inputs:
        - det_bboxes：detector生成的bbox
        - tck_bboxes：SAM2生成的bbox，来自memory bank

        - outputs:
        - matches：已分配的bbox
        - unmatched_dets：未分配的bbox
        """
        matches = []
        unmatched_dets = list(range(len(det_bboxes)))
        if len(tck_bboxes) > 0:
            cost_matrix = np.zeros((len(det_bboxes), len(tck_bboxes)))
            for i, db in enumerate(det_bboxes):
                for j, tb in enumerate(tck_bboxes):
                    cost_matrix[i, j] = 1 - self.compute_iou(db, tb)

            row_ind, col_ind = linear_sum_assignment(
                cost_matrix
            )  # row_ind 和 col_ind 是成本矩阵的最优分配矩阵索引

            for i, j in zip(row_ind, col_ind):
                iou = self.compute_iou(det_bboxes[i], tck_bboxes[j])
                if iou >= self.iou_thresh:
                    matches.append((i, j))
                    if i in unmatched_dets:
                        unmatched_dets.remove(i)

        return matches, unmatched_dets

    def compute_iou(self, det_bbox, tck_bbox):
        """
        Compute IoU between two [x1, y1, x2, y2] boxes
        """
        xA = max(det_bbox[0], tck_bbox[0])
        yA = max(det_bbox[1], tck_bbox[1])
        xB = min(det_bbox[2], tck_bbox[2])
        yB = min(det_bbox[3], tck_bbox[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        det_bboxArea = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
        tck_bboxArea = (tck_bbox[2] - tck_bbox[0]) * (tck_bbox[3] - tck_bbox[1])

        iou = interArea / float(det_bboxArea + tck_bboxArea - interArea + 1e-6)
        return iou
