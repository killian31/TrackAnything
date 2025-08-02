import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import cv2
from PIL import Image


# 建立一个离散 colormap，用 obj_id 做种子颜色
def get_color_from_id(obj_id, max_colors=20):
    cmap = plt.colormaps.get_cmap("tab20")
    color = cmap(obj_id % max_colors)  # 返回 RGBA (0~1)
    rgb = tuple(int(255 * c) for c in color[:3])
    return rgb


def show_mask(image, mask, obj_id=None, alpha=0.6):
    color = get_color_from_id(obj_id=obj_id)
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for i in range(3):  # R, G, B通道
        colored_mask[..., i] = mask * color[i]
        # 透明叠加
    mask_bool = mask.astype(bool)
    image[mask_bool] = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)[
        mask_bool
    ]
    return image


# 绘制边界框
def show_box(image, box, obj_id):
    """在图像上绘制bbox和id标签"""
    color = get_color_from_id(obj_id)
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[2]), int(box[3])

    # 绘制边框
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=2)

    # 绘制标签
    label = str(obj_id)
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        image, (x0, y0 - text_h - baseline), (x0 + text_w, y0), (255, 255, 255), -1
    )
    cv2.putText(
        image,
        label,
        (x0, y0 - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        thickness=1,
    )
    return image
