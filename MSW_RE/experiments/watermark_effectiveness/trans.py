import os.path

import cv2
import numpy as np

# 目标尺寸
target_width = 8976
target_height = 13086

# 定义目标图片上的四个角点（固定不变）
target_points = np.float32([
    [0, 0],  # 左上角
    [target_width - 1, 0],  # 右上角
    [target_width - 1, target_height - 1],  # 右下角
    [0, target_height - 1]  # 左下角
])

# 每张图片在原图中的四个点（源点）
source_points_list = [
    # 第一张图片
    np.float32([
        [911, 764],
        [9531, 701],
        [9580, 13269],
        [976, 13342]
    ]),
    # 第二张图片
    np.float32([
        [986, 650],
        [9602, 633],
        [9578, 13202],
        [974, 13230]
    ]),
    # 第三张图片
    np.float32([
        [950, 719],
        [9570, 671],
        [9592, 13239],
        [986, 13298]
    ]),
    # 第四张图片
    np.float32([
        [966, 732],
        [9580, 678],
        [9618, 13243],
        [1012, 13310]
    ]),
    # 第五张图片
    np.float32([
        [944, 691],
        [9561, 736],
        [9450, 13303],
        [845, 13270]
    ]),
    # 第六张图片
    np.float32([
        [975, 685],
        [9592, 624],
        [9637, 13191],
        [1037, 13264]
    ]),
    # 第七张图片
    np.float32([
        [1104, 670],
        [9718, 677],
        [9662, 13244],
        [1060, 13248]
    ])
]

# 图片路径列表，请替换为你自己的实际路径
image_paths = [
    "img946.bmp",
    "img947.bmp",
    "img948.bmp",
    "img949.bmp",
    "img950.bmp",
    "img951.bmp",
    "img952.bmp"
]

output_prefix = "transformed_image_"

for idx in range(len(image_paths)):
    image = cv2.imread(os.path.join("captured_images", image_paths[idx]))

    source_points = source_points_list[idx]

    M = cv2.getPerspectiveTransform(source_points, target_points)

    transformed_image = cv2.warpPerspective(image, M, (target_width, target_height))

    output_path = f"{output_prefix}{idx}.png"
    cv2.imwrite(output_path, transformed_image)

    print(f"已保存: {output_path}")

print("所有图片已完成透视变换并保存。")