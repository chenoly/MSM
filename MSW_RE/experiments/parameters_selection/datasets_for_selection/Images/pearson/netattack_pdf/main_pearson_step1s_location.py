import os
import cv2
import json
import numpy as np


class ImageAnnotator:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.current_index = 0
        self.points_list = []
        self.current_points = []

        self.main_window_name = "Image Viewer - Click 4 Points (ESC to quit)"
        cv2.namedWindow(self.main_window_name)
        cv2.setMouseCallback(self.main_window_name, self.mouse_callback)

        self.roi_size = 300  # 矩形框大小
        self.mouse_pos = (0, 0)  # 鼠标位置
        self.show_rect = True  # 是否显示矩形框

        self.load_current_image()

    def load_current_image(self):
        if self.current_index >= len(self.image_paths):
            print("所有图像标注完成！")
            return

        self.image_path = self.image_paths[self.current_index]
        print(f"\n正在处理: {os.path.basename(self.image_path)}")

        # 使用 imdecode 支持中文路径
        with open(self.image_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            self.original_image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # 合并图像：前3000行 + 后3000行
        h, w = self.original_image.shape[:2]
        top_part = self.original_image[:3000]
        bottom_part = self.original_image[h - 3000:]
        self.combined_image = np.vstack([top_part, bottom_part])

        # 调整图像以适应屏幕宽度
        self.display_image = self.fit_to_screen(self.combined_image)
        self.current_points = []

        self.show_image_with_rect()  # 显示带红框图像

    def fit_to_screen(self, img):
        screen_width = 1280
        scale = screen_width / img.shape[1]
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        return cv2.resize(img, new_size)

    def show_debug_window(self, roi, point=None, corner=""):
        debug_img = roi.copy()
        if point is not None:
            cv2.circle(debug_img, point, 5, (255, 0, 0), -1)
            cv2.putText(debug_img, f"Corner: {corner}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        debug_img = cv2.resize(debug_img, None, fx=1.5, fy=1.5)
        cv2.imshow("Debug ROI", debug_img)
        cv2.waitKey(1000)  # 显示1秒后关闭
        cv2.destroyWindow("Debug ROI")

    def detect_corner_point(self, x, y, corner_type="top_left"):
        """
        在 (x,y) 周围裁剪 500x500 区域，使用 Otsu + Canny 检测边缘，并查找距离指定角最近的边缘点
        """
        h, w = self.combined_image.shape[:2]
        half_size = self.roi_size // 2
        x1 = max(x - half_size, 0)
        y1 = max(y - half_size, 0)
        x2 = min(x + half_size, w)
        y2 = min(y + half_size, h)

        roi = self.combined_image[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return x, y

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary, 50, 150)

        ys, xs = np.where(edges > 0)
        if len(xs) == 0 or len(ys) == 0:
            return x, y

        h_roi, w_roi = roi.shape[:2]

        if corner_type == "top_left":
            ref_x, ref_y = 0, 0
        elif corner_type == "top_right":
            ref_x, ref_y = w_roi, 0
        elif corner_type == "bottom_right":
            ref_x, ref_y = w_roi, h_roi
        elif corner_type == "bottom_left":
            ref_x, ref_y = 0, h_roi
        else:
            return x, y

        distances = (xs - ref_x) ** 2 + (ys - ref_y) ** 2
        idx = np.argmin(distances)
        edge_x, edge_y = xs[idx], ys[idx]

        self.show_debug_window(roi, (edge_x, edge_y), corner_type)

        return x1 + edge_x, y1 + edge_y

    def draw_mouse_rect(self, img, x, y):
        """在图像上绘制红色矩形框"""
        overlay = img.copy()
        rect_half = self.roi_size // 2
        height, width = img.shape[:2]

        # 计算矩形范围
        left = max(x - rect_half, 0)
        right = min(x + rect_half, width)
        top = max(y - rect_half, 0)
        bottom = min(y + rect_half, height)

        # 绘制红色矩形
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 255), 2)
        alpha = 0.6  # 透明度
        result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return result

    def show_image_with_rect(self):
        display_h, display_w = self.display_image.shape[:2]
        combined_h, combined_w = self.combined_image.shape[:2]

        # 计算缩放比例（基于宽度）
        scale_x = display_w / combined_w
        scale_y = display_h / combined_h

        # 将 roi_size 转换为 display_image 上的大小
        roi_size_display = int(self.roi_size * scale_x)

        # 创建副本并画框
        display_copy = self.display_image.copy()

        # ---- 添加进度条 ----
        bar_height = 30
        progress_bar_top = 0
        progress_bar_bottom = bar_height

        # 白底黑字背景
        cv2.rectangle(display_copy, (0, 0), (display_w, bar_height), (50, 50, 50), -1)

        # 进度信息
        total = len(self.image_paths)
        current = self.current_index + 1  # 当前第几张（从1开始计数）

        # 绘制文字
        text = f"Processing Image: {current} / {total}"
        font_scale = 0.6
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = 10
        text_y = int(bar_height / 2 + text_height / 2)
        cv2.putText(display_copy, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    font_thickness)

        # 绘制进度条
        bar_x_start = text_x + text_width + 20
        bar_x_end = display_w - 20
        bar_length = bar_x_end - bar_x_start
        filled_length = int((current / total) * bar_length)

        # 整体进度条边框
        cv2.rectangle(display_copy, (bar_x_start, bar_height // 4),
                      (bar_x_end, bar_height * 3 // 4), (200, 200, 200), 1)

        # 已完成部分填充
        cv2.rectangle(display_copy, (bar_x_start, bar_height // 4),
                      (bar_x_start + filled_length, bar_height * 3 // 4), (0, 255, 0), -1)

        # ---- 鼠标矩形框绘制 ----
        x, y = self.mouse_pos
        half = roi_size_display // 2

        left = max(x - half, 0)
        right = min(x + half, display_w)
        top = max(y - half, 0)
        bottom = min(y + half, display_h)

        # 绘制红色矩形
        cv2.rectangle(display_copy, (left, top), (right, bottom), (0, 0, 255), 2)

        # 显示图像
        cv2.imshow(self.main_window_name, display_copy)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
            self.show_image_with_rect()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.handle_click(x, y)

    def save_points(self, save_dir):
        serializable_points = [
            [[int(x), int(y)] for x, y in image_points]
            for image_points in self.points_list
        ]

        json_path = os.path.join(save_dir, "all_points.json")
        with open(json_path, "w") as f:
            json.dump(serializable_points, f, indent=4)

        print(f"\n所有标注点已保存到：\n  → {json_path}")

    def handle_click(self, x, y):
        click_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        if len(self.current_points) >= 4:
            return

        corner_type = click_order[len(self.current_points)]

        # 获取拼接图像坐标
        display_h, display_w = self.display_image.shape[:2]
        combined_h, combined_w = self.combined_image.shape[:2]

        real_x_combined = int(x / display_w * combined_w)
        real_y_combined = int(y / display_h * combined_h)

        print(f"拼接图像上的点击位置: ({real_x_combined}, {real_y_combined})")

        # 边缘检测定位点（拼接图坐标）
        edge_x, edge_y = self.detect_corner_point(real_x_combined, real_y_combined, corner_type)

        # 映射到原始图像坐标
        original_h, original_w = self.original_image.shape[:2]
        if edge_y < 3000:
            orig_y = edge_y
        else:
            orig_y = (original_h - 3000) + (edge_y - 3000)
        orig_x = int(edge_x / combined_w * original_w)

        print(f"→ 定位点 (原始图像): ({orig_x}, {orig_y})")
        self.current_points.append((orig_x, orig_y))

        if len(self.current_points) == 4:
            self.points_list.append(self.current_points.copy())
            print(f"已保存第{self.current_index + 1}张图的4个点：{self.current_points}")
            self.current_index += 1
            self.load_current_image()
        else:
            pass


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # 隐藏Tk默认窗口

    print("请选择包含 BMP 图像的文件夹：")
    # folder_selected = filedialog.askdirectory(title="选择图像文件夹")
    folder_selected = "scanned_org"
    if not folder_selected:
        print("未选择文件夹，程序退出。")
        exit()

    image_paths = sorted([
        os.path.join(folder_selected, f)
        for f in os.listdir(folder_selected)
        if f.lower().endswith(".bmp")
    ])

    if not image_paths:
        print("所选文件夹中没有找到 .bmp 图像。")
    else:
        annotator = ImageAnnotator(image_paths)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        annotator.save_points(folder_selected)

target_width = 8976
target_height = 13086

target_points = np.float32([
    [0, 0],
    [target_width - 1, 0],
    [target_width - 1, target_height - 1],
    [0, target_height - 1]
])

json_file = "all_points.json"

save_path = "scanned_located/"


with open(json_file, "rb") as f:
    source_points_list = json.load(f)

image_paths = sorted([
    f for f in os.listdir('.') if f.lower().endswith(".bmp")
])
image_paths.sort()

for idx in range(len(image_paths)):
    image = cv2.imread(image_paths[idx])

    source_points = source_points_list[idx]

    M = cv2.getPerspectiveTransform(np.asarray(source_points, dtype=np.float32), target_points)

    transformed_image = cv2.warpPerspective(image, M, (target_width, target_height))

    cv2.imwrite(os.path.join(save_path, os.path.basename(image_paths[idx])).replace('.bmp', '.png'), transformed_image)

    print(f"已保存: {os.path.join(save_path, os.path.basename(image_paths[idx]))}")

print("所有图片已完成透视变换并保存。")
