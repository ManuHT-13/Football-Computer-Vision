"""Estimate how each video frame maps onto the football pitch. A trained HRNet predicts
field-line maps; fitted lines and the centre circle provide landmarks for calculating and
validating a homography. Line-based refinement, camera-motion propagation, smoothing and
scene-cut handling help keep the mapping stable between detections. Write landmark and
homography CSV files, with options to render the projected pitch or save a diagnostic
frame."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline_paths import resolve_output_dir

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

INPUT_H = 540
INPUT_W = 960
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
N_LINE_MAPS = 28

PITCH_W = 105.0
PITCH_H = 68.0
PITCH_MARGIN = 20.0
VISIBLE_SPAN = (5.0, 130.0)


GMC_WIDTH = 640
GMC_MARGIN = 0.05
MIN_POINTS = 10


MAX_REJECTS = 3

LINE_COLOR = (0, 255, 255)

USE_AMP = True

EXTRAP_MARGIN = 0.5

WORLD_LINES = {
    "Side line left": ("x", 0.0),
    "Side line right": ("x", PITCH_W),
    "Side line top": ("y", PITCH_H),
    "Side line bottom": ("y", 0.0),
    "Middle line": ("x", PITCH_W / 2),
    "Big rect. left main": ("x", 16.5),
    "Big rect. left top": ("y", 54.16),
    "Big rect. left bottom": ("y", 13.84),
    "Big rect. right main": ("x", 88.5),
    "Big rect. right top": ("y", 54.16),
    "Big rect. right bottom": ("y", 13.84),
    "Small rect. left main": ("x", 5.5),
    "Small rect. left top": ("y", 43.16),
    "Small rect. left bottom": ("y", 24.84),
    "Small rect. right main": ("x", 99.5),
    "Small rect. right top": ("y", 43.16),
    "Small rect. right bottom": ("y", 24.84),
}

WORLD_LINES_H = {
    name: np.array([1.0, 0.0, -c]) if axis == "x" else np.array([0.0, 1.0, -c])
    for name, (axis, c) in WORLD_LINES.items()
}

INTERSECTION_TO_PITCH_POINTS = {
    0: "L_GOAL_TL_POST",
    1: "L_GOAL_TR_POST",
    2: "L_GOAL_BL_POST",
    3: "L_GOAL_BR_POST",
    4: "L_GOAL_AREA_BR_CORNER",
    5: "L_GOAL_AREA_TR_CORNER",
    6: "L_GOAL_AREA_BL_CORNER",
    7: "L_GOAL_AREA_TL_CORNER",
    8: "L_PENALTY_AREA_BR_CORNER",
    9: "L_PENALTY_AREA_TR_CORNER",
    10: "L_PENALTY_AREA_BL_CORNER",
    11: "L_PENALTY_AREA_TL_CORNER",
    12: "BL_PITCH_CORNER",
    13: "TL_PITCH_CORNER",
    14: "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    15: "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    16: "R_PENALTY_AREA_BL_CORNER",
    17: "R_PENALTY_AREA_TL_CORNER",
    18: "R_PENALTY_AREA_BR_CORNER",
    19: "R_PENALTY_AREA_TR_CORNER",
    20: "R_GOAL_AREA_BL_CORNER",
    21: "R_GOAL_AREA_TL_CORNER",
    22: "R_GOAL_AREA_BR_CORNER",
    23: "R_GOAL_AREA_TR_CORNER",
    24: "R_GOAL_TL_POST",
    25: "R_GOAL_TR_POST",
    26: "R_GOAL_BL_POST",
    27: "R_GOAL_BR_POST",
    28: "BR_PITCH_CORNER",
    29: "TR_PITCH_CORNER",
    30: "CENTER_CIRCLE_TANGENT_TR",
    31: "CENTER_CIRCLE_TANGENT_TL",
    32: "CENTER_CIRCLE_TANGENT_BR",
    33: "CENTER_CIRCLE_TANGENT_BL",
    34: "CENTER_CIRCLE_TR",
    35: "CENTER_CIRCLE_TL",
    36: "CENTER_CIRCLE_BR",
    37: "CENTER_CIRCLE_BL",
    38: "CENTER_CIRCLE_R",
    39: "CENTER_CIRCLE_L",
    40: "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    41: "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    42: "CENTER_MARK",
    43: "LEFT_CIRCLE_R",
    44: "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    45: "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    46: "LEFT_CIRCLE_TANGENT_T",
    47: "LEFT_CIRCLE_TANGENT_B",
    48: "L_PENALTY_MARK",
    49: "L_MIDDLE_PENALTY",
    50: "RIGHT_CIRCLE_L",
    51: "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    52: "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    53: "RIGHT_CIRCLE_TANGENT_T",
    54: "RIGHT_CIRCLE_TANGENT_B",
    55: "R_PENALTY_MARK",
    56: "R_MIDDLE_PENALTY"
}

NOT_ON_PLANE = {0, 1, 24, 25}

LINE_INTERSECTIONS = [
    ("Side line left", "Side line top", 13),
    ("Side line left", "Side line bottom", 12),
    ("Side line right", "Side line top", 29),
    ("Side line right", "Side line bottom", 28),
    ("Middle line", "Side line top", 15),
    ("Middle line", "Side line bottom", 14),
    ("Big rect. left bottom", "Big rect. left main", 8),
    ("Big rect. left top", "Big rect. left main", 9),
    ("Big rect. left bottom", "Side line left", 10),
    ("Big rect. left top", "Side line left", 11),
    ("Big rect. right bottom", "Big rect. right main", 16),
    ("Big rect. right top", "Big rect. right main", 17),
    ("Big rect. right bottom", "Side line right", 18),
    ("Big rect. right top", "Side line right", 19),
    ("Small rect. left bottom", "Small rect. left main", 4),
    ("Small rect. left top", "Small rect. left main", 5),
    ("Small rect. left bottom", "Side line left", 6),
    ("Small rect. left top", "Side line left", 7),
    ("Small rect. right bottom", "Small rect. right main", 20),
    ("Small rect. right top", "Small rect. right main", 21),
    ("Small rect. right bottom", "Side line right", 22),
    ("Small rect. right top", "Side line right", 23)
]

GROUND_TRUTH_POINTS = {
    "CENTER_MARK": (52.5, 34.0, 0.0),
    "TL_PITCH_CORNER": (0.0, 68.0, 0.0),
    "BL_PITCH_CORNER": (0.0, 0.0, 0.0),
    "TR_PITCH_CORNER": (105.0, 68.0, 0.0),
    "BR_PITCH_CORNER": (105.0, 0.0, 0.0),
    "L_PENALTY_MARK": (11.0, 34.0, 0.0),
    "R_PENALTY_MARK": (94.0, 34.0, 0.0),
    "L_PENALTY_AREA_TL_CORNER": (0.0, 54.16, 0.0),
    "L_PENALTY_AREA_TR_CORNER": (16.5, 54.16, 0.0),
    "L_PENALTY_AREA_BL_CORNER": (0.0, 13.84, 0.0),
    "L_PENALTY_AREA_BR_CORNER": (16.5, 13.84, 0.0),
    "R_PENALTY_AREA_TL_CORNER": (88.5, 54.16, 0.0),
    "R_PENALTY_AREA_TR_CORNER": (105.0, 54.16, 0.0),
    "R_PENALTY_AREA_BL_CORNER": (88.5, 13.84, 0.0),
    "R_PENALTY_AREA_BR_CORNER": (105.0, 13.84, 0.0),
    "L_GOAL_AREA_TL_CORNER": (0.0, 43.16, 0.0),
    "L_GOAL_AREA_TR_CORNER": (5.5, 43.16, 0.0),
    "L_GOAL_AREA_BL_CORNER": (0.0, 24.84, 0.0),
    "L_GOAL_AREA_BR_CORNER": (5.5, 24.84, 0.0),
    "R_GOAL_AREA_TL_CORNER": (99.5, 43.16, 0.0),
    "R_GOAL_AREA_TR_CORNER": (105.0, 43.16, 0.0),
    "R_GOAL_AREA_BL_CORNER": (99.5, 24.84, 0.0),
    "R_GOAL_AREA_BR_CORNER": (105.0, 24.84, 0.0),
    "L_GOAL_TL_POST": (0.0, 30.34, -2.44),
    "L_GOAL_TR_POST": (0.0, 37.66, -2.44),
    "L_GOAL_BL_POST": (0.0, 30.34, 0.0),
    "L_GOAL_BR_POST": (0.0, 37.66, 0.0),
    "R_GOAL_TL_POST": (105.0, 37.66, -2.44),
    "R_GOAL_TR_POST": (105.0, 30.34, -2.44),
    "R_GOAL_BL_POST": (105.0, 37.66, 0.0),
    "R_GOAL_BR_POST": (105.0, 30.34, 0.0),
    "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION": (52.5, 68.0, 0.0),
    "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION": (52.5, 0.0, 0.0),
    "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION": (52.5, 43.15, 0.0),
    "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION": (52.5, 24.85, 0.0),
    "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION": (16.5, 41.31248931623151, 0.0),
    "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION": (16.5, 26.687510683768487, 0.0),
    "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION": (88.5, 41.31248931623151, 0.0),
    "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION": (88.5, 26.687510683768487, 0.0),
    "CENTER_CIRCLE_TANGENT_TR": (61.31243189346428, 36.462426470588234, 0.0),
    "CENTER_CIRCLE_TANGENT_TL": (43.68756810653572, 36.46242647058824, 0.0),
    "CENTER_CIRCLE_TANGENT_BR": (61.31243189346428, 31.537573529411766, 0.0),
    "CENTER_CIRCLE_TANGENT_BL": (43.68756810653572, 31.53757352941176, 0.0),
    "CENTER_CIRCLE_TR": (58.97002704785691, 40.47002704785691, 0.0),
    "CENTER_CIRCLE_TL": (46.02997295214309, 40.47002704785691, 0.0),
    "CENTER_CIRCLE_BR": (58.97002704785691, 27.52997295214309, 0.0),
    "CENTER_CIRCLE_BL": (46.02997295214309, 27.52997295214309, 0.0),
    "CENTER_CIRCLE_R": (61.65, 34.0, 0.0),
    "CENTER_CIRCLE_L": (43.35, 34.0, 0.0),
    "LEFT_CIRCLE_R": (20.15, 34.0, 0.0),
    "RIGHT_CIRCLE_L": (84.85, 34.0, 0.0),
    "LEFT_CIRCLE_TANGENT_T": (19.9906727467215, 35.70008928040832, 0.0),
    "LEFT_CIRCLE_TANGENT_B": (19.9906727467215, 32.29991071959168, 0.0),
    "L_MIDDLE_PENALTY": (16.5, 34.0, 0.0),
    "RIGHT_CIRCLE_TANGENT_T": (85.0093272532785, 35.70008928040832, 0.0),
    "RIGHT_CIRCLE_TANGENT_B": (85.0093272532785, 32.29991071959168, 0.0),
    "R_MIDDLE_PENALTY": (88.5, 34.0, 0.0)
}


BN_MOMENTUM = 0.1
STAGES = [(1, [32, 64]), (4, [32, 64, 128]), (3, [32, 64, 128, 256])]


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride, 1, bias=False)


def bn(c, momentum=BN_MOMENTUM):
    return nn.BatchNorm2d(c, momentum=momentum)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = conv3x3(cin, cout)
        self.bn1 = bn(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = bn(cout)
        self.downsample = None

    def forward(self, x):
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return self.relu(out + x)


class Bottleneck(nn.Module):
    def __init__(self, cin, planes, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, planes, 1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = bn(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class HighResolutionModule(nn.Module):
    def __init__(self, channels, multi_scale_output=True):
        super().__init__()
        self.num_branches = len(channels)
        self.branches = nn.ModuleList(
            [nn.Sequential(*[BasicBlock(c, c) for _ in range(4)]) for c in channels]
        )
        fuse = []
        for i in range(self.num_branches if multi_scale_output else 1):
            row = []
            for j in range(self.num_branches):
                if j > i:
                    row.append(nn.Sequential(nn.Conv2d(channels[j], channels[i], 1, 1, 0, bias=False), bn(channels[i])))
                elif j == i:
                    row.append(None)
                else:
                    convs = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        cout = channels[i] if last else channels[j]
                        layers = [nn.Conv2d(channels[j], cout, 3, 2, 1, bias=False), bn(cout)]
                        if not last:
                            layers.append(nn.ReLU(inplace=True))
                        convs.append(nn.Sequential(*layers))
                    row.append(nn.Sequential(*convs))
            fuse.append(nn.ModuleList(row))
        self.fuse_layers = nn.ModuleList(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = [b(xi) for b, xi in zip(self.branches, x)]
        out = []
        for i, row in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else row[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(row[j](x[j]), size=x[i].shape[-2:], mode="nearest")
                else:
                    y = y + row[j](x[j])
            out.append(self.relu(y))
        return out


def make_transition(pre, cur):
    layers = []
    for i, c in enumerate(cur):
        if i < len(pre):
            if c != pre[i]:
                layers.append(nn.Sequential(nn.Conv2d(pre[i], c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)))
            else:
                layers.append(None)
        else:
            convs = []
            for j in range(i + 1 - len(pre)):
                cin = pre[-1]
                cout = c if j == i - len(pre) else cin
                convs.append(nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)))
            layers.append(nn.Sequential(*convs))
    return nn.ModuleList(layers)


class HRNetW32Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = bn(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = bn(64)
        self.relu = nn.ReLU(inplace=True)
        down = nn.Sequential(nn.Conv2d(64, 256, 1, bias=False), bn(256))
        self.layer1 = nn.Sequential(Bottleneck(64, 64, down), *[Bottleneck(256, 64) for _ in range(3)])

        pre = [256]
        for n, (n_modules, channels) in enumerate(STAGES, start=2):
            setattr(self, f"transition{n - 1}", make_transition(pre, channels))
            last = n == len(STAGES) + 1
            setattr(self, f"stage{n}", nn.Sequential(
                *[HighResolutionModule(channels, multi_scale_output=not (last and m == n_modules - 1)) for m in range(n_modules)]
            ))
            pre = channels
        self.out_channels = pre[0]

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        ys = [self.layer1(x)]
        for n, (_, channels) in enumerate(STAGES, start=2):
            trans = getattr(self, f"transition{n - 1}")
            xs = []
            for i in range(len(channels)):
                src = ys[i] if i < len(ys) else ys[-1]
                xs.append(trans[i](src) if trans[i] is not None else src)
            ys = getattr(self, f"stage{n}")(xs)
        return ys[0]


class LineMapModel(HRNetW32Backbone):
    def __init__(self, n_line_maps=N_LINE_MAPS):
        super().__init__()
        self.final_layer = nn.Conv2d(self.out_channels, n_line_maps, 1, stride=1, padding=0)

    def forward(self, x):
        return self.final_layer(super().forward(x))


def open_cap(sample_dir):
    cap = cv2.VideoCapture(sample_dir)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open {sample_dir}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, w, h, frame_count


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_line_model(weights_path, device):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names")
    if checkpoint.get("num_classes") != N_LINE_MAPS or not class_names or len(class_names) != N_LINE_MAPS:
        raise ValueError("The line-map checkpoint must contain 28 class_names")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is missing its state_dict")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = LineMapModel(N_LINE_MAPS)
    bad = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        print(f"WARNING: {len(bad)} tensors with NaN/inf in the checkpoint, e.g. {bad[:3]}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    class_names = [name.strip() for name in class_names]
    missing = {n for a, b, _ in LINE_INTERSECTIONS for n in (a, b)} - set(class_names)
    if missing:
        print(f"WARNING: classes used for landmarks but missing in the checkpoint: {sorted(missing)}")
    print(f"Loaded HRNet-W32 line-map checkpoint with {len(class_names)} channels "
          f"(epoch {checkpoint.get('epoch')}, val Dice {checkpoint.get('best_val_macro_dice')})")
    return model, class_names


def fit_image_line(line_map, threshold, image_width, image_height):
    ys, xs = np.where(line_map >= threshold)
    if len(xs) < 12:
        return None
    points = np.column_stack((
        xs * (image_width - 1) / max(line_map.shape[1] - 1, 1),
        ys * (image_height - 1) / max(line_map.shape[0] - 1, 1),
    )).astype(np.float32)
    if len(points) > 12000:
        points = points[::int(np.ceil(len(points) / 12000))]
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_WELSCH, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    return np.array([x0, y0], dtype=float), direction / norm


def fit_image_ellipse(line_map, threshold, image_width, image_height):
    ys, xs = np.where(line_map >= threshold)
    if len(xs) < 20:
        return None
    points = np.column_stack((
        xs * (image_width - 1) / max(line_map.shape[1] - 1, 1),
        ys * (image_height - 1) / max(line_map.shape[0] - 1, 1),
    )).astype(np.float32)
    if len(points) > 12000:
        points = points[::int(np.ceil(len(points) / 12000))]
    ellipse = cv2.fitEllipse(points.reshape(-1, 1, 2))
    center, axes, angle = ellipse
    if not np.isfinite((*center, *axes, angle)).all() or min(axes) < 2.0:
        return None
    return ellipse


def intersect_image_line_ellipse(line, ellipse, width, height, margin=0.0):
    point, direction = line
    center, axes, angle = ellipse
    angle = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    def rotate(vector):
        return np.array([
            cos_a * vector[0] + sin_a * vector[1],
            -sin_a * vector[0] + cos_a * vector[1],
        ])

    local_point = rotate(point - np.asarray(center, dtype=float))
    local_direction = rotate(direction)
    radius_x, radius_y = np.asarray(axes, dtype=float) / 2.0
    a = (local_direction[0] / radius_x) ** 2 + (local_direction[1] / radius_y) ** 2
    b = 2.0 * (
        local_point[0] * local_direction[0] / radius_x ** 2
        + local_point[1] * local_direction[1] / radius_y ** 2
    )
    c = (local_point[0] / radius_x) ** 2 + (local_point[1] / radius_y) ** 2 - 1.0
    discriminant = b * b - 4.0 * a * c
    if a <= 1e-12 or discriminant <= 1e-9:
        return []

    roots = ((-b - np.sqrt(discriminant)) / (2.0 * a),
             (-b + np.sqrt(discriminant)) / (2.0 * a))
    intersections = []
    for root in roots:
        candidate = point + root * direction
        if (-margin * width <= candidate[0] < (1 + margin) * width
                and -margin * height <= candidate[1] < (1 + margin) * height):
            intersections.append(candidate)
    return intersections


def supported_map_point(line_map, point, threshold, image_width, image_height, radius=4):
    x = int(round(point[0] * (line_map.shape[1] - 1) / max(image_width - 1, 1)))
    y = int(round(point[1] * (line_map.shape[0] - 1) / max(image_height - 1, 1)))
    x1, x2 = max(0, x - radius), min(line_map.shape[1], x + radius + 1)
    y1, y2 = max(0, y - radius), min(line_map.shape[0], y + radius + 1)
    return y1 < y2 and x1 < x2 and float(line_map[y1:y2, x1:x2].max()) >= threshold


def center_circle_landmarks(line_map, lines, scores, width, height, threshold, margin):
    middle_name = "Middle line"
    if middle_name not in lines:
        return []
    ellipse = fit_image_ellipse(line_map, threshold, width, height)
    if ellipse is None:
        return []

    middle_line = lines[middle_name]
    points = intersect_image_line_ellipse(
        middle_line, ellipse, width, height, margin
    )
    points = [
        point for point in points
        if supported_map_point(line_map, point, threshold * 0.5, width, height)
    ]
    if len(points) != 2:
        return []

    top_line = lines.get("Side line top")
    bottom_line = lines.get("Side line bottom")
    if top_line is None and bottom_line is None:
        return []
    anchor_line = top_line if top_line is not None else bottom_line
    anchor = intersect_image_lines(middle_line, anchor_line, width, height, margin)
    if anchor is None:
        return []

    origin, direction = middle_line
    parameters = [float(np.dot(point - origin, direction)) for point in points]
    anchor_parameter = float(np.dot(anchor - origin, direction))
    near_anchor = int(np.argmin([abs(value - anchor_parameter) for value in parameters]))
    upper_index = near_anchor if top_line is not None else 1 - near_anchor
    ordered = {
        40: points[upper_index],
        41: points[1 - upper_index],
    }
    world_y = {40: 43.15, 41: 24.85}
    line_score = scores[middle_name]
    circle_score = float(line_map.max())
    point_score = float(np.sqrt(line_score * circle_score))
    result = []

    for point_id in (40, 41):
        point = ordered[point_id]
        wx, wy, _ = GROUND_TRUTH_POINTS[INTERSECTION_TO_PITCH_POINTS[point_id]]
        result.append((
            point_id, int(round(point[0])), int(round(point[1])),
            point_score, wx, wy, INTERSECTION_TO_PITCH_POINTS[point_id],
        ))

    samples = [
        (float(np.dot(anchor - origin, direction)), 68.0 if top_line is not None else 0.0),
        (parameters[upper_index], world_y[40]),
        (parameters[1 - upper_index], world_y[41]),
    ]
    t_values = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    y_values = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    t_center = float(t_values.mean())
    t_scale = float(np.max(np.abs(t_values - t_center)))
    if t_scale > 1e-9:
        normalized_t = (t_values - t_center) / t_scale
        matrix = np.column_stack((
            normalized_t,
            np.ones(3, dtype=np.float64),
            -y_values * normalized_t,
        ))
        try:
            a, b, c = np.linalg.solve(matrix, y_values)
            target_y = GROUND_TRUTH_POINTS["CENTER_MARK"][1]
            denominator = target_y * c - a
            if abs(denominator) > 1e-9:
                center_t = t_center + t_scale * (b - target_y) / denominator
                lower, upper = sorted(parameters)
                if lower < center_t < upper:
                    center_point = origin + center_t * direction
                    if (-margin * width <= center_point[0] < (1 + margin) * width
                            and -margin * height <= center_point[1] < (1 + margin) * height):
                        wx, wy, _ = GROUND_TRUTH_POINTS["CENTER_MARK"]
                        result.append((
                            42, int(round(center_point[0])), int(round(center_point[1])),
                            point_score, wx, wy, "CENTER_MARK",
                        ))
        except np.linalg.LinAlgError:
            return result
    return result


def intersect_image_lines(line_a, line_b, width, height, margin=0.0):
    point_a, direction_a = line_a
    point_b, direction_b = line_b
    denominator = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(denominator) < 0.05:
        return None
    delta = point_b - point_a
    distance = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / denominator
    point = point_a + distance * direction_a
    if not (-margin * width <= point[0] < (1 + margin) * width
            and -margin * height <= point[1] < (1 + margin) * height):
        return None
    return point


@torch.no_grad()
def predict_maps(model, frame, device):
    """Returns sigmoid line maps, shape (28, INPUT_H // 4, INPUT_W // 4)."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    image = (image.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)

    pad_h, pad_w = (-INPUT_H) % 32, (-INPUT_W) % 32
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))

    def run(use_amp):
        ctx = torch.autocast("cuda", dtype=torch.float16) if (use_amp and device == "cuda") else nullcontext()
        with ctx:
            out = model(tensor)
        return out[:, :, :INPUT_H // 4, :INPUT_W // 4].float()

    logits = run(USE_AMP)
    if not torch.isfinite(logits).all() and USE_AMP and device == "cuda":
        if not predict_maps.warned:
            print("WARNING: non-finite output with fp16, retrying in fp32 (use --fp32 to skip fp16)")
            predict_maps.warned = True
        logits = run(False)
    if not torch.isfinite(logits).all():
        raise RuntimeError("The model outputs NaN/inf even in fp32: corrupted weights or a different architecture")
    return torch.sigmoid(logits[0]).cpu().numpy()


predict_maps.warned = False


_TABLE_ID = {frozenset((a, b)): i for a, b, i in LINE_INTERSECTIONS}


def sample_line_pixels(line_map, threshold, width, height, n=60):
    """Up to n pixels of a predicted line map above the threshold, in image coordinates."""
    ys, xs = np.where(line_map >= threshold)
    if len(xs) > n:
        sel = np.linspace(0, len(xs) - 1, n).astype(int)
        xs, ys = xs[sel], ys[sel]
    return np.column_stack((
        xs * (width - 1) / max(line_map.shape[1] - 1, 1),
        ys * (height - 1) / max(line_map.shape[0] - 1, 1),
    )).astype(np.float64)


def landmarks_from_lines(lines, scores, width, height, margin=EXTRAP_MARGIN):
    """Intersects every detected vertical pitch line with every horizontal one.

    Intersections of the infinite lines are valid homography correspondences even when the real
    corner is outside the frame. Each landmark is (id, x, y, score, world_x, world_y, label), with
    id = -1 for points that are not in the 57-point table.
    """
    names = [n for n in lines if n in WORLD_LINES]
    out = []
    for a in names:
        for b in names:
            if WORLD_LINES[a][0] != "x" or WORLD_LINES[b][0] != "y":
                continue
            point = intersect_image_lines(lines[a], lines[b], width, height, margin)
            if point is None:
                continue
            lid = _TABLE_ID.get(frozenset((a, b)), -1)
            label = INTERSECTION_TO_PITCH_POINTS[lid] if lid >= 0 else f"{a} | {b}"
            score = float(np.sqrt(scores[a] * scores[b]))
            out.append((lid, int(round(point[0])), int(round(point[1])), score,
                        WORLD_LINES[a][1], WORLD_LINES[b][1], label))
    return out


def detect_line_landmarks(model, frame, device, class_names, threshold, margin=EXTRAP_MARGIN):
    """Returns (landmarks, line_pts): line intersections and pixel samples of every detected pitch line."""
    height, width = frame.shape[:2]
    maps = predict_maps(model, frame, device)

    lines = {}
    scores = {}
    line_pts = {}
    for channel, name in enumerate(class_names):
        if name == "Circle central":
            continue
        line = fit_image_line(maps[channel], threshold, width, height)
        if line is None:
            continue
        lines[name] = line
        scores[name] = float(maps[channel].max())
        if name in WORLD_LINES:
            line_pts[name] = sample_line_pixels(maps[channel], threshold, width, height)

    landmarks = landmarks_from_lines(lines, scores, width, height, margin)
    if "Circle central" in class_names:
        circle_channel = class_names.index("Circle central")
        landmarks.extend(center_circle_landmarks(
            maps[circle_channel], lines, scores, width, height, threshold, margin
        ))
    detect_line_landmarks.diag = {"lines": len(lines), "peak": float(maps.max())}
    return sorted(landmarks, key=lambda item: (item[0] < 0, item[0])), line_pts


detect_line_landmarks.diag = {"lines": 0, "peak": 0.0}


def debug_frame(model, cap, idx, device, class_names, threshold, out_dir):
    """Prints per-class stats for one frame and saves an overlay of the predicted line maps."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx}")
    maps = predict_maps(model, frame, device)
    height, width = frame.shape[:2]
    print(f"\nFrame {idx} ({width}x{height}) | threshold {threshold}")
    print(f"{'class':28s} {'max prob':>8s} {'px>=thr':>8s} {'shape fitted':>12s}")
    for c, name in enumerate(class_names):
        n_px = int((maps[c] >= threshold).sum())
        if name == "Circle central":
            fitted = fit_image_ellipse(maps[c], threshold, width, height) is not None
        else:
            fitted = fit_image_line(maps[c], threshold, width, height) is not None
        print(f"{name:28s} {maps[c].max():8.3f} {n_px:8d} {str(fitted):>12s}")
    landmarks, line_pts = detect_line_landmarks(model, frame, device, class_names, threshold)
    print(f"Landmarks found: {len(landmarks)} ({sum(1 for k in landmarks if k[0] >= 0)} from the 57-point table) "
          f"| pitch lines with pixels: {len(line_pts)}")

    heat = (maps.max(axis=0) * 255).clip(0, 255).astype(np.uint8)
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(frame, 0.55, cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.45, 0)
    cv2.imwrite(str(out_dir / f"debug_frame{idx}.png"), np.hstack([frame, overlay]))
    print(f"Saved {out_dir / f'debug_frame{idx}.png'} (left: frame, right: predicted lines)")


def to_pitch(H, pts):
    """Map image coordinates to pitch coordinates (meters)."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    den = np.where(np.abs(p[:, 2:3]) < 1e-9, 1e-9, p[:, 2:3])
    return p[:, :2] / den


def degenerate(pts):
    s = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
    return s[0] < 1e-6 or s[1] / s[0] < 0.05


def plausible(H, w, h):
    p = H @ np.array([w / 2, h / 2, 1.0])
    if abs(p[2]) < 1e-9:
        return False
    x, y = p[0] / p[2], p[1] / p[2]
    if not (-PITCH_MARGIN <= x <= PITCH_W + PITCH_MARGIN and -PITCH_MARGIN <= y <= PITCH_H + PITCH_MARGIN):
        return False
    a = to_pitch(H, np.array([[0.05 * w, 0.9 * h], [0.95 * w, 0.9 * h]]))
    span = float(np.hypot(*(a[1] - a[0])))
    return np.isfinite(span) and VISIBLE_SPAN[0] <= span <= VISIBLE_SPAN[1]


def grid_points(w, h):
    gx, gy = np.meshgrid(np.linspace(0.05 * w, 0.95 * w, 5), np.linspace(0.3 * h, 0.95 * h, 4))
    return np.c_[gx.ravel(), gy.ravel()]


def pitch_shift(Ha, Hb, w, h):
    """Mean distance (m) between two homographies projected onto the same image grid."""
    g = grid_points(w, h)
    return float(np.linalg.norm(to_pitch(Ha, g) - to_pitch(Hb, g), axis=1).mean())


def blend_h(H_old, H_new, w, h, alpha):
    """Blend two homographies by averaging their projections on a grid and refitting."""
    g = grid_points(w, h)
    p = alpha * to_pitch(H_new, g) + (1 - alpha) * to_pitch(H_old, g)
    Hb, _ = cv2.findHomography(g.astype(np.float32), p.astype(np.float32), 0)
    if Hb is None or not np.isfinite(Hb).all() or abs(Hb[2, 2]) < 1e-12:
        return H_new
    return Hb / Hb[2, 2]


def fit_homography(kps, w, h, ransac_px, min_inliers, max_reproj):
    inl = np.zeros(len(kps), bool)
    used, img, world = [], [], []
    for k, (_i, px, py, _s, wx, wy, _label) in enumerate(kps):
        used.append(k)
        img.append([px, py])
        world.append([wx, wy])

    if len(used) < max(4, min_inliers):
        return None, inl, np.nan

    img = np.array(img, dtype=np.float32)
    world = np.array(world, dtype=np.float32)
    Hpi, mask = cv2.findHomography(world, img, cv2.RANSAC, ransac_px)
    if Hpi is None or mask is None:
        return None, inl, np.nan

    mask = mask.ravel().astype(bool)
    if mask.sum() < min_inliers or degenerate(world[mask]) or degenerate(img[mask]):
        return None, inl, np.nan

    proj = cv2.perspectiveTransform(world[mask][None], Hpi)[0]
    err = float(np.linalg.norm(proj - img[mask], axis=1).mean())
    if err > max_reproj:
        return None, inl, err

    try:
        H = np.linalg.inv(Hpi)
    except np.linalg.LinAlgError:
        return None, inl, err
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None, inl, err

    H = H / H[2, 2]
    if not plausible(H, w, h):
        return None, inl, err

    inl[np.array(used)[mask]] = True
    return H, inl, err

def _norm_matrix(w, h):
    """Pixels -> normalized coordinates (roughly [-1, 1]) used to keep the optimisation well conditioned."""
    return np.array([[2.0 / w, 0.0, -1.0], [0.0, 2.0 / w, -h / w], [0.0, 0.0, 1.0]])


def line_residuals(G, pts_n, scale):
    """Signed distances (px) from detected line pixels to the pitch lines projected by G.

    G maps normalized image coordinates to pitch meters, so the image line of a pitch line l is G^T l.
    """
    out = []
    for name, p in pts_n.items():
        li = G.T @ WORLD_LINES_H[name]
        n = np.hypot(li[0], li[1])
        if n < 1e-12:
            out.append(np.full(len(p), 1e3))
            continue
        out.append((p @ li[:2] + li[2]) / n * scale)
    return np.concatenate(out)


def refine_homography(H0, line_pts, w, h, ridge=100.0):
    """Refines H (image -> pitch) so the projected pitch lines fall on the detected line pixels.

    Needs at least 3 detected lines, with both a vertical and a horizontal pitch line.
    Returns (H, mean_abs_residual_px), or (None, nan) when it fails or does not improve the fit.
    """
    if least_squares is None:
        return None, np.nan
    names = [n for n, p in line_pts.items() if n in WORLD_LINES and len(p) >= 12]
    if len(names) < 3 or {WORLD_LINES[n][0] for n in names} != {"x", "y"}:
        return None, np.nan

    scale = w / 2.0
    N = _norm_matrix(w, h)
    base = H0 @ np.linalg.inv(N)
    pts_n = {n: (np.asarray(line_pts[n], float) - [w / 2.0, h / 2.0]) / scale for n in names}

    def gmat(p):
        return base @ (np.eye(3) + np.append(p, 0.0).reshape(3, 3))

    def fun(p):
        return np.concatenate([line_residuals(gmat(p), pts_n, scale), ridge * p])

    before = float(np.abs(line_residuals(base, pts_n, scale)).mean())
    try:
        sol = least_squares(fun, np.zeros(8), loss="soft_l1", f_scale=3.0, max_nfev=60)
    except Exception:
        return None, np.nan
    G = gmat(sol.x)
    H = G @ N
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None, np.nan
    H = H / H[2, 2]
    after = float(np.abs(line_residuals(G, pts_n, scale)).mean())
    if not after < before:
        return None, np.nan
    return H, after


def estimate_motion(prev_gray, gray, prev_boxes):
    hh, ww = prev_gray.shape
    mask = np.zeros((hh, ww), np.uint8)
    my, mx = int(hh * GMC_MARGIN), int(ww * GMC_MARGIN)
    mask[my:hh - my, mx:ww - mx] = 255
    for x1, y1, x2, y2 in prev_boxes.astype(int):
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0

    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=3, mask=mask)
    if pts is None or len(pts) < MIN_POINTS:
        return None, None

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < MIN_POINTS:
        return None, 0.0

    M, inliers = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None, 0.0
    return M, float(inliers.sum()) / len(pts)


def to_full(M, scale):
    full = np.eye(3)
    full[:2, :2] = M[:, :2]
    full[:2, 2] = M[:, 2] / scale
    return full


def pitch_polylines():
    lines = [
        np.array([[0, 0], [PITCH_W, 0], [PITCH_W, PITCH_H], [0, PITCH_H], [0, 0]], dtype=float),
        np.array([[PITCH_W / 2, 0], [PITCH_W / 2, PITCH_H]], dtype=float),
    ]
    t = np.linspace(0, 2 * np.pi, 73)
    lines.append(np.stack([PITCH_W / 2 + 9.15 * np.cos(t), PITCH_H / 2 + 9.15 * np.sin(t)], axis=1))
    for x0, sgn in ((0.0, 1.0), (PITCH_W, -1.0)):
        lines.append(np.array([[x0, 13.84], [x0 + sgn * 16.5, 13.84], [x0 + sgn * 16.5, 54.16], [x0, 54.16]], dtype=float))
        lines.append(np.array([[x0, 24.84], [x0 + sgn * 5.5, 24.84], [x0 + sgn * 5.5, 43.16], [x0, 43.16]], dtype=float))
    a = np.arccos(5.5 / 9.15)
    t = np.linspace(-a, a, 25)
    lines.append(np.stack([11 + 9.15 * np.cos(t), 34 + 9.15 * np.sin(t)], axis=1))
    lines.append(np.stack([94 - 9.15 * np.cos(t), 34 + 9.15 * np.sin(t)], axis=1))
    return lines


def draw_pitch(frame, H, lines):
    h, w = frame.shape[:2]
    Hp = np.linalg.inv(H)
    c = H @ np.array([w / 2, h / 2, 1.0])
    c = c[:2] / c[2]
    s0 = np.sign((Hp @ np.array([c[0], c[1], 1.0]))[2])
    for pts in lines:
        P = np.c_[pts, np.ones(len(pts))] @ Hp.T
        front = P[:, 2] * s0 > 1e-6
        den = np.where(np.abs(P[:, 2:3]) < 1e-9, 1e-9, P[:, 2:3])
        xy = P[:, :2] / den
        ok = front & (np.abs(xy[:, 0]) < 5 * w) & (np.abs(xy[:, 1]) < 5 * h)
        for k in range(len(xy) - 1):
            if ok[k] and ok[k + 1]:
                cv2.line(frame, (int(xy[k, 0]), int(xy[k, 1])), (int(xy[k + 1, 0]), int(xy[k + 1, 1])), LINE_COLOR, 2)


def draw_detected_field_lines(frame, line_pts):
    """Draw detected image-space pitch lines while a full homography is unavailable."""
    h, w = frame.shape[:2]
    extension = 2.0 * np.hypot(w, h)
    drawn = 0
    for points in line_pts.values():
        if len(points) < 12:
            continue
        vx, vy, x0, y0 = cv2.fitLine(
            np.asarray(points, dtype=np.float32), cv2.DIST_WELSCH, 0, 0.01, 0.01
        ).flatten()
        p1 = (int(round(x0 - vx * extension)), int(round(y0 - vy * extension)))
        p2 = (int(round(x0 + vx * extension)), int(round(y0 + vy * extension)))
        visible, p1, p2 = cv2.clipLine((0, 0, w, h), p1, p2)
        if visible:
            cv2.line(frame, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)
            drawn += 1
    return drawn


def draw_keypoints(frame, kps, inl):
    for k, (i, px, py, *_rest) in enumerate(kps):
        color = (0, 255, 0) if inl[k] else (0, 0, 255)
        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.putText(frame, str(i) if i >= 0 else "x", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main():
    global USE_AMP
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--line-conf", type=float, default=0.3)
    parser.add_argument("--tracks", default=None)
    parser.add_argument("--mask-conf", type=float, default=0.3)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--ransac-px", type=float, default=8.0)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--max-reproj", type=float, default=8.0)
    parser.add_argument("--max-jump", type=float, default=6.0)
    parser.add_argument("--smooth", type=float, default=0.6)
    parser.add_argument("--cut-ratio", type=float, default=0.25)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--out", default="out")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fp32", action="store_true",)
    parser.add_argument("--margin", type=float, default=EXTRAP_MARGIN)
    parser.add_argument("--no-refine", action="store_true", help="Disable the line-based refinement")
    parser.add_argument("--refine-max-resid", type=float, default=8.0)
    parser.add_argument("--debug-frame", type=int, default=None)

    args = parser.parse_args()
    USE_AMP = not args.fp32
    if least_squares is None and not args.no_refine:
        print("scipy not found: line-based refinement disabled (pip install scipy)")

    out_dir = resolve_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, line_class_names = load_line_model(args.weights, device)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    boxes_by_frame = {}
    if args.tracks:
        tdf = pd.read_csv(args.tracks)
        if "conf" in tdf.columns:
            tdf = tdf[tdf["conf"] >= args.mask_conf]
        if "label" in tdf.columns:
            tdf = tdf[tdf["label"] != "ball"]
        boxes_by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float) for f, g in tdf.groupby("frame")}

    cap, fps, w, h, frame_count = open_cap(args.sample)
    if args.debug_frame is not None:
        debug_frame(model, cap, args.debug_frame, device, line_class_names, args.line_conf, out_dir)
        return
    start = max(args.start_frame, 0)
    if frame_count > 0 and start >= frame_count:
        raise RuntimeError(f"--start-frame {start} is out of range ({frame_count} frames)")
    available = frame_count - start
    n_frames = available if args.frames is None else min(args.frames, available)
    end = start + n_frames - 1
    scale = min(1.0, GMC_WIDTH / w)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    writer = None
    if args.render:
        writer = cv2.VideoWriter(str(out_dir / "pitch.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    lines = pitch_polylines()
    H_ref = None
    T = np.eye(3)
    last_detect = -10 ** 9
    prev_gray = None
    prev_boxes = np.zeros((0, 4))
    cuts = []
    kp_rows = []
    h_rows = []
    sources = {"detected": 0, "refined": 0, "propagated": 0, "none": 0}
    n_refined = 0
    n_tracked = 0
    n_det = 0
    n_acc = 0
    n_jump = 0
    n_rej = 0
    kp_counts = []
    n_lines_list = []
    peaks = []
    inl_counts = []
    errs = []
    last_idx = start

    for idx in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        last_idx = idx

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        cut = False
        if prev_gray is not None:
            M, ratio = estimate_motion(prev_gray, gray, prev_boxes * scale)
            if ratio is not None and ratio < args.cut_ratio:
                cut = True
            elif M is not None:
                T = T @ np.linalg.inv(to_full(M, scale))
        prev_gray = gray
        prev_boxes = boxes_by_frame.get(idx, np.zeros((0, 4)))

        if cut:
            cuts.append(idx)
            H_ref = None
            T = np.eye(3)
            n_rej = 0

        kps = []
        line_pts = {}
        inl = np.zeros(0, bool)
        detected = False
        tracked = False
        err_row = float("nan")
        resid_row = float("nan")
        gap = args.stride if H_ref is not None else args.retry
        if cut or idx - last_detect >= gap:
            last_detect = idx
            kps, line_pts = detect_line_landmarks(model, frame, device, line_class_names, args.line_conf, args.margin)
            H_new, inl, err = fit_homography(kps, w, h, args.ransac_px, args.min_inliers, args.max_reproj)
            n_det += 1
            kp_counts.append(len(kps))
            n_lines_list.append(detect_line_landmarks.diag["lines"])
            peaks.append(detect_line_landmarks.diag["peak"])

            if H_new is not None and not args.no_refine:
                H_fit, rr = refine_homography(H_new, line_pts, w, h)
                if H_fit is not None and plausible(H_fit, w, h):
                    H_new, resid_row = H_fit, rr
                    n_refined += 1

            if H_new is not None and H_ref is not None:
                # Keep the new homography consistent with the propagated estimate.
                H_prop = H_ref @ T
                H_prop = H_prop / H_prop[2, 2]
                shift = pitch_shift(H_prop, H_new, w, h)
                if shift > args.max_jump and n_rej < MAX_REJECTS:
                    H_new = None
                    inl = np.zeros(len(kps), bool)
                    n_rej += 1
                    n_jump += 1
                else:
                    n_rej = 0
                    if args.smooth < 1.0:
                        H_new = blend_h(H_prop, H_new, w, h, args.smooth)

            if H_new is not None:
                H_ref = H_new
                T = np.eye(3)
                detected = True
                n_acc += 1
                inl_counts.append(int(inl.sum()))
                errs.append(err)
                err_row = err
            elif not args.no_refine and H_ref is not None:
                H_prop = H_ref @ T
                H_prop = H_prop / H_prop[2, 2]
                H_trk, rr = refine_homography(H_prop, line_pts, w, h)
                if (H_trk is not None and rr <= args.refine_max_resid and plausible(H_trk, w, h)
                        and pitch_shift(H_prop, H_trk, w, h) <= args.max_jump):
                    H_ref = H_trk
                    T = np.eye(3)
                    tracked = True
                    n_tracked += 1
                    resid_row = rr
            for k, (i, px, py, s, _wx, _wy, label) in enumerate(kps):
                kp_rows.append((idx, i, label, px, py, s, bool(inl[k])))

        H = None
        if H_ref is not None:
            H = H_ref @ T
            H = H / H[2, 2]
        source = "detected" if detected else ("refined" if tracked else ("propagated" if H is not None else "none"))
        sources[source] += 1

        flat = H.ravel().tolist() if H is not None else [float("nan")] * 9
        h_rows.append((idx, source, len(kps), int(inl.sum()), err_row, resid_row, *flat))

        if writer is not None:
            if H is not None:
                draw_pitch(frame, H, lines)
            elif line_pts:
                n_lines_drawn = draw_detected_field_lines(frame, line_pts)
                if n_lines_drawn:
                    cv2.putText(
                        frame, "detected lines (no homography)", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_COLOR, 2,
                    )
            if kps:
                draw_keypoints(frame, kps, inl)
            status = f"{idx}  {source}" + ("  scene cut" if cut else "")
            cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    if not h_rows:
        raise RuntimeError("Could not read any frames")

    pd.DataFrame(kp_rows, columns=["frame", "idx", "label", "x", "y", "score", "inlier"]).to_csv(out_dir / "keypoints.csv", index=False)
    pd.DataFrame(
        h_rows,
        columns=["frame", "source", "n_keypoints", "n_inliers", "reproj", "line_resid"] + [f"h{i}" for i in range(9)],
    ).to_csv(out_dir / "homography.csv", index=False)

    total = len(h_rows)
    with_h = sources["detected"] + sources["refined"] + sources["propagated"]
    print(f"Frames: {total} ({start}-{last_idx})")
    print(f"Keypoint detections: {n_det}")
    if n_lines_list:
        print(f"Lines fitted per detection: average {np.mean(n_lines_list):.1f} | peak probability: average {np.mean(peaks):.2f}")
    if kp_counts:
        print(f"Keypoints per detection: average {np.mean(kp_counts):.1f}, max {int(np.max(kp_counts))}")
    print(f"Accepted homographies: {n_acc} of {n_det} (rejected for jumps: {n_jump})")
    if inl_counts:
        print(f"Inliers per homography: average {np.mean(inl_counts):.1f}, min {int(np.min(inl_counts))}")
        print(f"Mean reprojection error: {np.mean(errs):.2f} px")
    print(f"Frames with homography: {with_h} ({100 * with_h / total:.1f}%)")
    print(f"  detected: {sources['detected']}, line-refined: {sources['refined']}, propagated: {sources['propagated']}")
    print(f"Line refinement: {n_refined} keypoint fits refined, {n_tracked} line-only updates")
    print(f"Frames without homography: {sources['none']}")
    print(f"Scene cuts: {len(cuts)} {cuts[:10]}")


if __name__ == "__main__":
    main()