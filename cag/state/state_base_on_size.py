import os
import cv2
import numpy as np
from collections import defaultdict
import argparse
from ultralytics import YOLO

# 分组及其宽高范围（无死角，优先级从上到下）
GROUPS = [
    ('极小', '<20x20', lambda w, h: w < 20 or h < 20),
    ('小', '20x20~32x32', lambda w, h: (20 <= w < 32 or 20 <= h < 32)),
    ('中', '32x32~96x96', lambda w, h: (32 <= w < 96 or 32 <= h < 96)),
    ('大', '≥96x96', lambda w, h: w >= 96 or h >= 96),
]

def get_size_group(width, height):
    for name, _, cond in GROUPS:
        if cond(width, height):
            return name
    return GROUPS[-1][0]  # 默认分到最后一组（大）

# 读取YOLO格式txt
def read_yolo_txt(txt_path, img_w, img_h):
    objs = []
    if not os.path.exists(txt_path):
        return objs
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            # 归一化转为绝对坐标
            abs_w = w * img_w
            abs_h = h * img_h
            xmin = (x - w/2) * img_w
            ymin = (y - h/2) * img_h
            xmax = (x + w/2) * img_w
            ymax = (y + h/2) * img_h
            group = get_size_group(abs_w, abs_h)
            objs.append({'cls': cls, 'bbox': [xmin, ymin, xmax, ymax], 'group': group, 'conf': float(parts[5]) if len(parts) > 5 else 1.0})
    return objs

# 计算IoU
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# 计算每组mAP
def compute_ap(rec, prec):
    """VOC2007 11点插值法"""
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.
    return ap

def compute_metrics(gt_boxes, preds, iou_thresh=0.5):
    preds = sorted(preds, key=lambda x: -x[1])
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    detected = []
    for i, (box_pred, conf) in enumerate(preds):
        ious = [iou(box_pred, box_gt) for box_gt in gt_boxes]
        if len(ious) > 0 and max(ious) >= iou_thresh and np.argmax(ious) not in detected:
            tp[i] = 1
            detected.append(np.argmax(ious))
        else:
            fp[i] = 1
    npos = len(gt_boxes)
    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    rec = tp_cum / npos if npos else np.zeros_like(tp_cum)
    prec = tp_cum / (tp_cum + fp_cum + 1e-16)
    ap = compute_ap(rec, prec) if npos else 0
    recall = rec[-1] if len(rec) else 0
    precision = prec[-1] if len(prec) else 0
    return ap, recall, precision, npos

def main(args):
    # 初始化模型
    model = YOLO(args.model)
    # 统计结构
    group_names = [f"{name}（{desc}）" for name, desc, _ in GROUPS]
    group_gts = {name: [] for name, _, _ in GROUPS}
    group_preds = {name: [] for name, _, _ in GROUPS}

    # 读取图片列表，并处理为绝对路径
    img_list_dir = os.path.dirname(os.path.abspath(args.img_list))
    with open(args.img_list, 'r') as f:
        img_paths = [os.path.normpath(os.path.join(img_list_dir, line.strip())) for line in f if line.strip()]

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"图片读取失败: {img_path}")
            continue
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(args.gt_dir, base + '.txt')

        # 读取标注
        gts = read_yolo_txt(gt_path, w, h)
        for obj in gts:
            group_gts[obj['group']].append(obj)

        # 推理
        results = model.predict(img_path, conf=args.conf, iou=args.iou_thresh, save=False, verbose=False)
        preds = []
        for r in results:
            if hasattr(r, 'boxes'):
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    xmin, ymin, xmax, ymax = xyxy
                    width = xmax - xmin
                    height = ymax - ymin
                    group = get_size_group(width, height)
                    preds.append({'cls': cls, 'bbox': [xmin, ymin, xmax, ymax], 'group': group, 'conf': conf})
        for obj in preds:
            group_preds[obj['group']].append(obj)

    # 统计与输出
    for name, desc, _ in GROUPS:
        group_name = f"{name}（{desc}）"
        gt_cls_boxes = defaultdict(list)
        pred_cls_boxes = defaultdict(list)
        for obj in group_gts[name]:
            gt_cls_boxes[obj['cls']].append(obj['bbox'])
        for obj in group_preds[name]:
            pred_cls_boxes[obj['cls']].append((obj['bbox'], obj['conf']))

        print(f'分组: {group_name}')
        ap_list, recall_list, prec_list, npos_list = [], [], [], []
        print(f"{'类别':<8}{'AP':<10}{'Recall':<10}{'Precision':<10}{'目标数':<8}")
        for cls in sorted(set(list(gt_cls_boxes.keys()) + list(pred_cls_boxes.keys()))):
            gt_boxes = gt_cls_boxes[cls]
            preds = pred_cls_boxes[cls]
            ap, recall, precision, npos = compute_metrics(gt_boxes, preds, args.iou_thresh)
            ap_list.append(ap)
            recall_list.append(recall)
            prec_list.append(precision)
            npos_list.append(npos)
            print(f"{str(cls):<8}{ap:<10.4f}{recall:<10.4f}{precision:<10.4f}{npos:<8}")
        if ap_list:
            print(f"{'平均':<8}{np.mean(ap_list):<10.4f}{np.mean(recall_list):<10.4f}{np.mean(prec_list):<10.4f}{np.sum(npos_list):<8}")
        print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="边推理边按目标大小分组统计mAP/Recall/Precision")
    parser.add_argument('--model', type=str, required=True, help='模型权重路径(.pt)')
    parser.add_argument('--img_list', type=str, required=True, help='图片列表txt文件，每行一个图片路径（相对路径基于本文件）')
    parser.add_argument('--gt_dir', type=str, required=True, help='验证集标注txt路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU阈值')
    args = parser.parse_args()
    main(args)
