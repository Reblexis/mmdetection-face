#!/usr/bin/env python3
"""
Visualize COCO face annotations using OpenCV.
Usage:
  python visualize_coco_faces.py --ann_file ANNOT_JSON --img_dir IMAGE_DIR
Controls:
  RIGHT / d : next image
  LEFT / a  : previous image
  q         : quit
"""
import argparse
import json
import os
import cv2

def main():
    parser = argparse.ArgumentParser(
        description='Visualize COCO face annotations')
    parser.add_argument('--ann_file', '-a', required=True,
                        help='Path to COCO annotation JSON file')
    parser.add_argument('--img_dir', '-d', required=True,
                        help='Directory containing images')
    args = parser.parse_args()

    data = json.load(open(args.ann_file, 'r'))
    images = data.get('images', [])
    annotations = data.get('annotations', [])

    # Map image_id to annotations
    ann_map = {}
    for ann in annotations:
        ann_map.setdefault(ann['image_id'], []).append(ann)

    # Print total number of faces
    total_faces = sum(len(v) for v in ann_map.values())
    print(f"Total face annotations: {total_faces}")

    idx = 0
    n = len(images)
    if n == 0:
        print('No images found in annotation file')
        return

    while True:
        img_info = images[idx]
        # Skip images that have no face annotations
        if img_info['id'] not in ann_map or len(ann_map[img_info['id']]) == 0:
            idx = (idx + 1) % n
            continue
        img_path = os.path.join(args.img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f'Could not read {img_path}')
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):
                break
            idx = (idx + 1) % n
            continue

        # Draw all face bboxes
        for ann in ann_map.get(img_info['id'], []):
            x, y, w, h = ann['bbox']
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

        # Overlay filename and index
        disp = img.copy()
        cv2.putText(
            disp,
            f"{idx+1}/{n}: {img_info['file_name']}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2)

        cv2.imshow('Face Annotations', disp)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key in (ord('d'), 83):  # d or right arrow
            idx = (idx + 1) % n
        elif key in (ord('a'), 81):  # a or left arrow
            idx = (idx - 1) % n
        else:
            idx = (idx + 1) % n

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 