#!/usr/bin/env python3
"""
Convert COCO WholeBody annotations to COCO format with only face boxes.
Usage:
  python wholebody2coco.py --input INPUT_JSON --output OUTPUT_JSON
"""
import argparse
import json

def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO WholeBody annotations to COCO format with only face boxes.')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input COCO WholeBody annotation JSON file.')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output COCO annotation JSON file.')
    args = parser.parse_args()

    data = json.load(open(args.input, 'r'))
    # Copy images as is
    images = data.get('images', [])

    # Convert annotations: keep only valid face boxes
    new_anns = []
    for ann in data.get('annotations', []):
        if not ann.get('face_valid', False):
            continue
        face_box = ann.get('face_box')
        if face_box is None or not any(face_box):
            continue
        x, y, w, h = face_box
        if w <= 0 or h <= 0:
            continue
        new_ann = {
            'id': ann.get('id', len(new_anns) + 1),
            'image_id': ann['image_id'],
            'category_id': 1,
            'segmentation': [],
            'area': float(w * h),
            'bbox': [x, y, w, h],
            'iscrowd': ann.get('iscrowd', 0),
        }
        new_anns.append(new_ann)

    # Define face category
    categories = [
        {'id': 1, 'name': 'face', 'supercategory': 'face'}
    ]

    coco = {
        'images': images,
        'annotations': new_anns,
        'categories': categories
    }

    with open(args.output, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f'Wrote {len(images)} images and {len(new_anns)} face annotations to {args.output}')

if __name__ == '__main__':
    main() 