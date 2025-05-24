# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .coco import CocoDataset


@DATASETS.register_module()
class CocoWholeBodyDataset(CocoDataset):
    """Dataset for COCO WholeBody face detection."""

    METAINFO = {
        'classes': ('face', ),  # Only detecting faces
        'palette': [(220, 20, 60)]  # Single color for face bounding boxes
    }

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue

            # Check if face annotation is valid
            if not ann.get('face_valid', False):
                continue
                
            face_bbox = ann.get('face_box')  # [x, y, w, h] format
            if face_bbox is None or not any(face_bbox):  # Skip if bbox is None or all zeros
                continue

            x1, y1, w, h = face_bbox

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]

            instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = 0  # Only one class (face)

            instances.append(instance)
        
        data_info['instances'] = instances
        return data_info 