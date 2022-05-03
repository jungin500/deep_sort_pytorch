__all__ = ['build_detector']

def build_detector(cfg, use_cuda):
    if cfg.USE_MMDET:
        from .MMDet import MMDet
        return MMDet(cfg.MMDET.CFG, cfg.MMDET.CHECKPOINT,
                    score_thresh=cfg.MMDET.SCORE_THRESH,
                    is_xywh=True, use_cuda=use_cuda)
    else:
        from .YOLOv3 import YOLOv3
        return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, 
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                    is_xywh=True, use_cuda=use_cuda)
