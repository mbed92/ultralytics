from ultralytics import YOLO

PATH = 'stock.mp4'
MODEL = 'yolov8n.pt'

# inference
model = YOLO(MODEL)

# config = {'task': 'segment', 'mode': 'train', 'model': 'yolov8n-seg.yaml', 'data': 'coco.yaml', 'patience': 50,
#           'imgsz': 640, 'save': True, 'workers': 8, 'exist_ok': False, 'pretrained': False, 'optimizer': 'SGD',
#           'verbose': False, 'seed': 0, 'deterministic': True, 'single_cls': False, 'image_weights': False,
#           'rect': False, 'cos_lr': False, 'close_mosaic': 10, 'resume': False, 'overlap_mask': True, 'mask_ratio': 4,
#           'dropout': False, 'val': True, 'save_hybrid': False, 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'half': False,
#           'dnn': False, 'plots': True, 'source': 'ultralytics/assets/', 'show': False, 'save_txt': False,
#           'save_conf': False, 'save_crop': False, 'hide_labels': False, 'hide_conf': False, 'vid_stride': 1,
#           'line_thickness': 3, 'visualize': False, 'augment': False, 'agnostic_nms': False, 'retina_masks': False,
#           'format': 'torchscript', 'keras': False, 'optimize': False, 'int8': False, 'dynamic': False,
#           'simplify': False, 'opset': 17, 'workspace': 4, 'nms': False, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937,
#           'weight_decay': 0.001, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 7.5,
#           'cls': 0.5, 'dfl': 1.5, 'fl_gamma': 0.0, 'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7,
#           'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0,
#           'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0,
#           'hydra': {'output_subdir': None, 'run': {'dir': '.'}}, 'v5loader': False}

# customize a config
config = {
    'save': True
}

outputs = model.predict(source=PATH, **config)
