from YOLO import yolo_detection
from Model import Model
from util import utility
import cv2
from xml.dom import minidom
import xml.etree.ElementTree as xml

def detection(frame, i, path_xml):
    img = frame
    # YOLO 416 DETECTION
    boxes_416, confidences_416, class_IDs_416, image = yolo_detection('yolov3.cfg', 'yolov3.weights', img, 416, 0.7,
                                                                      0.7, './coco.names', (255, 0, 0))
    print('No. of boxes in YOLO 416', len(boxes_416))
    # RELOAD IMAGE TO AVOID OVER WRITING
    img = frame
    # YOLO 320 DETECTION
    boxes_320, confidences_320, class_IDs_320, image1 = yolo_detection('yolov3.cfg', 'yolov3.weights', img, 320, 0.7,
                                                                       0.7, './coco.names', (255, 255, 0))
    print('No. of boxes in YOLO 320', len(boxes_320))
    # RELOAD IMAGE TO AVOID OVER WRITING
    img = frame
    # YOLO 608 DETECTION
    boxes_608, confidences_608, class_IDs_608, image2 = yolo_detection('yolov3.cfg', 'yolov3.weights', img, 608, 0.7,
                                                                       0.7, './coco.names', (0, 0, 255))
    print('No. of boxes in YOLO 608', len(boxes_608))
    # RELOAD IMAGE TO AVOID OVER WRITING
    img = frame
    # YOLO v4
    boxes_v4, confidences_v4, class_IDs_v4, image7 = yolo_detection('yolov4.cfg', 'yolov4.weights', img, 416, 0.7,
                                                                       0.7, './coco.names', (0, 0, 255))
    print('No. of boxes in YOLO v4', len(boxes_v4))
    # RELOAD IMAGE TO AVOID OVER WRITING
    img = frame
    # SSD inception DETECTION
    ssd = Model('ssd_inception_frozen_inference_graph.pb', 'cfg.pbtxt', img, 'coco.names')
    boxes_ssd, confidences_ssd, class_IDs_ssd, image3 = ssd.detection(confidence=0.7, model_name='SSD Inception')
    print('No. of boxes in SSD Inception', len(boxes_ssd))
    # RELOAD IMAGE TO AVOsdfnmID OVER WRITING
    img = frame
    # SSD mobilenet DETECTION
    ssd = Model('ssd_mobilenet_frozen_inference_graph.pb', 'mobilnet_cfg.pbtxt', img, 'coco.names')
    boxes_ssd1, confidences_ssd1, class_IDs_ssd1, image4 = ssd.detection(confidence=0.7, model_name='SSD MobileNet')
    print('No. of boxes in SSD mobilent', len(boxes_ssd1))
    # RELOAD IMAGE TO AVOID OVER WRITING
    img = frame
    # RCNN inception DETECTION
    rcnn = Model('rcnn_frozen_inference_graph.pb', 'graph.pbtxt', img, 'coco.names')
    boxes_rcnn, confidences_rcnn, class_IDs_rcnn, image5 = rcnn.detection(confidence=0.7, model_name='RCNN Inception')
    print('No. of boxes in RCNN inception', len(boxes_rcnn))
    # RELOAD IMAGE TO AVOID OVER WRITING
    img = frame
    rcnn = Model('rcnn_resnet_frozen_inference_graph.pb', 'resnet_graph.pbtxt', img, 'coco.names')
    boxes_rcnn1, confidences_rcnn1, class_IDs_rcnn1, image6 = rcnn.detection(confidence=0.7, model_name='RCNN resnet')
    print('No. of boxes in RCNN resnet', len(boxes_rcnn1))
    # CREATE BOUNDING BOX AND XML FILE
    all_boxes = [boxes_416, boxes_320]
    # RCNN resnet DETECTION0, boxes_608, boxes_ssd, boxes_ssd1, boxes_rcnn, boxes_rcnn1]
    all_detected_classes = [class_IDs_416, class_IDs_320, class_IDs_608, class_IDs_ssd, class_IDs_ssd1, class_IDs_rcnn,
                            class_IDs_rcnn1]
    img = frame
    utility(all_boxes, all_detected_classes, img, i, path_xml)
