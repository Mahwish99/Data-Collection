import numpy as np
import cv2

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 100, size=(len(classes), 3))


def models(dimensions):
# loading image
    img = cv2.imread('car.jpg')
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (dimensions, dimensions), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                label = str(classes[class_id])
                print('label: ',label,'\theight: ',h,'\twidth: ',w,'\tProbability: ',confidence)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    cv2.imshow("Output", img)
    cv2.waitKey(0)

models(608)