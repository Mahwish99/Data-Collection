import numpy as np
import cv2


def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []
    for output in outputs:
        for detection in output:
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def get_yolo_prediction(net, layer_names, labels, image, confidence, threshold, input_shape):
    height, width = image.shape[:2]
    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (input_shape, input_shape), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)
    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)
    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
    return boxes, confidences, classIDs, idxs

def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, color, labels, input_shape):
    box = []
    confi = []
    classes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            box.append([(x,y),(x + w, y + h)])
            confi.append(confidences[i])
            classes.append(classIDs[i])
            # draw the bounding box and label on the image
            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            #print(input_shape,'label: ', labels[classIDs[i]], '\theight: ', h, '\twidth: ', w, '\tProbability: ', confidences[i])
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return box, confi, classes, image

def yolo_detection(cfg, weights, image, input_shape, confidence, threshold, labels, color):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    labels = open(labels).read().strip().split('\n')
    boxes, confidences, classIDs, idxs = get_yolo_prediction(net, layer_names, labels, image, confidence, threshold, input_shape)
    boxes, confidences, classIDs, image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, color, labels, input_shape)
    return boxes, confidences, classIDs, image
