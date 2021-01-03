import cv2

class Model:
    def __init__(self, model, config, image, labels):
        self.tensorflowNet = cv2.dnn.readNetFromTensorflow(model, config)
        self.img = image
        self.labels = open(labels).read().strip().split('\n')

    def detection(self, confidence, model_name):
        boxes = []
        labels = []
        confidences = []
        # Input image
        rows, cols, channels = self.img.shape
        # Use the given image as input, which needs to be blob(s).
        self.tensorflowNet.setInput(cv2.dnn.blobFromImage(self.img, size=(300, 300), swapRB=True, crop=False))
        # Runs a forward pass to compute the net output
        networkOutput = self.tensorflowNet.forward()
        # Loop on the outputs
        for detection in networkOutput[0, 0]:
            score = float(detection[2])
            if score > confidence:
                labels.append(int(detection[1]))
                confidences.append(score)
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                boxes.append([(int(left), int(top)), (int(right), int(bottom))])
                # draw a red rectangle around detected objects
                # cv2.rectangle(self.img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                #print(model_name,'label: ', self.labels[int(detection[1])], '\theight: ', int(bottom), '\twidth: ', int(right), '\tProbability: ',score)
                # print(len(self.labels))
                # print(len(detection))
                # print(detection[1])
                if detection[1]<80:
                    text = "{}: {:.4f}".format(self.labels[int(detection[1])], score)
                    # cv2.putText(self.img,text,(int(left), int(top - 5)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)

        # Show the image with a rectagle surrounding the detected objects
        return boxes, confidences, labels, self.img
