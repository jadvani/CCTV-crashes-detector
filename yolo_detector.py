# import the necessary packages
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt

from shapely.geometry import box


class yolo_detector():

    
    def __init__(self,coco_folderpath, confidence=0.5,threshold=0.3):
        self.coco_folder_path = coco_folderpath
        self.confidence=confidence
        self.threshold = threshold
        self.COLORS=[]
        self.LABELS=[]
        labelsPath = os.path.sep.join([coco_folderpath, "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # colores para representar clases
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),dtype="uint8")
        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([coco_folderpath, "yolov3.weights"])
        configPath = os.path.sep.join([coco_folderpath, "yolov3.cfg"])
        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    def print_folderpath(self):
        print(self.coco_folder_path)
    def process_image(self, image_path):
        # load our input image and grab its spatial dimensions
        image = cv2.imread(image_path)
        (H, W) = image.shape[:2]
        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()
        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        dict_result = dict()
        for output in layerOutputs:
            for detection in output:
                scores=detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence,self.threshold)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dict_result[i] = classIDs[i], boxes[i]
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
        return image, boxes, classIDs
                
                


obj = yolo_detector("C:\\Users\\Javier\\Downloads\\darknet-master\\cfg",0.2,0.3)
obj.print_folderpath()
img,boxes, ids=obj.process_image("F:\\TFM_datasets\\extracted_frames\\000051\\130.jpg")
# plt.imshow(img)
# plt.pause(0.1)
rectangles= []
interesections = []
for sub_box in boxes:
    rectangles.append(box(abs(int(sub_box[0])),abs(int(sub_box[1])),abs(int(sub_box[2])),abs(int(sub_box[3]))))

#TODO view intersections!
for i in range(0, len(rectangles)-1):
    for j in range(0, len(rectangles)-1):
        if(i!=j):
            intersection = rectangles[i].intersection(rectangles[j])
            if(not(intersection.is_empty)):
                if not any(p.equals(intersection) for p in interesections):
                    interesections.append(intersection)

for intersection in interesections:
    print(intersection)
    
    
    
    

        


