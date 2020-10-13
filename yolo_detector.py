# import the necessary packages
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt
# import geopandas as gpd
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
img,boxes, ids=obj.process_image("F:\\TFM_datasets\\extracted_frames\\000101\\70.jpg")
# plt.imshow(img)
# plt.pause(0.1)
rectangles= []
intersections = []
for sub_box in boxes:
    rectangles.append(box(abs(int(sub_box[0])),abs(int(sub_box[1])),abs(int(sub_box[2])),abs(int(sub_box[3]))))

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def return_xywh(coordinates):
    return  coordinates[0],coordinates[1],coordinates[2]-coordinates[0],coordinates[3]-coordinates[1]        

#TODO view intersections!
i = len(rectangles)-1

# for rectangle in rectangles:
#     print(rectangle.exterior.coords.xy)
# while (i>0):
#     rectangles_copy = rectangles.copy()
#     j = len(rectangles_copy)-1
#     while(j>0):
#         if(i!=j):
#             intersection = rectangles[i].intersection(rectangles_copy[j])
#             # print("1: ",rectangles[i].bounds)
#             # print("2: ",rectangles_copy[j].bounds)
#             if(not(intersection.is_empty)):
#                 # intersections.append(union(return_xywh(rectangles[i].bounds),return_xywh(rectangles_copy[j].bounds)))
#                 intersections.append(intersection)
#         del rectangles_copy[j]
#         j = j-1
#     del rectangles[i]
#     i = i-1
# print(len(intersections))
# for intersection in intersections:
#     gpd.GeoSeries(intersection).plot()
#     # cv2.rectangle(img, (int(intersection[0]), int(intersection[2])), (int(intersection[1]+intersection[0]), int(intersection[3]+intersection[2])), (255, 0, 0), 2)
       
# plt.imshow(img)
# plt.pause(0.1)    



