# import the necessary packages
import numpy as np
import time
import cv2
import os
# import geopandas as gpd


class yolo_detector():

    
    def __init__(self,coco_folderpath, confidence=0.5,threshold=0.3, draw_over_image=False, coord_similarity=20):
        self.coco_folder_path = coco_folderpath
        self.draw_over_image = draw_over_image
        self.confidence=confidence
        self.coord_similarity = coord_similarity
        self.threshold = threshold
        labelsPath = os.path.sep.join([coco_folderpath, "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # colores para representar clases
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),dtype="uint8")
        # cargamos el modelo y los pesos de YOLO
        weightsPath = os.path.sep.join([coco_folderpath, "yolov3.weights"])
        configPath = os.path.sep.join([coco_folderpath, "yolov3.cfg"])
        # cargamos el detector de YOLO sobre el dataset COCO (80 clases)
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.original_image, self.coord_unions, self.potential_crashes = ([],[],[])

        
    def print_coco_names_folderpath(self):
        print(self.coco_folder_path)
        
    def get_yolo_single_boxes(self, W, H, layerOutputs, image):
               # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes, confidences, classIDs = ([],[],[])
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
                if(self.draw_over_image):
                    color = [int(c) for c in self.COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    			0.5, color, 2)
        return image, boxes, classIDs
        
    def process_image(self, image):
        # load our input image and grab its spatial dimensions
        self.original_image = image
        (H, W) = image.shape[:2]
        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()
        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        # bounding boxes from image. Image is returned in case we set draw_over_image as True in class
        image, boxes, classIDs = self.get_yolo_single_boxes(W, H, layerOutputs, image)

 
        return image, boxes, classIDs
                
    def union(self, a,b):
      x = min(a[0], b[0])
      y = min(a[1], b[1])
      w = max(a[0]+a[2], b[0]+b[2]) - x
      h = max(a[1]+a[3], b[1]+b[3]) - y
      return (x, y, w, h)
    
    def return_xywh(self, coordinates):
        return  coordinates[0],coordinates[1],coordinates[2]-coordinates[0],coordinates[3]-coordinates[1]       
     
    def intersection(self, a,b):
      x = max(a[0], b[0])
      y = max(a[1], b[1])
      w = min(a[0]+a[2], b[0]+b[2]) - x
      h = min(a[1]+a[3], b[1]+b[3]) - y
      if w<0 or h<0: return () # or (0,0,0,0) ?
      return (x, y, w, h)
    
    # comprobamos si las tuplas, o cualquier otra estructura de datos, está vacía o no.
    def is_empty(self, any_structure):
        return False if any_structure else True
    
    #given a limit number, check if two ints are similar or not. 
    def number_in_range(self,num1,num2):
        return (abs(num1-num2)<=self.coord_similarity and abs(num1-num2)>=0)

# given two coordinates as tuples, check if these are very similar or not. This means that a similar region was already grabbed to be studied 
    def similar_tuples(self,a,b):
        return self.number_in_range(a[0],b[0]) and self.number_in_range(a[1],b[1]) and self.number_in_range(a[2],b[2]) and self.number_in_range(a[3],b[3])
    def similar_tuple_in_list(self, tuple_to_analyze):
        for union in self.coord_unions:
            if(self.similar_tuples(tuple_to_analyze, union)):
                return True
        return False
    
    def get_union_areas(self, boxes):
        for i in range(0, len(boxes)-1):
            for j in range(0, len(boxes)-1):
                if(i!=j):
                    intersect = self.intersection(boxes[i],boxes[j])
                    if(not(self.is_empty(intersect))):
                        union_from_intersection = self.union(boxes[i],boxes[j])
                        if(union_from_intersection not in self.coord_unions and not self.similar_tuple_in_list(union_from_intersection)):
                            self.coord_unions.append(union_from_intersection)
                            crop = self.original_image[union_from_intersection[1]:union_from_intersection[1]+union_from_intersection[3], union_from_intersection[0]:union_from_intersection[0]+union_from_intersection[2]]
                            self.potential_crashes.append(crop)
                                        
# # import matplotlib.pyplot as plt
# yolo = yolo_detector("C:\\Users\\Javier\\Downloads\\darknet-master\\cfg",0.2,0.3, draw_over_image=False, coord_similarity=20)
# yolo.print_coco_names_folderpath()
# final_crashes = []
# res = []
# possible_crash=cv2.imread('F:\\TFM_datasets\\extracted_frames\\000079\\20.jpg')
# img,boxes, ids=yolo.process_image(possible_crash)
# yolo.get_union_areas(boxes)
# potential_crashes=yolo.potential_crashes
# i = 0
# print(len(yolo.coord_unions))
# for coord in yolo.coord_unions:
#     print(coord)
    