import cv2
import numpy as np
import os
import time 
def dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person 
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image, 0))
            labels.append(person)

    return(images, np.array(labels), labels_dic)

images, labels, labels_dic = dataset()

cap = cv2.VideoCapture(1)
count1 = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imwrite('nibba' + str(count1) + '.jpg', frame)
    count1 += 1
    time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

class FaceDetector(object):

    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30,30)
        biggest_only = True
        faces_coord = self.classifier.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE)
        
        return faces_coord

def cut_faces(image, faces_coord):
    faces = []

    for(x, y, w, h) in faces_coord:
        w_rm = int(.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    return faces

def resize(images, size=(224, 224)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

        images_norm.append(image_norm)

    return images_norm

def normalize_faces(image, faces_coord):

    faces = cut_faces(image, faces_coord)
    faces = resize(faces)
    
    return faces

for image in images:
    count = 0
    detector = FaceDetector("haarcascade_frontalface_default.xml")
    faces_coord = detector.detect(image, True)
    faces = normalize_faces(image, faces_coord)
    for i, face in enumerate(faces):
            cv2.imwrite('%s.jpeg' % (count), faces[i])
            count += 1  