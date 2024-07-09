import cv2
import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_resnet_features(image):
  
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    features = resnet_model.predict(img)
    return features

sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")

best_score = 0 
filename = None
image = None
best_features = None

counter = 0
for file in os.listdir("SOCOFing/Real")[:1000]:
    fingerprint_image = cv2.imread(os.path.join("SOCOFing/Real", file))


    fingerprint_features = extract_resnet_features(fingerprint_image)
    sample_features = extract_resnet_features(sample)

    similarity_score = np.dot(sample_features, fingerprint_features.T) / (np.linalg.norm(sample_features) * np.linalg.norm(fingerprint_features))

    if similarity_score > best_score:
        best_score = similarity_score
        filename = file
        image = fingerprint_image
        best_features = fingerprint_features

        print("BEST MATCH:", filename) 
        print("SCORE:", best_score * 100, "%")

cv2.imshow("Best Match", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
