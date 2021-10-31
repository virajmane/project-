from calorie import calories
from cnn_model import get_model
import os  
import cv2
import numpy as np
import time

start = time.time()
IMG_SIZE = 400
LR1 = 1e-3
no_of_fruits=7

MODEL_NAME = 'Fruits_dectector-{}-{}.model'.format(LR1, '5conv-basic')

model_save_at=os.path.join("model",MODEL_NAME)

model=get_model(IMG_SIZE,no_of_fruits,LR1)

model.load(model_save_at)
labels=list(np.load('labels.npy'))

test_data='test.jpg'
img=cv2.imread(test_data)
img1=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
model_out=model.predict([img1])
result=np.argmax(model_out)
name=labels[result]
cal=round(calories(result+1,img),2)
print("FOOD",name)
end = time.time()
print("The time of execution of above program is :", end-start)
import matplotlib.pyplot as plt
plt.imshow(img)
#cal=0.996476
plt.title('{0}({1}kcal)'.format(name,cal))
print(cal)
plt.title(name)

plt.axis('off')
plt.show()

