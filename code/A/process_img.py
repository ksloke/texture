#process image into numpy arrays to be saved

import os
import PIL.Image as Image
import numpy as np
import math
import numpy as np
import itertools



def replace_all(text, dic):
  for i, j in dic.items():
      text = text.replace(i, j)
  return text
def getClass(file):
    fno=int(replace_all(file, {".jpg" : "", ".png" : "", ".bmp": ""}))
    cid=math.trunc(fno/20) #class no in outex
    return cid


#directory = '/content/drive/MyDrive/Outex_TC_00013/Outex_TC_00013/images'
directory = '../Texture/texure_2/Outex_TC_00013/Outex_TC_00013/images'
# size = 128x128
# Total 1369 
# 68 classes each 20 samples

#file=files[0] #grab first file for testing
images = []
labels = []
files = os.listdir(directory)
for file in files:
  img = Image.open(directory + '/' + file)
  img = img.convert("RGB") # for 3 channel PNG 
  #imggray=img.convert('LA') # convert to gray scale 

  imgWidth, imgHeight = img.size
  print(file, img.size) #128x128

  rimg=np.array(img)
  print(rimg.shape)
  images.append(rimg)
  labels.append(getClass(file))



images = np.array(images)
labels = np.array(labels)
print(images.shape)
print(labels.shape)
with open('img.npy','wb') as f:
    np.save(f,images)
    np.save(f,labels)