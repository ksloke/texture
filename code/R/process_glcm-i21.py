# 9 Sep 2021 process glcm only and save into 
# glcm-i21-64L.npy or
# glcm-i21-16L.npy as arrays
#

import os
import PIL.Image as Image
import numpy as np
import math
import numpy as np
import itertools

def get_quant(i, level):
    out=i//level
    return out

def get_64quant(i):
    return get_quant(i,64)

def get_16quant(i):
    return get_quant(i,16)

def second_angular_monent(glcm):
    return np.sum(glcm*glcm)

def contrast(glcm):
    w,h=glcm.shape
    sum=0
    for i in range(0,w):
        for j in range(0,h):
            sum+=(i-j)*(i-j)*glcm[i,j]
    return sum

def glcm_features(glcm):
    w,h=glcm.shape
    contrast=0 # inverse of idm, weights further from diagonals
    mean, var=0,0
    ent=0
    idm=0 #local homogeneity - closeness to diagonal
    for i in range(0,w):
        for j in range(0,h):
            ij2=(i-j)*(i-j)
            contrast+=ij2*glcm[i,j]
            mean+=i*glcm[i,j]
            idm+=(1/(1+ij2))*glcm[i,j]
            if(glcm[i,j] != 0):
                ent+=glcm[i,j]*math.log2(glcm[i,j])
    for i in range(0,w):
        for j in range(0,h):
            var+=(i-mean)*(i-mean)*glcm[i,j]
    corr=0
    for i in range(0,w):
        for j in range(0,h):
            if(var!=0):
                corr+=((i-mean)*(j-mean)/var)*glcm[i,j]
    sam=np.sum(glcm*glcm)  #second angular moment       
    largest=np.where(glcm == glcm.max()) #index where the largest element is  
    #print(largest)
    #features = contrast, idm, entropy, variance, correlation, second angular moment, largest
    return [contrast, idm, abs(ent), var, corr, sam, np.sum(largest)]

def replace_all(text, dic):
  for i, j in dic.items():
      text = text.replace(i, j)
  return text
def getClass(file):
    fno=int(replace_all(file, {".jpg" : "", ".png" : "", ".bmp": ""}))
    cid=math.trunc(fno/20) #class no in outex
    return cid

# 6x6 receptive field for 4 3x3 glcm window - assumed all symmetrical squares
# glcm haralick features in each 3x3 glcm window

#00034 has 2 sets of separate test images - we ignore them
directory = '../Texture/texure_2/Outex_TC_00034/Outex-TC-00034/images' 
# size = 128x128
# Total 1369 
# 68 classes each 20 samples

#receptive window size
height,width=5,5
index = 0
images = []
labels = []
#file=files[0] #grab first file for testing

files = os.listdir(directory)
u=0
for file in files:
  img = Image.open(directory + '/' + file)
  img = img.convert("RGB") # for 3 channel PNG 
  #imggray=img.convert('LA') # convert to gray scale 

  if(u>=1360): #ignore the rest of files
      break

  imgWidth, imgHeight = img.size
  print(u, file, img.size) #128x128

  stride=height-2
  #glcm features of image array
  size=int(np.ceil((imgHeight-height+1)/stride)) #size =42
  #print(size) #42
  g=np.zeros([24, size-1,size-1]) #size-1 = 41 .. 0 - 40

  #for each image, extract window 
  i,j=0,0
  for h in range(0, imgHeight-height, stride):
      i=0
      for w in range(0, imgWidth-width, stride):
          
          box = (w, h, w+width, h+height)
          rfImage = np.array(img.crop(box)).astype(int)
        
          #extract glcm here from 5x5 window
          glcm1=np.zeros([16,16]) #change this 64 or 16
          glcm2=np.zeros([16,16])
          glcm3=np.zeros([16,16])
          #print(box)
          for k in range(1,4,2): #returns the glcm center (1,1), (1,3)
              for l in range(1,4,2): # (1,1), (1,3)
                  glcm1[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k-1,l-1][0])]+=1
                  glcm1[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k-1,l][0])]+=1
                  glcm1[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k-1,l+1][0])]+=1
                  glcm1[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k,l-1][0])]+=1
                  glcm1[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k,l+1][0])]+=1
                  glcm1[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k+1,l-1][0])]+=1
                  glcm1[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k+1,l][0])]+=1
                  glcm1[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k+1,l-1][0])]+=1

                  glcm2[get_16quant(rfImage[k,l][0]),get_16quant(rfImage[k-1,l-1][2])]+=1
                  glcm2[get_16quant(rfImage[k,l][0]),get_16quant(rfImage[k-1,l][2])]+=1
                  glcm2[get_16quant(rfImage[k,l][0]),get_16quant(rfImage[k-1,l+1][2])]+=1
                  glcm2[get_16quant(rfImage[k,l][0]),get_16quant(rfImage[k,l-1][2])]+=1
                  glcm2[get_16quant(rfImage[k,l][0]),get_16quant(rfImage[k,l+1][2])]+=1
                  glcm2[get_16quant(rfImage[k,l][0]),get_16quant(rfImage[k+1,l-1][2])]+=1
                  glcm2[get_16quant(rfImage[k,l][0]),get_16quant(rfImage[k+1,l][2])]+=1
                  glcm2[get_16quant(rfImage[k,l][0]),get_16quant(rfImage[k+1,l-1][2])]+=1

                  glcm3[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k-1,l-1][2])]+=1
                  glcm3[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k-1,l][2])]+=1
                  glcm3[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k-1,l+1][2])]+=1
                  glcm3[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k,l-1][2])]+=1
                  glcm3[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k,l+1][2])]+=1
                  glcm3[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k+1,l-1][2])]+=1
                  glcm3[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k+1,l][2])]+=1
                  glcm3[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k+1,l-1][2])]+=1

          glcm1=glcm1/32 #total 32 pixels pairs in 5x5 window
          glcm2=glcm2/32 
          glcm3=glcm3/32
          #print(glcm)
          #features = contrast, idm, entropy, variance, correlation, second angular moment, largest
          fea1=glcm_features(glcm1)
          fea2=glcm_features(glcm2)
          fea3=glcm_features(glcm3)
          #print(fea) 
          
          g[0,i,j]=fea1[0]
          g[1,i,j]=fea1[1]
          g[2,i,j]=fea1[2]
          g[3,i,j]=fea1[3]
          g[4,i,j]=fea1[4]
          g[5,i,j]=fea1[5]
          g[6,i,j]=fea1[6]

          g[7,i,j]=fea2[0]
          g[8,i,j]=fea2[1]
          g[9,i,j]=fea2[2]
          g[10,i,j]=fea2[3]
          g[11,i,j]=fea2[4]
          g[12,i,j]=fea2[5]
          g[13,i,j]=fea2[6]

          g[14,i,j]=fea3[0]
          g[15,i,j]=fea3[1]
          g[16,i,j]=fea3[2]
          g[17,i,j]=fea3[3]
          g[18,i,j]=fea3[4]
          g[19,i,j]=fea3[5]
          g[20,i,j]=fea3[6]
          i+=1
      j+=1 
       
  rimg=np.array(img.resize((41,41)))
  #print(rimg.shape)

  g[21]=rimg[:,:,0]
  g[22]=rimg[:,:,1]
  g[23]=rimg[:,:,2]
  images.append(g)
  labels.append(getClass(file))
  del glcm1
  del glcm2
  del glcm3
  del g
  del rfImage
  u+=1

images = np.array(images)
labels = np.array(labels)
print(images.shape)
print(labels.shape)

# or glcm-i21-64L.npy
with open('glcm-i21-16L.npy','wb') as f: 
    np.save(f,images)
    np.save(f,labels)