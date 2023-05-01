#process glcm only and save into glcm.npy as arrays
# 1 color pair only - accuracy about 75-80%
import os
import PIL.Image as Image
import numpy as np
import math
import numpy as np
import itertools

def get_8quant(color):
    # colors divided into 8 quantized regions for each color space
    eight = [[0,31], [32,63], [64,95], [96,127], [128,159], [160,191], [192,223], [224,255]]
    for i, value in enumerate(eight):
        #print(color_value)
        if color >= value[0] and  color <= value[1]:
            return i

def get_16quant(color):
    # colors divided into 8 quantized regions for each color space
    eight = [[0,15], [16,31], [32,47],[48,63], [64,79],[80,95], [96,111],[112,127], 
             [128,143], [144,159], [160,175],[176,191], [192,207],[208,223], [224,239],[240,255]]
    for i, value in enumerate(eight):
        #print(color_value)
        if color >= value[0] and  color <= value[1]:
            return i

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

# 5x5 receptive field for 4 3x3 glcm window - assumed all symmetrical squares
# glcm haralick features in each 3x3 glcm window


#directory = '/content/drive/MyDrive/Outex_TC_00013/Outex_TC_00013/images'
directory = '../Texture/texure_2/Outex_TC_00013/Outex_TC_00013/images'
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
for file in files:
  img = Image.open(directory + '/' + file)
  img = img.convert("RGB") # for 3 channel PNG 
  #imggray=img.convert('LA') # convert to gray scale 

  imgWidth, imgHeight = img.size
  print(file, img.size) #128x128

  stride=height-2
  #glcm features of image array
  size=int(np.ceil((imgHeight-height+1)/stride)) #size =42
  #print(size) #42
  g=np.zeros([10, size-1,size-1]) #size-1 = 41 .. 0 - 40

  #for each image, extract window 
  i,j=0,0
  for h in range(0, imgHeight-height, stride):
      i=0
      for w in range(0, imgWidth-width, stride):
          
          box = (w, h, w+width, h+height)
          rfImage = np.array(img.crop(box)).astype(int)
        
          #extract glcm here from 5x5 window
          glcm=np.zeros([16,16])
          
          for k in range(1,4,2): #returns the glcm center (1,1), (1,3)
              for l in range(1,4,2): # (1,1), (1,3)
                  glcm[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k-1,l-1][0])]+=1
                  glcm[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k-1,l][0])]+=1
                  glcm[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k-1,l+1][0])]+=1
                  glcm[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k,l-1][0])]+=1
                  glcm[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k,l+1][0])]+=1
                  glcm[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k+1,l-1][0])]+=1
                  glcm[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k+1,l][0])]+=1
                  glcm[get_16quant(rfImage[k,l][1]),get_16quant(rfImage[k+1,l-1][0])]+=1
          glcm=glcm/32 #total 32 pixels pairs in 5x5 window
          #print(glcm)
          #features = contrast, idm, entropy, variance, correlation, second angular moment, largest
          fea=glcm_features(glcm)
          #print(fea) 
          
          g[0,i,j]=fea[0]
          g[1,i,j]=fea[1]
          g[2,i,j]=fea[2]
          g[3,i,j]=fea[3]
          g[4,i,j]=fea[4]
          g[5,i,j]=fea[5]
          g[6,i,j]=fea[6]
          i+=1
      j+=1       
  images.append(g)
  labels.append(getClass(file))
  del glcm
  del g
  del rfImage


images = np.array(images)
labels = np.array(labels)
print(images.shape)
print(labels.shape)
with open('glcm-ii.npy','wb') as f:
    np.save(f,images)
    np.save(f,labels)