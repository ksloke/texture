Color Textures

Outex 00013 : 0-1359: train (20 per 68 class) - done; 1360 samples

Outex 00034 : 0-1359: train (20 per 68 class) - outext-00034-glcm-i21-16L.npy; 1360 samples, different illuminants
1360-2719: test; 2720-4079: test

Outex 00031 : 0-1359: train (40 per 68 class) - outext-00031-glcm-i21-16L.npy; 2720 samples, different resolution
1360-2719: test

#00030 - separate test files each with different rotation. Train on 1360 samples, validate on 340 sample
#train 0 - 
# rot 1 : 0.9088 epoch 292
# rot 2 : 0.87058824 Epoch: 161
# ROT 3 : 0.87352943 Epoch: 277
# ROT 4 :  0.86764705 Epoch: 226
# ROT 5 : 0.85882354 Epoch: 223
# rot 6 :  0.84411764 Epoch: 247
# rot 7 : 0.85588235 Epoch: 300
# rot 8 :  0.8264706 Epoch: 235

Outex 00032: 0-2759: train (40 per 68 class) [noise] 95.22% Epoch: 185
Outex 00033: 0-2759: train (40 per 68 class) [blur] 95.2  epoch: 160

MBT: require to extract subimage from 640x640 - mbt-glcm-i21-16L.npy; 154 classes (25 per class), 3850 128x128 samples - retest with 6C

Color Brodatz: require to extract subimage from 640x640; 112 classes (25 per class); 2800 128x128 samples


#00031 - 0.61 175 epoch -- incorrect file processing
#00031 max Accuracy: 0.9375 Epoch: 107
#00034 - .93 113 epoch 
# mbt x Accuracy: 0.6571429 Epoch: 169
# mbt 6C Accuracy: 0.3 around 128x128 (128 is too large, and not enough training samples)

