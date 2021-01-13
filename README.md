# IP-Lab


1. Develop a program to display gray scale image using read and write operation.

Description: Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.

Binary images are images whose pixels have only two possible intensity values. ... Binary images are often produced by thresholding a grayscale or color image, in order to separate an object in the image from the background. The color of the object (usually white) is referred to as the foreground color. to read an image we use the function cv2.imread(). to save a image we use cv2.imwrite(). to destroy all the windows(). 

program:


import cv2

image = cv2.imread ('flower.jpg')

cv2.imshow ('Original', image)

cv2.waitKey ()

gray_image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('vijay.jpg',gray_image)

cv2.imshow ('Grayscale', gray_image)

cv2.waitKey (0)

cv2.destroyAllWindows ()


Output:

![image](https://user-images.githubusercontent.com/72337128/104428557-71f8e080-55a6-11eb-9f6f-4048f8e741c3.png)

![image](https://user-images.githubusercontent.com/72337128/104428727-9d7bcb00-55a6-11eb-8ede-c3d172297c8b.png)






2. Develop a program to perform linear transformation on an image.

Description:
A linear transformation is a function from one vector space to another that respects the underlying structure of each vector space.
Image scaling refers to the resizing of a digital image.When scaling a vector graphic image, the graphic primitives that make up the image can be scaled using geometric transformations, with no loss of image quality.Rotation is a process of changing the angle of the object. Rotation can be clockwise or anticlockwise.

Program:

a).Scaling

import cv2 as c

import numpy as np

image = c.imread('flow.jpg')

gray = c.cvtColor(image,c.COLOR_BGR2RGB)

h,w = image.shape[0:2]

width = int(w * .5)

height = int(h *.5)

res = c.resize(image,(width,height))

c.imshow('First Lab',res)

c.waitKey(0)

c.destroyAllWindows()

Output:


![image](https://user-images.githubusercontent.com/72337128/104430582-b1283100-55a8-11eb-97ad-0ece225780d2.png)

import cv2 as c

import numpy as np

image = c.imread("flow.jpg")

gray = c.cvtColor(image,c.COLOR_BGR2RGB)

h,w = image.shape[0:2]

width = int(w * 2)

height = int(h *.5)

res = c.resize(image,(width,height))

c.imshow("First Lab",res)

c.waitKey(0)

c.destroyAllWindows()

Output:

![image](https://user-images.githubusercontent.com/72337128/104430924-209e2080-55a9-11eb-9fb1-09282565d634.png)



b).Rotation

import cv2 as c

import numpy as np

image = c.imread("flow.jpg")

gray = c.cvtColor(image,c.COLOR_BGR2RGB)

h,w = image.shape[0:2]

rotationMatrix = c.getRotationMatrix2D((w/2, h/2), 200, .5)

rotated_image = c.warpAffine(image,rotationMatrix,(w,h))

c.imshow("First Lab",rotated_image)

c.waitKey(0)

c.destroyAllWindows()

Output:

![image](https://user-images.githubusercontent.com/72337128/104431471-bf2a8180-55a9-11eb-8ed9-69f3e4c85ffe.png)




3. Develop a program to find the sum and mean of a set of images

Description:




Program:

import cv2

import os

path = 'D:\Pictures'

imgs = []

files = os.listdir(path)

for file in files:

    filepath=path+"\\"+file
    
    imgs.append(cv2.imread(filepath))
    
i=0

im = []

for im in imgs:

    #cv2.imshow(files[i],imgs[i])
    
    im+=imgs[i]
    
    i=i+1
    
cv2.imshow("sum of four pictures",im)

meanImg = im/len(files)

cv2.imshow("mean of four pictures",meanImg)

cv2.waitKey(0)

Output:

![image](https://user-images.githubusercontent.com/72337128/104435093-cce20600-55ad-11eb-9ab4-f14420d60906.png)

![image](https://user-images.githubusercontent.com/72337128/104435243-f7cc5a00-55ad-11eb-82f7-c012078ea92e.png)



4. Convert color image gray scale to binary image

Description:

Program:

import cv2

img = cv2.imread('cat.jpg')

cv2.imshow('Input',img)

cv2.waitKey(0)

grayimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscaleimage',grayimg)

cv2.waitKey(0)

ret, bw_img = cv2.threshold(img,127,255, cv2.THRESH_BINARY)

cv2.imshow("Binary Image",bw_img)

cv2.waitKey(0)

cv2.destroyAllWindows()


Output:

*Normal Image*

![image](https://user-images.githubusercontent.com/72337128/104435811-a375aa00-55ae-11eb-9b79-ab69221a17c9.png)

*Gray scale image*

![image](https://user-images.githubusercontent.com/72337128/104436022-e9327280-55ae-11eb-926e-6de65cae8301.png)

*Binary image*

![image](https://user-images.githubusercontent.com/72337128/104436152-12530300-55af-11eb-87d0-4bd441d682f4.png)



5.Develop a program to convert given color image to different color space.

Description:

Program:

import cv2

image=cv2.imread('cat1.jpg')

cv2.imshow('pic',image)

cv2.waitKey(0)

yuv_img = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)

cv2.imshow('ychannel',yuv_img[:,:,0])

cv2.imshow('uchannel',yuv_img[:,:,1])

cv2.imshow('vchannel',yuv_img[:,:,2])

cv2.waitKey(0)

hsv_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

cv2.imshow('hchannel',hsv_img[:,:,0])

cv2.imshow('schannel',hsv_img[:,:,1])

cv2.imshow('vchannel',hsv_img[:,:,2])

cv2.waitKey(0)

cv2.destroyAllWindows()


Output:

![image](https://user-images.githubusercontent.com/72337128/104436910-fe5bd100-55af-11eb-9f51-77c3a9631a4b.png)

![image](https://user-images.githubusercontent.com/72337128/104437261-5e527780-55b0-11eb-9128-887f9fe692b1.png)

