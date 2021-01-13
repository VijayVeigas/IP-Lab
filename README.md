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


