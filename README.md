# IP-Lab


# 1. Develop a program to display gray scale image using read and write operation.

Description: Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.

Binary images are images whose pixels have only two possible intensity values. ... Binary images are often produced by thresholding a grayscale or color image, in order to separate an object in the image from the background. The color of the object (usually white) is referred to as the foreground color. to read an image we use the function cv2.imread(). to save a image we use cv2.imwrite(). to destroy all the windows(). 

Program:


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






# 2. Develop a program to perform linear transformation on an image.

Description:
A linear transformation is a function from one vector space to another that respects the underlying structure of each vector space.
Image scaling refers to the resizing of a digital image.When scaling a vector graphic image, the graphic primitives that make up the image can be scaled using geometric transformations, with no loss of image quality.Rotation is a process of changing the angle of the object. Rotation can be clockwise or anticlockwise.

Program:

# a).Scaling

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



# b).Rotation

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




# 3. Develop a program to find the sum and mean of a set of images


Description:

Mean is most basic of all statistical measure. Means are often used in geometry and analysis; a wide range of means have been developed for these purposes. In contest of image processing filtering using mean is classified as spatial filtering and used for noise reduction.



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



# 4. Convert color image gray scale to binary image

Description:

A gray scale image has a certain number (probably 8) bits of information per pixel, hence, 256 possible grey values. Of course, a grey scale image has a binary representation, but the smallest size of information is not a bit, so we don't call it a binary image.
A binary image is the type of image where each pixel is black or white. You can also say the pixels as 0 or 1 value. Here 0 represents black and 1 represents a white pixel.

Approach:

1.Read the image from the location.
2.As a colored image has RGB layers in it and is more complex, convert it to its Grayscale form first.
3.Set up a Threshold mark, pixels above the given mark will turn white, and below the mark will turn black.

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



# 5.Develop a program to convert given color image to different color space.

Description:
 A color space is actually a combination of two things: a color model and a mapping function. The reason we want color models is because it helps us in representing pixel values using tuples. The mapping function maps the color model to the set of all possible colors that can be represented.
There are many different color spaces that are useful. Some of the more popular color spaces are RGB, YUV, HSV. Different color spaces provide different advantages.


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

![image](https://user-images.githubusercontent.com/72337128/104437454-95288d80-55b0-11eb-8d55-47a5cfcf06eb.png)



# 6. DEVELOP A PROGRAM TO CREATE AN ARRAY FROM 2D ARRAY

Description:
Two dimensional array is an array within an array. It is an array of arrays. In this type of array the position of an data element is referred by two indices instead of one. So it represents a table with rows an dcolumns of data. In the below example of a two dimensional array, observer that each array element itself is also an array.

Program:

import numpy,cv2 

img=numpy.zeros([200,200,3])

img[:,:,0]=numpy.ones([200,200])*255

img[:,:,1]=numpy.ones([200,200])*255 

img[:,:,2]=numpy.ones([200,200])*0

cv2.imwrite('flow1.jpg',img)

cv2.imshow('Color image',img)

cv2.waitKey(0) 

cv2.destroyAllWindows()

Output:

![image](https://user-images.githubusercontent.com/72337128/105331316-22916080-5bf5-11eb-8237-97de676847af.png)


# 7. Find the neighborhood matrix.

Description:

A pixel's neighborhood is some set of pixels, defined by their locations relative to that pixel, which is called the center pixel. The neighborhood is a rectangular block, and as you move from one element to the next in an image matrix, the neighborhood block slides in the same direction.

Program:

import numpy as np

axis = 3

x =np.empty((axis,axis))

y = np.empty((axis+2,axis+2))

s =np.empty((axis,axis))

x = np.array([[1,4,3],[2,8,5],[3,4,6]])


print('matrix\n')

for i in range(0,axis):

    for j in range(0,axis):
    
        print(int(x[i][j]),end = '\t')
        
    print('\n')

print('Temp matrix\n')

for i in range(0,axis+2):

    for j in range(0,axis+2):
    
        if i == 0 or i == axis+1 or j == 0 or j==axis+1:
        
            y[i][j]=0
            
        else:
        
            #print("i = {}, J = {}".format(i,j))
            
            y[i][j]=x[i-1][j-1]
           
for i in range(0,axis+2):

    for j in range(0,axis+2):
    
        print(int(y[i][j]),end = '\t')
        
    print('\n')

Output:

![image](https://user-images.githubusercontent.com/72337128/104445093-8646d880-55ba-11eb-94d5-ba46ef557fd6.png)


# 8.Calculate the Neighborhood of Matrix.

Description:

Given a M x N matrix, find sum of all K x K sub-matrix 2. Given a M x N matrix and a cell (i, j), find sum of all elements of the matrix in constant time except the elements present at row i & column j of the matrix. Given a M x N matrix, calculate maximum sum submatrix of size k x k in a given M x N matrix in O (M*N) time. Here, 0 < k < M, N.

Program:

import numpy as np

axis = 3  

x =np.empty((axis,axis))

y = np.empty((axis+2,axis+2))

r=np.empty((axis,axis))

s =np.empty((axis,axis))

x = np.array([[1,4,3],[2,8,5],[3,4,6]])


print('Matrix\n')

for i in range(0,axis):

    for j in range(0,axis):
    
        print(int(x[i][j]),end = '\t')
        
    print('\n')

print('Temp matrix\n')

for i in range(0,axis+2):

    for j in range(0,axis+2):
    
        if i == 0 or i == axis+1 or j == 0 or j==axis+1:
        
            y[i][j]=0
            
        else:
        
            #print("i = {}, J = {}".format(i,j))
            
            y[i][j]=x[i-1][j-1]
           

for i in range(0,axis+2):

    for j in range(0,axis+2):
    
        print(int(y[i][j]),end = '\t')
        
    print('\n')
   
   
print('Output calculated Neighbours of matrix\n')

print('sum of Neighbours of matrix\n')

for i in range(0,axis):

    for j in range(0,axis):
         
        r[i][j]=((y[i][j]+y[i][j+1]+y[i][j+2]+y[i+1][j]+y[i+1][j+2]+y[i+2][j]+y[i+2][j+1]+y[i+2][j+2])
        
        print(r[i][j],end = '\t')
       
    print('\n')

print('\n Average of Neighbours of matrix\n')

for i in range(0,axis):

    for j in range(0,axis):
       
        s[i][j]=((y[i][j]+y[i][j+1]+y[i][j+2]+y[i+1][j]+y[i+1][j+2]+y[i+2][j]+y[i+2][j+1]+y[i+2][j+2])/8)
       
        print(s[i][j],end = '\t')
        
    print('\n')
   

Output:

![image](https://user-images.githubusercontent.com/72337128/104445676-36b4dc80-55bb-11eb-8364-f4e9c324ef4a.png)

![image](https://user-images.githubusercontent.com/72337128/104445723-47655280-55bb-11eb-9ec0-b033e1a5b691.png)



# 9.Develop a program to implement Negative Transformation of an image.

Description:

Enhancing an image provides better contrast and a more detailed image as compare to non enhanced image. Image enhancement has very applications. It is used to enhance medical images, images captured in remote sensing, images from satellite e.t.c
The second linear transformation is negative transformation, which is invert of identity transformation. In negative transformation, each value of the input image is subtracted from the L-1 and mapped onto the output image.

Program:

import cv2 

import matplotlib.pyplot as plt  
  
img_orgn = cv2.imread('cat1.jpg', 1) 
  
plt.imshow(img_orgn) 

plt.show() 

img_neg = 255 - img_orgn 
  
plt.imshow(img_neg) 

plt.show() 

Output:

![image](https://user-images.githubusercontent.com/72337128/105327489-c0366100-5bf0-11eb-9bea-43ea76aacc54.png)


# Contrast an Image

Description:

Contrast can be simply explained as the difference between maximum and minimum pixel intensity in an image.

Program:

from PIL import Image, ImageEnhance

img = Image.open("cat.jpg")

img.show()

img=ImageEnhance.Color(img)

img.enhance(2.0).show()

Output:

![image](https://user-images.githubusercontent.com/72337128/105328020-6aae8400-5bf1-11eb-9486-394f189c1a9b.png)

![image](https://user-images.githubusercontent.com/72337128/105328069-77cb7300-5bf1-11eb-9841-bddd326f8b32.png)


# Thresholding Brightness

Description:

Brightness is a relative term. It depends on your visual perception. Since brightness is a relative term, so brightness can be defined as the amount of energy output by a source of light relative to the source we are comparing it to. In some cases we can easily say that the image is bright, and in some cases, its not easy to perceive.


Program:

import cv2  

import numpy as np  

image1 = cv2.imread('cat1.jpg')  
 
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
 
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)

ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)

ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)

ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
 
cv2.imshow('Binary Threshold', thresh1)

cv2.imshow('Binary Threshold Inverted', thresh2)

cv2.imshow('Truncated Threshold', thresh3)

cv2.imshow('Set to 0', thresh4)

cv2.imshow('Set to 0 Inverted', thresh5)
    
if cv2.waitKey(0) & 0xff == 27:  

 cv2.destroyAllWindows() 
 
 Output:
 
 ![image](https://user-images.githubusercontent.com/72337128/105329213-b7df2580-5bf2-11eb-955f-f5608d8c751e.png)


# 10.Develop a program to implement Power Law Transformation.

Description:

A variety of devices used for image capture, printing and display respond according to a power law. The exponent in the power law equation is referred to as gamma. The process is used to correct these power-law response phenomena is called gamma correction.Gamma correction is important if it is displaying an image an image accurately on a computer screen is of concern. Images that are not corrected properly can look either bleached out, or what is more likely too dark.

Program:

import numpy as np

import cv2

img = cv2.imread('cat.jpg')

gamma_two_point_two = np.array(230*(img/255)**2.1,dtype='uint8')

gamma_point_four = np.array(255*(img/255)**0.1,dtype='uint8')

img3 = cv2.hconcat([gamma_two_point_two,gamma_point_four])

cv2.imshow('a2',img3)

cv2.waitKey(0)

Output:

![image](https://user-images.githubusercontent.com/72337128/105329951-8c106f80-5bf3-11eb-8ec2-39c553b21ae8.png)


# 11.Develop a program to display Histogram of an image.

Description:

A histogram is a graph. A graph that shows frequency of anything. Usually histogram have bars that represent frequency of occurring of data in the whole data set.
A Histogram has two axis the x axis and the y axis.
The x axis contains event whose frequency you have to count.
The y axis contains frequency.
The different heights of bar shows different frequency of occurrence of data.

Histogram of an image, like other histograms also shows frequency. But an image histogram, shows frequency of pixels intensity values. In an image histogram, the x axis shows the gray level intensities and the y axis shows the frequency of these intensities.

Program:

import cv2 

from matplotlib import pyplot as plt 

img = cv2.imread('flow1.jpg',0) 
  
plt.hist(img.ravel(),256,[0,256])

plt.show()

Output:

![image](https://user-images.githubusercontent.com/72337128/105335253-a6e5e280-5bf9-11eb-98b6-62332e3b2844.png)


# 12.Program to enhance image using Arithmetic and logic operations.
Description:

Program:
Output:

