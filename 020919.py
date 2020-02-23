import numpy as np
import cv2

image_color= cv2.imread("check1.jpg")
image_ori = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)
# cv2.namedWindow("output1", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
# imS = cv2.resize(image_color, (960, 540))
image_color= cv2.imread("check1.jpg",0)


cv2.namedWindow("output1", cv2.WINDOW_NORMAL)                 # Resize image
cv2.imshow("output1", image_color)
cv2.waitKey(0)
lower_bound = np.array([0,0,10])
upper_bound = np.array([255,255,195])

image = image_color

mask = cv2.inRange(image_color, lower_bound, upper_bound)

# mask = cv2.adaptiveThreshold(image_ori,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY_INV,33,2)

kernel = np.ones((3, 3), np.uint8)

#Use erosion and dilation combination to eliminate false positives. 
#In this case the text Q0X could be identified as circles but it is not.
mask = cv2.erode(mask, kernel, iterations=6)
mask = cv2.dilate(mask, kernel, iterations=3)

closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[0]
contours.sort(key=lambda x:cv2.boundingRect(x)[0])

array = []
ii = 1
print (len(contours))
for c in contours:
    (x,y),r = cv2.minEnclosingCircle(c)
    center = (int(x),int(y))
    r = int(r)
    if r >= 6 and r<=10:
        cv2.circle(image,center,r,(0,255,0),2)
        array.append(center)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
# imS = cv2.resize(image_color, (960, 540))                    # Resize image
cv2.imshow("output", image_color)
cv2.waitKey(0)