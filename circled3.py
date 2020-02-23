import numpy as np
import cv2
import time

img = cv2.imread('check1.jpg', cv2.IMREAD_COLOR) 
  
########################################################################
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

retval, mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY) 
mask1=mask.copy()
kernel = np.ones((3, 3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.erode(mask, kernel, iterations=2)

mask = cv2.blur(mask, (3, 3)) 

kernel = np.ones((3, 3), np.uint8)
mask = cv2.erode(mask, kernel, iterations=2)

mask = cv2.blur(mask, (3,3))
mask = cv2.dilate(mask, kernel, iterations=4)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#######################################################################

# Defining a kernel length
kernel_length = np.array(img).shape[1]//80
 
# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
# A kernel of (3 X 3) ones.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Morphological operation to detect vertical lines from an image
mask2 = cv2.erode(mask1, verticle_kernel, iterations=1)
# mask2 = cv2.dilate(mask2, verticle_kernel, iterations=2)


# cv2.imwrite("verticle_lines.jpg",mask2)
# Morphological operation to detect horizontal lines from an image

mask3 = cv2.erode(mask1, hori_kernel, iterations=1)
# mask3 = cv2.dilate(mask3, hori_kernel, iterations=2)
cv2.imwrite("horizontal_lines.jpg",mask3)
# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
alpha = 0.5
beta = 1.0 - alpha
# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
img_final_bin = cv2.addWeighted(mask2, alpha, mask3, beta, 0.0)
img_final_bin = cv2.erode(img_final_bin, kernel, iterations=1)
(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imwrite("img_final_bin.jpg",img_final_bin)
contours,_ = cv2.findContours(img_final_bin.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
	(x,y),r = cv2.minEnclosingCircle(c)
	approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
	if len(approx)==4:
		if(cv2.contourArea(c)>500):
			cv2.drawContours(img,[c],0,(0,0,255),-1)
			print(c)
cv2.namedWindow("out", cv2.WINDOW_NORMAL)     
cv2.imshow("out", img) 
cv2.waitKey(0)

########################################################################################
contours,_ = cv2.findContours(mask.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# contours.sort(key=lambda x:cv2.boundingRect(x)[0])

for c in contours:
	(x,y),r = cv2.minEnclosingCircle(c)
	center = (int(x),int(y))
	r = int(r)
	if cv2.contourArea(c)>30 :
		cv2.circle(img,center,20,(0,255,0),-1)
		# array.append(center)
cv2.namedWindow("output1", cv2.WINDOW_NORMAL)          
cv2.imshow("output1", img) 
cv2.imwrite("output.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

