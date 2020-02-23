import cv2 
import numpy as np 
  
# Read image. 
img = cv2.imread('check3.jpg', cv2.IMREAD_COLOR) 
  
# Convert to grayscale. 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

retval, mask = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY) 


#Use erosion and dilation combination to eliminate false positives. 
#In this case the text Q0X could be identified as circles but it is not.
kernel = np.ones((3, 3), np.uint8)
mask = cv2.erode(mask, kernel, iterations=2)

mask = cv2.blur(mask, (3, 3)) 

kernel = np.ones((2, 2), np.uint8)
mask = cv2.erode(mask, kernel, iterations=3)

mask = cv2.blur(mask, (3,3))
mask = cv2.dilate(mask, kernel, iterations=4)

cv2.namedWindow("output2", cv2.WINDOW_NORMAL)                 # Resize image
cv2.imshow("output2", mask) 
# Blur using 3 * 3 kernel. 

# Apply Hough transform on the blurred image. 
detected_circles = cv2.HoughCircles(mask,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 30, minRadius = 10, maxRadius = 40) 
  
# Draw circles that are detected. 

# contours,_ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# for c in contours:
#     if cv2.contourArea(c)>100:
#         M = cv2.moments(c)
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         cv2.circle(img, (cx, cy),25, (0, 255, 0), 2)



if detected_circles is not None: 
  
    # Convert the circle parameters a, b and r to integers. 
    detected_circles = np.uint16(np.around(detected_circles)) 
  
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
  
        # Draw the circumference of the circle. 
        cv2.circle(img, (a, b), r, (0, 0, 255), -1) 
  
        # Draw a small circle (of radius 1) to show the center. 
        # cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
cv2.namedWindow("output1", cv2.WINDOW_NORMAL)                 # Resize image
cv2.imshow("output1", img) 
cv2.waitKey(0) 