import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    #Define the region of interest in a frame.
    #vertices define the region of interest
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    #apply the mask to the image
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

# = cv2.imread('road.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def process(image):
    #processan image to determine the lane markings
    #qprint(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    #define the region of interest
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/1.6),
        (width, height)
    ]
    #convert the image to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Perform a canny transformation
    canny_image = cv2.Canny(gray_image, 110, 120)
    cv2.imshow('Canny',canny_image)
    #Mask the image except for the region of interest
   
    
    masked_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    
    inputImageHough = masked_image
   
    #Use a hugh transform to find the edges
    lines = cv2.HoughLinesP(inputImageHough,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    #DRaw lines based on what what found in the hough transform
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

def processVideo(cap,process):
    while cap.isOpened():
        ret, frame = cap.read()
        try:
            frame = process(frame)
            cv2.imshow('frame', frame)
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 
                break
        except:
            
            break
def processFrame(cap,process):
    count =0
    
    while cap.isOpened():#
        count = count+1
        ret, frame = cap.read()
        frame = process(frame)
        
        cv2.imshow('frame', frame)
        if count == 3:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
#Open the test video
cap = cv2.VideoCapture('Test1.mp4')


#processVideo(cap,process)
processFrame(cap,process)
cap.release()

#cv2.destroyAllWindows()
