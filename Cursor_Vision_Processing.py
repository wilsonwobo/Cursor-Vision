# import the necessary packages
from multiprocessing import Queue, Process, Value, Lock, freeze_support
from multiprocessing.dummy import Lock as DummyLock
from multiprocessing.dummy import Value as DummyValue
from multiprocessing.dummy import Pool
from webcamvideostream_EDIT import  WebcamVideoStream
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import FPS
from copy import deepcopy
import pyautogui, sys
import numpy as np
import imutils
import time
import dlib
import math
import cv2
import sys
import wx

class Iris_Detect:
        # optimal brightness values
        gamma_face = 1.3    
        gamma_eye = 1.8
        gamma_face_grey = 0
        
        # optimal contrast values
        clahe_rgb = 0.7     
        clahe_grey = 1.8
        clahe_grey_eye = 0
        
        EYE_AR_THRESH = 0.23        # eye aspect ratio must fall below this threshold
        EYE_AR_CONSEC_FRAMES = 2    # frames must fall below this threshold
        
        COUNTER = None      # the frame counters 
        TOTAL = None        # the total number of blinks
        lock1 = DummyLock()
        
        x = None            # the X eye coordinate
        y = None            # the Y eye coordinate
        r = None            # the eye radius
        lock2 = DummyLock()
        
        switch = None      # determintes whether the queue can be used
        switch1 = None     # determintes whether the program is terminated
        lock0 = Lock()
        lock3 = Lock()
        
        queue = None
        lock = Lock()
        
        s1_mode = None
        s2_mode = None
        arr_circles = None

        fps = None
        s_to_e_ratio = None
        top_left_pos = None
        
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')       # create the facial landmark predictor
        detector = dlib.get_frontal_face_detector()                                     # initialize dlib's face detector (HOG-based)

        def __init__(self, gamma_face, gamma_eye, clahe_rgb, clahe_grey, gamma_face_grey, clahe_grey_eye):
                self.gamma_face = gamma_face
                self.gamma_eye = gamma_eye
                self.clahe_rgb = clahe_rgb
                self.clahe_grey = clahe_grey
                self.gamma_face_grey = gamma_face_grey
                self.clahe_grey_eye = clahe_grey_eye

                self.queue = Queue()

                self.fps = None
                self.s1_mode = None
                self.s2_mode = None
                self.arr_circles = None
                self.s_to_e_ratio = None
                self.top_left_pos = None

                self.TOTAL = DummyValue('i',0)        # the total number of blinks
                self.COUNTER = DummyValue('i',0)      # the frame counter

                self.switch = Value('i', 1)      # determintes whether the queue can be used
                self.switch1 = Value('i', 1)     # determintes whether the program is terminated
        
                self.x = DummyValue('d',0)            # the X eye coordinate
                self.y = DummyValue('d',0)            # the Y eye coordinate
                self.r = DummyValue('d',0)            # the eye radius

        #------------------------------------------------------------------------------------------------------------------------------------
        # FRAME MANIPULATION & ENHANCEMENT
        #------------------------------------------------------------------------------------------------------------------------------------

        def set_frame_brightness(self, frame, gamma_value):    # improves 'GRAY & RGB' frame brightness (using gamma enhancement)
                gamma = gamma_value

                # inverts the gamma value
                invGamma = 1.0 / gamma

                # creates a lookup table with the gamma values
                table = np.array([((i / 255.0) ** invGamma) * 255       
                for i in np.arange(0, 256)]).astype("uint8")

                # maps the lookup table values to the image
                frame = cv2.LUT(frame, table)   
                
                return frame    # returns the brightened image
                
        def set_RGB_frame_contrast(self, frame, clip_value):    # improves 'RGB' frame contrast
                # converts image to the LAB Color model
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

                # splits the LAB image to different channels
                l, a, b = cv2.split(lab)

                # applys CLAHE to L-channel
                clahe = cv2.createCLAHE(clipLimit=clip_value, tileGridSize=(8,8))
                cl = clahe.apply(l)

                # merges the CLAHE enhanced L-channel with the a and b channel
                limg = cv2.merge((cl,a,b))

                # converts image from LAB Color model back to the RGB model
                frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                
                return frame
                
        def set_GRAY_frame_contrast(self, frame, clip_value):    # improves 'GRAY' frame contrast
                # sets the clip value and grid size of the CLAHE function
                clahe = cv2.createCLAHE(clipLimit=clip_value, tileGridSize=(8,8))

                # applies OpenCV's CLAHE function to the frame
                frame = clahe.apply(frame)      
                
                return frame
                       
        def frame_manipulation(self, frame, mode):
                # this mode applies the brightness and contrast adjustments to the face frames
                if mode == 1:
                        if self.gamma_face != 0:
                                frame = self.set_frame_brightness(frame, self.gamma_face)
                        if self.clahe_rgb != 0:
                                frame = self.set_RGB_frame_contrast(frame, self.clahe_rgb)
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        if self.gamma_face_grey != 0:
                                gray = self.set_frame_brightness(gray, self.gamma_face_grey)
                        if self.clahe_grey != 0:
                                gray = self.set_GRAY_frame_contrast(gray, self.clahe_grey)
 
                        return frame, gray

                # this mode applies the brightness and contrast adjustments to the face frames and displays the frames        
                elif mode == 2:
                        if self.gamma_face != 0:
                                frame = self.set_frame_brightness(frame, self.gamma_face)
                        if self.clahe_rgb != 0:
                                frame = self.set_RGB_frame_contrast(frame, self.clahe_rgb)

                        frame_c = frame.copy()
                        cv2.ellipse(frame_c,(225,165),(135,100),90,0,360,(0,255,0),2)     # recommend an area to position your head 
                        cv2.imshow("RGB FACE", frame_c)
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        if self.gamma_face_grey != 0:
                                gray = self.set_frame_brightness(gray, self.gamma_face_grey)
                        if self.clahe_grey != 0:
                                gray = self.set_GRAY_frame_contrast(gray, self.clahe_grey)
                        cv2.imshow("GRAY FACE", gray)
                        
                        return frame, gray

                # this mode applies the brightness and contrast adjustments to the cropped eye frame and displays the frames     
                elif mode == 3:
                        if self.gamma_eye != 0:
                                frame = self.set_frame_brightness(frame, self.gamma_eye)
                        if self.clahe_grey_eye != 0:
                                frame = self.set_GRAY_frame_contrast(frame, self.clahe_grey_eye)
                                
                        frame = cv2.copyMakeBorder(frame,20,20,215,215,cv2.BORDER_CONSTANT)
                        cv2.imshow("GRAY EYE", frame)

                # this mode applies the brightness and contrast adjustments to the cropped eye frame        
                elif mode == 4:
                        if self.gamma_eye != 0:
                                frame = self.set_frame_brightness(frame, self.gamma_eye)
                        if self.clahe_grey_eye != 0:
                                frame = self.set_GRAY_frame_contrast(frame, self.clahe_grey_eye)
                                
                        return frame

                # this mode grabs the frame from the threaded video stream and resizes it to have a maximum width of 450 pixels
                elif mode == 5:
                        real_frame = frame.read()       # camera_stream value not actually a frame
                        real_frame = cv2.flip(real_frame,1)     # reverts the mirror efect of the frame 
                        real_frame = imutils.resize(real_frame, width=450)
                        
                        return real_frame
                                
                return 0
         
                
        #------------------------------------------------------------------------------------------------------------------------------------
        # PREPROCESSING FOR EYE TRACKING
        #------------------------------------------------------------------------------------------------------------------------------------
        def detect_eyes(self, gray, frame):
                # grab the index of the facial landmarks for the left eye
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

                # detect faces in the grayscale frame
                rects = self.detector(gray, 0)

                # loop over the face detections
                eyes = None
                for rect in rects:
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = self.predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = shape[lStart:lEnd]

                        # compute the convex hull for the left and right eye, then
                        # visualize each of the eyes
                        ex,ey,ew,eh = cv2.boundingRect(leftEye)
                        ey = ey - 4
                        eh = eh + 8

                        cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0, 255, 0), 1)
                        return (ex,ey,ew,eh), gray, frame

                return eyes, gray, frame
        
        def track_iris(self, cropped_eye_gray, cropped_eye):
                contours, heirachy = cv2.findContours(cropped_eye_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # checks if the area of the contour is within a range of 50mm - 312mm
                contours_area = []
                for con in contours:
                        area = cv2.contourArea(con)
                        if 50 < area < 312:
                                contours_area.append(con)

                # checks if the contour is of circular shape
                contours_circles = []
                for con in contours_area:
                        perimeter = cv2.arcLength(con, True)
                        area = cv2.contourArea(con)
                        if perimeter == 0:
                                break
                        
                        circularity = 4*math.pi*(area/(perimeter*perimeter))
                        if 0.7 < circularity < 1.2:
                                contours_circles.append(con)

                # draws the contour onto the frame
                for circle in contours_circles:
                        (x,y),r = cv2.minEnclosingCircle(circle)
                        
                        Ix = int(x)
                        Iy = int(y)
                        Ir = int(r)
                        cv2.circle(cropped_eye, (Ix,Iy), Ir, (0,255,0),1)
                        cv2.rectangle(cropped_eye, (Ix - 1, Iy - 1), (Ix + 1, Iy + 1), (0, 128, 255), 1)

                        return x, y, r
                        
                return 0, 0, 0
                
        def track_iris_process(self, frame, gray):
                left_eye_store, gray, frame = self.detect_eyes(gray, frame)
                
                cropped_eye_gray = None
                if left_eye_store is not None:
                        # crops-out a GRAY & an RGB version of the eye
                        cropped_eye = frame[left_eye_store[1]:left_eye_store[1]+left_eye_store[3], left_eye_store[0]:left_eye_store[0]+left_eye_store[2]]
                        cropped_eye_gray = gray[left_eye_store[1]:left_eye_store[1]+left_eye_store[3], left_eye_store[0]:left_eye_store[0]+left_eye_store[2]]

                        # calls the frame manipulation function under mode 4
                        cropped_eye_gray = self.frame_manipulation(cropped_eye_gray, 4)         
                        
                        #------------------------------------------------------------------------------------------------------------------------------------
                        ############################### SPACE FOR IRIS TRACKING IMPROVEMENT ON THE LEFT-EYE
                        #------------------------------------------------------------------------------------------------------------------------------------
                        #cropped_eye_gray = cv2.bilateralFilter(cropped_eye_gray, 4, 35, 35)

                        # blurs the camera frame to reduce noise 
                        cropped_eye_gray = cv2.GaussianBlur(cropped_eye_gray,(5,7),0)           

                        # applies adaptive thresholding to the camera frame (Image Binarizationz)
                        cropped_eye_gray = cv2.adaptiveThreshold(cropped_eye_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,2) 
                        
                        # inverts the colour of the image for easier detection
                        cropped_eye_gray = cv2.bitwise_not(cropped_eye_gray)
                        kernel = np.ones((2,2),np.uint8)

                        # applies the canny edge detector to the image
                        canny = cv2.Canny(cropped_eye_gray, 100, 200)

                        # dilates the detected edges
                        canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)

                        # removes the dilated edge
                        cropped_eye_gray[canny==255] = 0

                        # inverts the image back to normal
                        cropped_eye_gray = cv2.bitwise_not(cropped_eye_gray)
                        cropped_eye_gray = cv2.erode(cropped_eye_gray, kernel, iterations = 1)
                        #------------------------------------------------------------------------------------------------------------------------------------

                        # calls the iris detecting function 
                        x, y, r = self.track_iris(cropped_eye_gray, cropped_eye)

                        # collects the vertically flipped coordinates (Y-Axis Flipping)
                        gray_eye_flip = cv2.flip(cropped_eye_gray, 0);
                        flipX, flipY, FlipR = self.track_iris(gray_eye_flip, cropped_eye.copy())

                        # displays the thresholded eye image
                        cv2.imshow("Eye_Image", cropped_eye_gray)

                        # inserts the iris location data into the appropriate daat store
                        if x != 0 and y != 0 and r != 0:
                                with self.lock2:
                                        self.x.value = x
                                        self.y.value = flipY
                                        self.r.value = r

                                        # stores the iris data for calibiration purposes
                                        if self.s2_mode == 1:
                                                self.arr_circles.append([self.x.value,self.y.value])

                                        # stores the iris data for the cursor movement functionality
                                        elif self.s2_mode == 2:
                                                if self.switch.value == 1:
                                                        with self.lock:
                                                                self.queue.put([self.x.value,self.y.value])

                        # writes the current iris coordinates and radius on the frame
                        with self.lock2:
                                cv2.putText(frame, "X: {:.2f}".format(self.x.value), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.putText(frame, "Y: {:.2f}".format(self.y.value), (130, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.putText(frame, "R: {:.2f}".format(self.r.value), (240, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                return frame
                
        #------------------------------------------------------------------------------------------------------------------------------------
        # PREPROCESSING FOR BLINK DETECTION
        #------------------------------------------------------------------------------------------------------------------------------------        
                
        def eye_aspect_ratio(self, eye):
                # compute the euclidean distances between the two sets of
                # vertical eye landmarks (x, y)-coordinates
                A = dist.euclidean(eye[1], eye[5])
                B = dist.euclidean(eye[2], eye[4])

                # compute the euclidean distance between the horizontal
                # eye landmark (x, y)-coordinates
                C = dist.euclidean(eye[0], eye[3])

                # compute the eye aspect ratio
                ear = (A + B) / (2.0 * C)

                # return the eye aspect ratio
                return ear
                
        def detect_blink(self, frame, gray, lStart, lEnd):
                # detect faces in the grayscale frame
                rects = self.detector(gray, 0)

                # loop over the face detections
                for rect in rects:
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = self.predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = shape[lStart:lEnd]
                        leftEAR = self.eye_aspect_ratio(leftEye)

                        # average the eye aspect ratio together for both eyes
                        ear = leftEAR

                        # compute the convex hull for the left and right eye, then
                        # visualize each of the eyes
                        leftEyeHull = cv2.convexHull(leftEye)
                        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)

                        # check to see if the eye aspect ratio is below the blink
                        # threshold, and if so, increment the blink frame counter
                        if ear < self.EYE_AR_THRESH:
                                with self.lock1:
                                        self.COUNTER.value += 1

                        # otherwise, the eye aspect ratio is not below the blink
                        # threshold
                        else:
                                # if the eyes were closed for a sufficient number of frames
                                # then increment the total number of blinks
                                if self.COUNTER.value >= self.EYE_AR_CONSEC_FRAMES:
                                        with self.lock1:
                                                self.TOTAL.value += 1
                                                self.COUNTER.value = 0 # reset the eye frame counter

                                        if self.switch.value == 1:        
                                                with self.lock:
                                                        self.queue.put("BLINK") # register a blink in the multiprocessing queue
                        
                        # draw the total number of blinks on the frame along with
                        # the computed eye aspect ratio for the frame
                        with self.lock1:
                                cv2.putText(frame, "Blinks: {}".format(self.TOTAL.value), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                return frame
                
                
        #------------------------------------------------------------------------------------------------------------------------------------
        # EYE COORDINATE CALLIBIRATION STAGE
        #------------------------------------------------------------------------------------------------------------------------------------        
        def position_locator(self, number):
                contours, heirachy = cv2.findContours(number, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # checks if the area of the contour is within a range of 0mm - 1258mm
                contours_area = []
                for con in contours:
                        area = cv2.contourArea(con)
                        if 0 < area < 1258:
                                contours_area.append(con)

                # check if contour is of circular shape
                contours_circles = []
                for con in contours_area:
                        perimeter = cv2.arcLength(con, True)
                        area = cv2.contourArea(con)
                        if perimeter == 0:
                                break
                        
                        circularity = 4*math.pi*(area/(perimeter*perimeter))
                        if 0.7 < circularity < 1.2:
                                contours_circles.append(con)

                return len(contours_circles)/2
                
        def init_circle_array(self):
                # load the image, clone it for output, and then convert it to grayscale
                image = cv2.imread('eye_initialize_NEW.png')

                # create a copy of the image
                img_output = image.copy()

                # converts the colour channel of the image from RGB to Grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # applies the canny edge detection function to the image
                gray = cv2.Canny(gray,25,195)

                # detect circles in the image (0 = x, 1 = y, 2 = r)
                img_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0]/8, param1=25, param2=35, minRadius=20, maxRadius=75)

                circle_list = []

                # ensure at least some circles were found (Should be in the order [circle 4, circle 2, circle 1, circle 3, circle 5])
                if img_circles is not None:
                        # convert the (x, y) coordinates and radius of the circles to integers
                        img_circles = np.round(img_circles[0, :]).astype("int")

                        # count the number of dots within each circle to find their position
                        for (x, y, r) in img_circles:
                                # crops out each individual circle
                                number = image[(y-r):(y+r), (x-r):(x+r)]
                                number_RGB = image[(y-r):(y+r), (x-r):(x+r)]

                                # converts the colour channel of the circle from RGB to Grayscale
                                number = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)

                                # applies the canny edge detection function to the circle for easier detection
                                number = cv2.Canny(number,25,195)

                                # automacically finds the allocated number associated with the circle
                                position = self.position_locator(number)

                                # adds the coordinates and radius of the circle to a list
                                circle_list.append([position, (x, y, r)])

                #sorts the circle list by the amount of dots located within each circle
                circle_list.sort(key=lambda x: x[0])
                
                return circle_list
                
        def mean_XY(self, arr_arr_circles):
                # print out the array values
                for i in arr_arr_circles:
                        print("arr = [", end = " ")
                        
                        for (j, k) in i:
                                print("[" + str(j) + "," + str(k) + "]", end = " ")
                        print("]")
                print()
                
                # find the mean of the X & Y array values    
                mean_coordinates = []   
                for i in arr_arr_circles:
                        a = 0
                        X = 0
                        Y = 0
                        
                        for (j, k) in i:
                             X = X + j
                             Y = Y + k
                             a = a + 1
                        
                        # Prevents division by zero error, if array is empty
                        if a == 0:
                                a = 1
                        
                        X = X / a
                        Y = Y / a
                        
                        mean_coordinates.append([X,Y])
                        
                # print out the mean array values
                for (i,j) in mean_coordinates:
                        print("arr = [", end = " ")
                        print("[" + str(i) + "," + str(j) + "]", end = " ")
                        print("]")
                print()
                
                return mean_coordinates
                
        def collect_boundaries(self, camera_stream):
                circle_list = self.init_circle_array()
                overlay_location = pyautogui.size()[1]-380 # calculates the height of the user's screen for the tracking overlay frame to be placed
                arr_arr_circles = []
                self.s2_mode = 1
                p = 1

                # displays the instructions for the calibiration function in a fullscreen window
                instructions = cv2.imread('calibiration_instructions.png')
                while True:
                        cv2.namedWindow("instructions", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("instructions", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow("instructions", instructions)

                        # breaks out of the instruction loop with the "S" key
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("s"):
                                break

                # capture boundries for the 5 different circle positions
                lFlag = True
                img_output = cv2.imread('blank_canvas.png')
                while len(arr_arr_circles) < 5 and lFlag:
                        # draws an amber circle onto the blank canvas with a black dot in its center
                        img_circles = circle_list[p-1][1]
                        cv2.circle(img_output, (img_circles[0], img_circles[1]), img_circles[2], (0, 191, 255), 5)
                        cv2.circle(img_output, (img_circles[0], img_circles[1]), 3, (0, 0, 0), 10)

                        # displays the black canvas on a fullscreen window
                        cv2.namedWindow("img_output", cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty("img_output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.imshow("img_output", img_output)

                        # breaks out of the calibiration loop with the "Q" key
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                                lFlag = False
                                break

                        # give the user a 1.5sec break to focus on the new circle
                        time.sleep(1.5)
                        
                        # collect 30 pupil coordinates
                        self.arr_circles = []
                        while len(self.arr_circles) < 30:
                                # calls the frame manipulation function under mode 5 & mode 1
                                frame = self.frame_manipulation(camera_stream, 5)
                                frame, gray = self.frame_manipulation(frame, 1)
                                
                                # display the tracked iris onto the screen
                                frame = self.track_iris_process(frame, gray)
                                frame_c = frame.copy()

                                # recommends an area for the user to position their head for optimal tracking results
                                cv2.ellipse(frame_c,(225,165),(135,100),90,0,360,(0,255,0),2)     

                                # displays the tracking overlay on top of the blank canvas
                                cv2.namedWindow("Frame")
                                cv2.moveWindow("Frame", 0, overlay_location)
                                cv2.imshow("Frame", frame_c)

                                # if the `q` key was pressed, break from the loop
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord("q"):
                                        lFlag = False
                                        break

                                # update the FPS counter
                                self.fps.update()

                        # once the iris coordinates have been stored change the colour of the circle to green and remove the black dot in its center
                        cv2.circle(img_output, (img_circles[0], img_circles[1]), img_circles[2], (0, 255, 0), 5)
                        cv2.circle(img_output, (img_circles[0], img_circles[1]), 3, (255, 255, 255), 10)
                        arr_arr_circles.append(self.arr_circles)

                        # display some extra text to notify the user that the iris location has been stored
                        cv2.putText(img_output, "Initialized Location " + str(p) + "...", (10, 30*p), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 215), 2)

                        p = p + 1

                        # waits 0.5sec for the next circle to be displayed
                        time.sleep(0.5)
                
                # output the boundary values & do a bit of cleanup
                mean_coordinates = self.mean_XY(arr_arr_circles)
                cv2.destroyAllWindows()
                
                return mean_coordinates, circle_list
        
        
        #------------------------------------------------------------------------------------------------------------------------------------
        # DETECTION & TRACKING STAGE
        #------------------------------------------------------------------------------------------------------------------------------------
                
        def realtime(self, camera_stream):
                # grab the index of the facial landmarks for the left eye
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                self.s2_mode = 2
                
                # loop over the frames from the video stream
                while True:
                        # calls the frame manipulation function under mode 5 & mode 1
                        frame = self.frame_manipulation(camera_stream, 5)
                        frame, gray = self.frame_manipulation(frame, 1)

                        # runs the blink detector & the iris tracker functions sequentially under the test setup mode
                        if self.s1_mode == 1:
                                # calls the blink detector & the iris tracker functions
                                p1 = self.detect_blink(frame.copy(), gray.copy(), lStart, lEnd)
                                p2 = self.track_iris_process(frame.copy(), gray.copy())

                                # recommends an area for the user to position their head for optimal tracking results
                                cv2.ellipse(p1,(225,165),(135,100),90,0,360,(0,255,0),2)
                                cv2.ellipse(p2,(225,165),(135,100),90,0,360,(0,255,0),2)

                                # displays the blink detector & the iris tracker frames side by side in a single window
                                cv2.imshow("Blink Detection & Iris Tracking (Test)", np.hstack((p1, p2)))

                        # spawns the blink detector & the iris tracker functions into their own threads under the auto & self setup modes        
                        elif self.s1_mode == 2:
                                with Pool(4) as pool:
                                        # uses the multiprocessing dummy to spawn new threads
                                        p1 = pool.apply_async(self.detect_blink, args=(frame.copy(), gray.copy(), lStart, lEnd))
                                        p2 = pool.apply_async(self.track_iris_process, args=(frame.copy(), gray.copy()))

                                        # obtains a copy of each frame produced by the two threads
                                        frame_c1 = p1.get().copy()
                                        frame_c2 = p2.get().copy()

                                        # recommends an area for the user to position their head for optimal tracking results
                                        cv2.ellipse(frame_c1,(225,165),(135,100),90,0,360,(0,255,0),2)
                                        cv2.ellipse(frame_c2,(225,165),(135,100),90,0,360,(0,255,0),2)

                                        # displays the blink detector & the iris tracker frames side by side in a single window
                                        cv2.imshow("Blink Detection & Iris Tracking", np.hstack((frame_c1, frame_c2)))

                                        # ensures the threads are completed and closes them
                                        pool.close()
                                        pool.join()
                        
                        # breaks out of the main loop with the "Q" key
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):

                                # sends a signal to the queue listener class to stop functioning
                                with self.lock3:
                                        self.switch1.value = 0
                                break

                        # update the FPS counter
                        self.fps.update()

                # do a bit of cleanup
                cv2.destroyAllWindows()
        
        
        #------------------------------------------------------------------------------------------------------------------------------------
        # CONTRAST & BRIGHTESS ADJUSTMENT STAGE
        #------------------------------------------------------------------------------------------------------------------------------------        
        def on_trackbar_1(self, val):
                self.gamma_face_grey = cv2.getTrackbarPos('BRIGHTNESS','GRAY FACE')/10
                self.clahe_grey = cv2.getTrackbarPos('CONTRAST','GRAY FACE')/10

        def on_trackbar_2(self, val):
                self.gamma_face = cv2.getTrackbarPos('BRIGHTNESS','RGB FACE')/10
                self.clahe_rgb = cv2.getTrackbarPos('CONTRAST','RGB FACE')/10

        def on_trackbar_3(self, val):
                self.gamma_eye = cv2.getTrackbarPos('BRIGHTNESS','GRAY EYE')/10
                self.clahe_grey_eye = cv2.getTrackbarPos('CONTRAST','GRAY EYE')/10
                            
        def show_adjusted_values(self, camera_stream):
                # creates a window for the grayscale face frame
                cv2.namedWindow("GRAY FACE")

                # displays a brightness & contrast adjustment trackbar on the window
                cv2.createTrackbar("BRIGHTNESS", "GRAY FACE", 0, 20, self.on_trackbar_1)
                cv2.createTrackbar("CONTRAST", "GRAY FACE", 0, 20, self.on_trackbar_1)

                # creates a window for the rgb face frame
                cv2.namedWindow("RGB FACE")

                # displays a brightness & contrast adjustment trackbar on the window
                cv2.createTrackbar("BRIGHTNESS", "RGB FACE", 0, 20, self.on_trackbar_2)
                cv2.createTrackbar("CONTRAST", "RGB FACE", 0, 20, self.on_trackbar_2)

                # creates a window for the grayscale cropped eye frame
                cv2.namedWindow("GRAY EYE", cv2.WINDOW_NORMAL)

                # displays a brightness & contrast adjustment trackbar on the window
                cv2.createTrackbar("BRIGHTNESS", "GRAY EYE", 0, 20, self.on_trackbar_3)
                cv2.createTrackbar("CONTRAST", "GRAY EYE", 0, 20, self.on_trackbar_3)
                
                # loop over the frames from the video stream
                while True:
                        # calls the frame manipulation function under mode 5 & mode 2
                        frame = self.frame_manipulation(camera_stream, 5)                        
                        frame, gray = self.frame_manipulation(frame, 2)

                        # calls the right eye detection function
                        left_eye_store, gray, frame = self.detect_eyes(gray, frame)
                        if left_eye_store is not None:
                                # crops-out a GRAY & an RGB version of the eye
                                cropped_eye = frame[left_eye_store[1]:left_eye_store[1]+left_eye_store[3], left_eye_store[0]:left_eye_store[0]+left_eye_store[2]]
                                cropped_eye_gray = gray[left_eye_store[1]:left_eye_store[1]+left_eye_store[3], left_eye_store[0]:left_eye_store[0]+left_eye_store[2]]

                                # calls the frame manipulation function under mode 3
                                self.frame_manipulation(cropped_eye_gray, 3)
                                
                        # if the `q` key was pressed, break from the loop
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                                break

                # do a bit of cleanup
                cv2.destroyAllWindows()
        
        
        #------------------------------------------------------------------------------------------------------------------------------------
        # CAMERA CONTROL
        #------------------------------------------------------------------------------------------------------------------------------------
                
        def start_camera(self):
                # initializes the video stream and allows the camera to warmup 
                print("[INFO] starting video stream...")
                #camera_stream = WebcamVideoStream(src=0, resolution = (1080, 720), framerate = 60).start()
                camera_stream = WebcamVideoStream(src=0, resolution = (1640,1232), framerate = 60).start()
                
                self.fps = FPS().start()
                time.sleep(2.0)
                
                return camera_stream
                
        def stop_camera(self, camera_stream):
                # stops the camera video stream
                camera_stream.stop()
                
                # stop the timer and display FPS information
                self.fps.stop()
                print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
                print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
                
        
        #------------------------------------------------------------------------------------------------------------------------------------
        # QUEUE LISTENER
        #------------------------------------------------------------------------------------------------------------------------------------
        
        def queue_listener(self):
                TFlag = True
                while TFlag:
                        # checks if the termination signal has been sent by the main loop
                        with self.lock3:
                                if self.switch1.value == 0:
                                        self.switch1.value = 1
                                        TFlag = False
                                        break

                        # once the queue has 6 items begin processing for cursor movements
                        if self.queue.qsize() >= 6:
                                # stops the blink detector & the iris tracker from inserting more data into the buffer
                                with self.lock0:
                                        self.switch.value = 0

                                # if the queue is not empty begin the cursor movement process
                                if not self.queue.empty():      
                                        message = None
                                        message_list = []

                                        # removes all the data from the buffer and inserts it into a temporary list
                                        while not self.queue.empty():
                                                with self.lock:
                                                        message = self.queue.get()      # Reads the most recent value from the queue
                                                        message_list.append(message)

                                        # checks the list for any blinks (Prioritized over iris movements due to the frequent nature of iris movements)
                                        for messages in message_list:       
                                                if messages == 'BLINK':
                                                        message = 'BLINK'
                                                        break

                                        # prints data to the console and clicks on the current position of the cursor        
                                        if (message == 'BLINK'):
                                                print("BLINK DETECTED") # call the click method from pyautogui   
                                                pyautogui.click()       # click the mouse

                                        # prints data to the console and changes the location of the cursor        
                                        else:
                                                screen_X, screen_Y = self.eye_to_screen_coods(message[0], message[1])
                                                print( "EYE X: " + str(message[0]) + ", EYE Y: " + str(message[1]) + "  SCREEN X: " + str(screen_X) + ", SCREEN Y: " + str(screen_Y))
                                                
                                                pyautogui.moveTo(screen_X, screen_Y, 0.5)       # makes the movement for 0.5sec

                                # reopens the buffer for data insertsion
                                with self.lock0:
                                        self.switch.value = 1
                        
        #------------------------------------------------------------------------------------------------------------------------------------
        # CURSOR MOVEMENT CALCULATIONS 
        #------------------------------------------------------------------------------------------------------------------------------------

        def screen_to_eye_ratio(self, center_eye, top_left_eye, center_screen):
                # translates the center coordinates of the iris to the top-left corner of the screen  
                correct_center_eye_X = abs(center_eye[0] - top_left_eye[0])
                correct_center_eye_Y = abs(center_eye[1] - top_left_eye[1])

                # determines the ratio of the screen to the eyes
                ratio_X = center_screen[0] / correct_center_eye_X
                ratio_Y = center_screen[1] / correct_center_eye_Y
                
                return ratio_X, ratio_Y
                
        def eye_to_screen_coods(self, eye_X, eye_Y):
                # translates the current coordinates of the iris to the top-left corner of the screen
                correct_eye_X = abs(eye_X - self.top_left_pos[0])
                correct_eye_Y = abs(eye_Y - self.top_left_pos[1])

                # determines the gaze location of the cursor using the screen-to-eye ratio
                screen_X = correct_eye_X * self.s_to_e_ratio[0]
                screen_Y = correct_eye_Y * self.s_to_e_ratio[1]
                
                return screen_X, screen_Y
        
class Configuration:
        def auto_setup(self):
                tracker = Iris_Detect(1.3, 1.8, 0.7, 1.8, 0, 0)
                tracker.s1_mode = 2
                
                camera_stream = tracker.start_camera()

                boundary_data, circle_data = tracker.collect_boundaries(camera_stream)
                if len(boundary_data) == 5:
                        center_eye = boundary_data[0]                                   # center eye coordinates
                        top_left_eye = [boundary_data[3][0],boundary_data[1][1]]        # top-left eye coordinates (0,0)
                        bottom_right_eye = [boundary_data[4][0],boundary_data[2][1]]    # bottom-right eye coordinates (1920,1080)
                        center_screen = circle_data[0][1]                               # center screen coordinates
                        
                        print("Center Eye: " + str(center_eye))
                        print("Top-Left Eye: " + str(top_left_eye))
                        print("Bottom-Right Eye: " + str(bottom_right_eye))
                        print("Center Screen: " + str(center_screen))
                        print()
                        
                        ratio_X, ratio_Y = tracker.screen_to_eye_ratio(center_eye, top_left_eye, center_screen)
                        tracker.s_to_e_ratio = [ratio_X, ratio_Y]
                        tracker.top_left_pos = [top_left_eye[0], top_left_eye[1]]
                        
                        p = Process(target = tracker.queue_listener)
                        p.start()
                        
                        tracker.realtime(camera_stream)
                        p.join()
                        
                tracker.stop_camera(camera_stream)

        def self_setup(self):
                tracker = Iris_Detect(1.3, 1.8, 0.7, 1.8, 0, 0)
                tracker.s1_mode = 2
                
                camera_stream = tracker.start_camera()
                tracker.show_adjusted_values(camera_stream)

                boundary_data, circle_data = tracker.collect_boundaries(camera_stream)
                if len(boundary_data) == 5:
                        center_eye = boundary_data[0]                            # center eye coordinates
                        top_left_eye = [boundary_data[3][0],boundary_data[1][1]] # top-left eye coordinates (0,0)
                        bottom_right_eye = [boundary_data[4][0],boundary_data[2][1]]    # bottom-right eye coordinates (1920,1080)
                        center_screen = circle_data[0][1]                        # center screen coordinates
                        
                        print("Center Eye" + str(center_eye))
                        print("Top-Left Eye" + str(top_left_eye))
                        print("Bottom-Right Eye: " + str(bottom_right_eye))
                        print("Center Screen" + str(center_screen))
                        print()
                        
                        ratio_X, ratio_Y = tracker.screen_to_eye_ratio(center_eye, top_left_eye, center_screen)
                        tracker.s_to_e_ratio = [ratio_X, ratio_Y]
                        tracker.top_left_pos = [top_left_eye[0], top_left_eye[1]]
                        
                        p = Process(target = tracker.queue_listener)
                        p.start()
                        
                        tracker.realtime(camera_stream)
                        p.join()
                
                tracker.stop_camera(camera_stream)

        def test_setup(self):
                tracker = Iris_Detect(1.3, 1.8, 0.7, 1.8, 0, 0)
                tracker.s1_mode = 1
                
                camera_stream = tracker.start_camera()
                tracker.show_adjusted_values(camera_stream)
                tracker.realtime(camera_stream)
                tracker.stop_camera(camera_stream)

def main():
        while True:
                print("Choose Your Preferred Configuration? (Auto[A]/Self[S]/Test[T]/Exit[E])")
                
                response = input()
                if response == "A" or response == "a":
                        config = Configuration()
                        config.auto_setup()
                elif response == "S" or response == "s":
                        config = Configuration()
                        config.self_setup()
                elif response == "T" or response == "t":
                        config = Configuration()
                        config.test_setup()
                elif response == "E" or response == "e":
                        break
                else:
                        continue

if __name__ == "__main__":
        freeze_support()
        main()
