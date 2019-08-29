# Cursor Vision (OpenCV)

## Platform Recommendation:
* This program has been run on Windows 10.0.15063 build 15063 and Mac OS X, developed using Python version 3.5+ & OpenCV Contrib version 4.0.0.21, Other systems have not been tested, and it is advised to have caution with untested OS.

## Software Requirements
### Python Distributions:
  * Python 3.5.X
  * Python 3.6.X
  * Python 3.7.X
### Library Dependencies:
  * pip 19.0.3
  * numpy 1.16.1
  * SciPy 1.2.1
  * wxPython 4.0.4
  * pywin32 224 (for Windows® users only)
  * dlib 19.17.0
  * opencv-contrib-python 4.0.0.21
  * imutils 0.5.2
  * PyAutoGUI 0.9.42

## To Start:
### Command Shell Setup
This setup method is intended for both the Windows® and Linux operating systems. However, to avoid repetition, this setup method will be demonstrated on a Windows® system only.
* Visit the [Python Downloads](https://www.python.org/downloads/) webpage
* Download an accepted python distribution (i.e. Python 3.5+)
* Follow the setup instructions to ensure that Python is correctly installed  
**NOTE: Ensure you check the setting to add Python to your PATH directory, or you will not be able to execute python programs on the Windows® command shell.**
* Navigate to the Cursor Vision GitLab directory
* Locate the download icon (i.e. top-right corner)
* Select the format you wish to download the file as (i.e. *.zip*, *.tar.gz*, etc…)
* Locate the file in your local downloads folder
* Extract the contents of the file
* Locate the [shape_predictor_68_face_landmarks.dat](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) file on AKSHAYUBHAT's GitHub page
* Click the Download button
* Navigate to your local downloads folder
* Move the *shape_predictor_68_face_landmarks.dat* file into the same directory of the extracted cursor vision files
* Open the Windows® command shell
* Navigate to the downloads folder using the command: *cd <PATH>/ Users/ … / Downloads*
* Execute the program using the command: *python3 Cursor Vision.py*  
**NOTE: For this step to work, you must correctly connect a webcam into your PC and follow its instruction manual to install all of its up-to-date drivers.**

### Windows Installer Setup
This setup method is intended for the Windows® operating system only, as it involves the use of a *.exe* file which only functions on Windows® distributions.
* Navigate to the Setup GitLab directory
* Locate the download icon (i.e. top-right corner)
* Select the format you wish to download the file as (i.e. *.zip*, *.tar.gz*, etc…)
* Locate the file in your local downloads folder
* Extract the contents of the file
* Open the folder holding the extracted setup files
* Locate the first *setup.7z* file without any further extensions (i.e. *.002*, *.003*, etc...)
* Extract the contents of the file
* Double-click on the *setup.exe* file  
**NOTE: You must have 7-Zip installed to extract *.7z* files.**
* Follow the setup instructions to ensure that Cursor Vision is correctly installed
