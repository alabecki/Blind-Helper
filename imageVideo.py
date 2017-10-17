
import numpy as np
from PIL import Image, ImageTk
import cv2
#import vlc
from appJar import gui
import winsound
import threading
from threading import Thread
import time
import math as mth
import pygame, pygame.sndarray 

file = None
width = 64
height = 64

threadLock = threading.Lock()
slider_min = 0
slider_max = 200
cap = None
numberOfQuantizionLevels = 0

sample_rate = 8000


class FrameGrabber(threading.Thread):
	def __init__(self, threadID, name, counter):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.counter = counter
	def run(self):
		print ("Starting " + self.name)
		framegrabber(self)
		print ("Exiting " + self.name)

	
def framegrabber(cap):
	
	#framePerSecond = cap.get(cv2.CAP_PROP_FPS)
	#thread1 = FrameGrabber(1, "frameGrabber", 1)
	#thread1.start()
	frame = np.zeros(shape = (width, height))
	#threadLock.acquire()
	framePerSecond = cap.get(cv2.CAP_PROP_FPS)

	if(cap.read()):  # decode successfully
		print("Any read?")
		ret, frame = cap.read()
		if(ret):
			print("Any ret?")
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			img = Image.fromarray(img)
			img = ImageTk.PhotoImage(img)
			app.reloadImageData("Video", img, fmt="mpg")

			#im = Utilities.mat2Image(frame);
			#Utilities.onFXThread(imageView.imageProperty(), im);
			currentFrameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
			totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			pos = (currentFrameNumber / totalFrameCount * (slider_max - slider_min))
			app.setScale("Slider", pos, callFunction=False)
		else:   #reach the end of the video
			#cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			app.setScale("Slider", 0, callFunction=False)
			print("No more ret")



		#threadLock.release()
		threading.Timer(framePerSecond, framegrabber, [str(next)]).start()
	 
#	else:   #reach the end of the video
	#	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
	#threadLock.release()
	#threading.Timer(framePerSecond, framegrabber, [str(next)]).start()



def initialize():
	global width, height, cap_rate, numberOfQuantizionLevels
	width = 64
	height = 64
	sampleRate = 8000
	sampleSizeInBits = 8
	numberOfChannels = 1
	numberOfQuantizionLevels = 16
	numberOfSamplesPerColumn = 500
	freq = {}
	freq[(height/2) - 1] = 440.0
	m = height/2
	cap_rate = 1
	while m < height:
		freq[m] = freq[m-1]* 2**(1.0/12.0)
		m += 1
	m = (height/2)-2
	while m >= 0: 
		freq[m] = freq[m+1] * 2**(-1.0/12.0) 
		m -= 1
	slider_min = 0
	slider_max = 200


def open_video(btn):

	
	global file
	
	frame = np.zeros(shape = (width, height))

	file = app.openBox(title= "Choose a Video", dirName=None, fileTypes=None, asFile=True, parent=None)
	cap = cv2.VideoCapture(file.name)
	ret, frame = cap.read()
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)
	app.reloadImageData("Video", img, fmt="mpg")
	cap.release()

	return
		

def play_video(btn):
	#global file, cap
	#print(type(file))
	sample_rate = 44100
	wave = sine_wave(440, 3000, sample_rate)
	play_sound(wave, 10000)
	cap = cv2.VideoCapture(file.name)	
	frame = np.zeros(shape = (width, height))
	framePerSecond = cap.get(cv2.CAP_PROP_FPS)

	#if(cap.read()):  # decode successfully
	while(True):
		#print("Any read?")
		ret, frame = cap.read()
		if(ret):
			print("Any ret?")
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			img = Image.fromarray(gray)
			img = ImageTk.PhotoImage(img)
			app.reloadImageData("Video", img, fmt="mpg")

			#im = Utilities.mat2Image(frame);
			#Utilities.onFXThread(imageView.imageProperty(), im);
			global cap_rate
			currentFrameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
			if (currentFrameNumber % int(framePerSecond * cap_rate) == 0) or currentFrameNumber == 1:

				play_sounds(gray)

			totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			pos = (currentFrameNumber / totalFrameCount * (slider_max - slider_min))
			app.setScale("Slider", pos, callFunction=False)
		else:   #reach the end of the video
			#cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			app.setScale("Slider", 0, callFunction=False)
			print("No more ret")
			break



		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
	app.setScale("Slider", 0, callFunction=False)
	#cap.wszx()
	cap.release()
	#cv2.destroyAllWindows()


def play_sounds(img):
	resized = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)

	print (resized.shape)
	print (resized.shape[0])
	
	roundedImg =  np.zeros(shape = (resized.shape[0], resized.shape[1]))
	print (roundedImg.shape)
	for i in range(resized.shape[0]):
		for j in range(resized.shape[1]):
			roundedImg[i,j] = (mth.floor(resized[i,j])/numberOfQuantizionLevels)/numberOfQuantizionLevels







def show_options(btn):
	return

def slide_video(btn):
	position = app.getScale("Slider")
	cap = cv2.VideoCapture(file.name)
	cap.set(cv2.CAP_PROP_POS_FRAMES, position)
	ret, frame = cap.read()
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img = Image.fromarray(img)
	img = ImageTk.PhotoImage(img)
	app.reloadImageData("Video", img, fmt="mpg")

	cap.release()
	return

def show_options(btn):
	return

def exit_program(btn):
	return

def pause_video(btn):
	return



def sine_wave(hz, peak, n_samples=sample_rate):
    length = sample_rate / float(hz)
    omega = np.pi * 2 / length
    xvalues = np.arange(int(length)) * omega
    onecycle = peak * np.sin(xvalues)
    return np.resize(onecycle, (n_samples,)).astype(np.int16)


def play_sound(wave, ms):
    """Play the given NumPy array, as a sound, for ms milliseconds."""
    sound = pygame.sndarray.make_sound(wave)
    sound.play(-1)
    pygame.time.delay(ms)
    sound.stop()




w = 600
h = 360
frame = "test.gif"
app = gui("Blind Helper")

pygame.mixer.init(channels = 1)
app.setPadding([5,5]) # 20 pixels padding outside the widget [X, Y]
app.setInPadding([5,5]) # 20 pixels padding inside the widget [X, Y]

sample_rate = 44100
wave = sine_wave(440, 3000, sample_rate)
play_sound(wave, 10000)

app.createMenu("Menu")
app.addMenuItem("Menu", "Open Video", func = open_video, shortcut=None, underline=-1)
app.addMenuItem("Menu", "Options", func = show_options, shortcut = None, underline = -1)
app.addMenuItem("Menu", "Exit", func = exit_program, shortcut = None, underline = -1)
#app.useTtk()#Set up GUI
app.setStretch("both")
app.addScale("Slider")
app.setScaleIncrement("Slider", 1)
app.setScaleRange("Slider", 0, 100)
app.setScaleChangeFunction("Slider", slide_video)


app.startLabelFrame("Player")
app.setFont(20)
#app.addLabel("Video", "Video",1, 1 , 4, 3)
app.addImage("Video", frame)
#app.setImageLocation()
app.setImageSize("Video", w, h)
app.setStretch("both")


app.stopLabelFrame()
app.startLabelFrame("Controls")
app.setStretch("both")
app.startLabelFrame("")
#app.addScale("Slider")
#app.setScaleIncrement("Slider", 1)
app.stopLabelFrame()
app.addButton("Open", open_video, 6, 2, 2)
app.addButton("Pause", pause_video, 6, 4, 2)
app.addButton("Play", play_video, 6, 6, 2)
app.stopLabelFrame()
video = "test.mp4"
app.setStretch("both")

app.startFrame("sidesd")
app.addScale("Slidedr")
app.setScaleIncrement("Slidedr", 1)
app.stopFrame()


initialize()
app.go()


	
