#import scipy.signal

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
slider_max = 100
cap = None
numberOfQuantizionLevels = 0
freq = {}
numberOfSamplesPerColumn = 10
sample_rate = 8000
VOL = 10





def initialize():
	global width, height, cap_rate, numberOfQuantizionLevels, freq
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
	slider_max = 100


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
	
	
def start_play_thread(btn):
	app.thread(play_video)


def play_video():	
	global file, PAUSE, STOP, SLIDE, VOL
	if file == None:
		return
	STOP = False
	PAUSE =False	
	SLIDE = False
	VOL = VOL/10
	
	cap = cv2.VideoCapture(file.name)
	totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	position = app.getScale("Slider")
	#print("position of slider on play %s" % (position))
	current_frame = int((position * totalFrameCount)/(slider_max - slider_min))
	#print("Current frame %s" % (current_frame))
	cap.set(1, current_frame)
	frame = np.zeros(shape = (width, height))
	framePerSecond = cap.get(cv2.CAP_PROP_FPS)
	#counter = 30
	sounded_frame =  np.zeros(shape = (width, height))
	play_position = 1

	#if(cap.read()):  # decode successfully
	while(True):
		VOL = app.getScale("Volume")/10

		print(SLIDE)
		if STOP == True:
			position = 0
			app.setScale("Slider", position, callFunction= True)
			cap.release()

			return
		if PAUSE == True:
			cap.release()
			return
		#app.queueFunction(app.setScale("Slider", position, callFunction=True))
		if SLIDE == True:
			position = app.getScale("Slider")
			current_frame = int((position * totalFrameCount)/(slider_max - slider_min))
			cap.set(1, current_frame + 1)
	
		ret, frame = cap.read()
		
		if(ret):
			
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			img = Image.fromarray(gray)
			img = ImageTk.PhotoImage(img)
			app.reloadImageData("Video", img, fmt="mpg")

			global cap_rate
			#currentFrameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
			if play_position == 1:
				played_sound = gray
				played_sound = prepare_frame(gray)
			play_col(played_sound, play_position)
			if play_position < 30:
				play_position += 1
			else:
				play_position = 1

			if SLIDE == True:
				position = app.getScale("Slider")
				current_frame = int((position * totalFrameCount)/(slider_max - slider_min))
				cap.set(1, current_frame)
				SLIDE = False

			currentFrameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
			pos = ((currentFrameNumber / totalFrameCount) * (slider_max - slider_min))
			if SLIDE == False:
				app.setScale("Slider", pos, callFunction= False)
		
		else:   #reach the end of the video
			#cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			app.setScale("Slider", 0, callFunction= False)
			print("No more ret")
			#cap.release()
			break



		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
	app.setScale("Slider", 0, callFunction=False)
	#cap.wszx()
	cap.release()
	#cv2.destroyAllWindows()


def prepare_frame(img):
	resized = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)	
	roundedImg =  np.zeros(shape = (resized.shape[0], resized.shape[1]))
	for i in range(resized.shape[0]):
		for j in range(resized.shape[1]):
			roundedImg[i,j] = mth.floor(resized[i,j]/numberOfQuantizionLevels)/numberOfQuantizionLevels
	return roundedImg

def play_col(img, position):
	print("In play_col")
	global sample_rate, freq, VOL
	current_col = img[:, position*2]
	#print(current_col)
	chord = sine_wave(1, current_col[0], sample_rate)
	
	for c in range(height):
		chord = sum([chord, sine_wave(c, current_col[c]*127*VOL, sample_rate)])
	print(chord)
	print("Play sound")
	play_sound(chord, 33)

 

def slide_video(value):
	global file, width, height, SLIDE
	if file == None:
		return
	cap = cv2.VideoCapture(file.name)
	position = app.getScale("Slider")
	print(position)
	totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("Total frame count: %s" % totalFrameCount)
	current_frame = int((position * totalFrameCount)/(slider_max - slider_min))
	cap.set(1, current_frame)
	currentFrameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
	print("current: %s" % currentFrameNumber)
	frame = np.zeros(shape = (width, height))
	ret, frame = cap.read()
	print(ret)
	print(frame)
	if(ret):
		print("slide ret?")
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img = Image.fromarray(img)
		img = ImageTk.PhotoImage(img)
		app.reloadImageData("Video", img, fmt="mpg")
	SLIDE = True
	#cap.release()
	return

def show_options(btn):
	return

def exit_program(btn):
	return

def pause_video(btn):
	global PAUSE
	PAUSE = True
	return

def stop_playing(btn):
	global STOP 
	STOP = True 
	return

def sine_wave(row, peak, n_samples=sample_rate):
	global height, freq, numberOfSamplesPerColumn
	pos = height - row -1
	hz = freq[pos]
	#print("hz: %s" % (hz))
	length = sample_rate / float(hz)
	omega = np.pi * 2 / length
	#omega = np.sin(2 * np.pi * freq[pos] * )
	xvalues = np.arange(int(length)) * omega
	onecycle = 4 * peak * np.sin(xvalues)
	return np.resize(onecycle, (n_samples,)).astype(np.int16)


def play_sound(wave, ms):
    """Play the given NumPy array, as a sound, for ms milliseconds."""
    sound = pygame.sndarray.make_sound(wave)
    sound.play(-1)
    pygame.time.delay(ms)
    sound.stop()

def make_chord(voices):
    """Make a chord from an array."""
    sampling = 4096
    chord = waveform(voice[v], sampling)
    for v in len(voices):
        chord = sum(sine_wave(freq[v], voices[v]*64, sample_rate))
    return chord




w = 600
h = 360
frame = "blind.gif"
app = gui("Blind Helper")

pygame.mixer.init(channels = 1)
app.setPadding([5,5]) # 20 pixels padding outside the widget [X, Y]
app.setInPadding([5,5]) # 20 pixels padding inside the widget [X, Y]

sample_rate = 8000
#wave = sine_wave(880, 10000, sample_rate)
#play_sound(wave, 10000)

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
app.addButton("Play", start_play_thread, 6, 6, 2)
app.addButton("Stop", stop_playing, 6, 8, 2)
app.addLabelScale("Volume", 6, 10)
app.setScaleVertical("Volume")
app.setScaleRange("Volume", 5, 20)
app.setScale("Volume", 10)
app.setScaleIncrement("Volume", 1)
app.stopLabelFrame()





initialize()
app.go()


	