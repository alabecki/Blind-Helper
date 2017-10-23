
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

#import scipy.signal

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

'''def create_pitch_match(height):
	pitches = []
	middle = 440
	lowerst = 0

	step = math.sin(2 * path.pi * freq[m] * (double)time/sampleRate);'''



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
		#print("Any read?")
		ret, frame = cap.read()
		if(ret):
		#	print("Any ret?")
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
	
	global file
	position = app.getScale("Slider")
	app.queueFunction(app.setScale("Slider", position, callFunction=True))
	#app.setButtonChangeFunction(title, function)

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
	
		ret, frame = cap.read()
		currentFrameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
		#if currentFrameNumber % 2 == 0:
		#	continue
		if(ret):
			#print("Any ret?")
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


			pos = ((currentFrameNumber / totalFrameCount) * (slider_max - slider_min))
			app.setScale("Slider", pos, callFunction=False)
		else:   #reach the end of the video
			#cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			app.setScale("Slider", 0, callFunction=False)
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
	global sample_rate, freq
	current_col = img[:, position*2]
	#print(current_col)
	chord = sine_wave(1, current_col[0], sample_rate)
	
	for c in range(height):
		chord = sum([chord, sine_wave(c, current_col[c]*127, sample_rate)])
	print(chord)
	print("Play sound")
	play_sound(chord, 33)

 

def show_options(btn):
	return

def slide_video(value):
	global file, width, height
	print("file name: %s" % (file.name))
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
	cap.release()
	return

def show_options(btn):
	return

def exit_program(btn):
	return

def pause_video(btn):
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
frame = "test.gif"
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
app.stopLabelFrame()
video = "test.mp4"
app.setStretch("both")

app.startFrame("sidesd")
app.addScale("Slidedr")
app.setScaleIncrement("Slidedr", 1)
app.stopFrame()


initialize()
app.go()


	
