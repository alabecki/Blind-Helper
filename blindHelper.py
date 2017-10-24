#Libraries
import numpy.core._methods
import numpy.lib.format
import numpy as np
from PIL import Image, ImageTk
import cv2
from appJar import gui
import winsound
import threading
from threading import Thread
import time
import math as mth
import pygame, pygame.sndarray
import pygame.midi
import sys
pygame.midi.init()

#Global Variables (originally planned to replace with Class variables)
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

#Based on the provided Java skeleton code
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


#Opens a video file and loads the first frame to the image window
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
        
 
#Creates a thread to take care of the frame grabbing and sound player. 
#Prevents the GUI from freezing.        
def start_play_thread(btn):
        app.thread(play_video)


#Primary function used to play the video along with FM generated tones. 
def play_video():       
        global file, PAUSE, STOP, SLIDE, VOL
        if file == None:
                return
        STOP = False
        PAUSE =False    
        SLIDE = False
        VOL = VOL/10
        #Capture initiated along with some variables
        cap = cv2.VideoCapture(file.name)
        totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        position = app.getScale("Slider")
        current_frame = int((position * totalFrameCount)/(slider_max - slider_min))
        #Set the capture to agree with the Slider
        cap.set(1, current_frame)
        frame = np.zeros(shape = (width, height))
        framePerSecond = cap.get(cv2.CAP_PROP_FPS)
        #counter = 30
        sounded_frame =  np.zeros(shape = (width, height))
        play_position = 1
        #Main play loop
        while(True):
        		#sets volume based on the position of the volume slider
                VOL = app.getScale("Volume")/10
                #Checks if the STOP button has been pressed
                if STOP == True:
                        position = 0
                        app.setScale("Slider", position, callFunction= True)
                        cap.release()
                        return
                #Checks if the PAUSE button has been pressed
                if PAUSE == True:
                        cap.release()
                        return
                #Checks if the SLIDER has been been moved by the user 
                #If so, resets the capture to agree with it 
                if SLIDE == True:
                        position = app.getScale("Slider")
                        current_frame = int((position * totalFrameCount)/(slider_max - slider_min))
                        cap.set(1, current_frame + 1)
               
                ret, frame = cap.read()
                                       
                #Checks if a frame as successfully been captured
                if(ret):
                		#Plays a bell sound that (plays very fast so its difficult to hear - easier to hear click(bell in midi function)
                        app.bell()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        img = Image.fromarray(gray)
                        img = ImageTk.PhotoImage(img)
                        #Updates the image presented in the display to current frame
                        app.reloadImageData("Video", img, fmt="mpg")

                        global cap_rate
                        #Checks if the program should being playing sound for the current frame
                        #Does so if position is 1
                        if play_position == 1:
                                played_sound = gray
                                played_sound = prepare_frame(gray)
                        #Calls function to play sound
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
                
                else:
                        app.setScale("Slider", 0, callFunction= False)
                        break



                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        app.setScale("Slider", 0, callFunction=False)
        cap.release()


#Creates a thread for the midi based playback        
def start_play_thread_midi(btn):
        app.thread(play_video_midi)


#Similar to the play_video() except ends up calling functions for the MIDI based playback
def play_video_midi():
        global file, PAUSE, STOP, SLIDE, VOL
        if file == None:
                return
        STOP = False
        PAUSE =False    
        SLIDE = False
        VOL = VOL/20
     
        cap = cv2.VideoCapture(file.name)
        totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        position = app.getScale("Slider")
        current_frame = int((position * totalFrameCount)/(slider_max - slider_min))
        cap.set(1, current_frame)
        frame = np.zeros(shape = (width, height))
        framePerSecond = cap.get(cv2.CAP_PROP_FPS)
        sounded_frame =  np.zeros(shape = (width, height))
        play_position = 1

        while(True):
                VOL = app.getScale("Volume")/20

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
                currentFrameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if currentFrameNumber % 2 == 0:
                        continue
                if(ret):
                        app.bell()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        img = Image.fromarray(gray)
                        img = ImageTk.PhotoImage(img)
                        app.reloadImageData("Video", img, fmt="mpg")

                 
                        if play_position ==1:
                                played_sound = gray
                                played_sound = prepare_frame(gray)
                        currentcol = play_col_midi(played_sound, play_position) #returns current column
                        midilist = make_midi(currentcol)
                        midi_chord(midilist)
                        
                        if play_position < 30:
                                play_position += 1
                        else:
                                play_position = 1

                        
                        totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        pos = (currentFrameNumber / totalFrameCount * (slider_max - slider_min))
                        app.setScale("Slider", pos, callFunction=False)
                else:   
                        app.setScale("Slider", 0, callFunction=False)
                        break



                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        app.setScale("Slider", 0, callFunction=False)
        cap.release()


       

#Formats frame for sound playback (based on the Java skeleton code)
def prepare_frame(img):
        resized = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)      
        roundedImg =  np.zeros(shape = (resized.shape[0], resized.shape[1]))
        for i in range(resized.shape[0]):
                for j in range(resized.shape[1]):
                        roundedImg[i,j] = mth.floor(resized[i,j]/numberOfQuantizionLevels)/numberOfQuantizionLevels
        return roundedImg

#Plays the sound based on a single column of the frame selected for audio representation.
def play_col(img, position):
        global sample_rate, freq, VOL
        current_col = img[:, position*2]
        chord = sine_wave(1, current_col[0], sample_rate)
        
        for c in range(height):
                chord = sum([chord, sine_wave(c, current_col[c]*127*VOL, sample_rate)])

        play_sound(chord, 33)

 #As above, only within the MIDI playback
def play_col_midi(img, position):
        current_col = img[:, position*2]

        return current_col
#When the user moves the Slider, the video position will be set to conform to its position
def slide_video(value):
        global file, width, height, SLIDE
        if file == None:
                return
        cap = cv2.VideoCapture(file.name)
        position = app.getScale("Slider")
        totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = int((position * totalFrameCount)/(slider_max - slider_min))
        cap.set(1, current_frame)
        currentFrameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame = np.zeros(shape = (width, height))
        ret, frame = cap.read()
        if(ret):
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)
                app.reloadImageData("Video", img, fmt="mpg")
        #This value used to communicate with the Play functions
        SLIDE = True
        return


#Exit the Application
def exit_program(btn):
        sys.exit()
        return
#Pause the application --> Hit Play to resume
def pause_video(btn):
        global PAUSE
        PAUSE = True
        return
#Stop the application
def stop_playing(btn):
        global STOP 
        STOP = True 
        return

#FM generation of single tone
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
    #Play the given NumPy array, as a sound, for ms milliseconds
    sound = pygame.sndarray.make_sound(wave)
    sound.play(-1)
    pygame.time.delay(ms)
    sound.stop()

#Generates a chord of tones from an array
def make_chord(voices):
    sampling = 4096
    chord = waveform(voice[v], sampling)
    for v in len(voices):
        chord = sum(sine_wave(freq[v], voices[v]*64, sample_rate))
    return chord

#Make an array to hold all the midi chords we want played
#chords are determined by pixel color and intensity
def make_midi(col):
        midis = []
        for i in range(height):
                
                midis.append([i+60,(mth.floor(((col[i])*63*VOL)+100))])
        return midis
                             
      
#Play midi chords - function is passed an array of midi chords to play 
def midi_chord(*notes):
    """Make a chord using the midi library"""
    player = pygame.midi.Output(0)
    player.set_instrument(0)

    
    VOL = app.getScale("Volume")/10
    for i in range(8):

            if i == 49: #prevents indexing outside of the array
                    break
            player.note_on(notes[0][i][0], notes[0][i+1][1])
            player.note_on(notes[0][i+1][0], notes[0][i+1][1])
            player.note_on(notes[0][i+2][0], notes[0][i+2][1])
            player.note_on(notes[0][i+3][0], notes[0][i+3][1])
            player.note_on(notes[0][i+4][0], notes[0][i+4][1])
            player.note_on(notes[0][i+5][0], notes[0][i+5][1])
            player.note_on(notes[0][i+6][0], notes[0][i+6][1])
            player.note_on(notes[0][i+7][0], notes[0][i+7][1])
      
            time.sleep(0.25)
            i = i+8   
    
#Beginning of GUI section
w = 600
h = 360
frame = "blind.gif"
app = gui("Blind Helper")

pygame.mixer.init(channels = 1)
app.setPadding([5,5]) # 20 pixels padding outside the widget [X, Y]
app.setInPadding([5,5]) # 20 pixels padding inside the widget [X, Y]

sample_rate = 8000

app.bell()
app.showSplash(text = 'Loading Application', fill = 'blue', stripe = 'black',font='44', fg='white')
app.setBg('#6a79ff')


app.createMenu("Menu")
app.addMenuItem("Menu", "Open Video", func = open_video, shortcut=None, underline=-1)
app.addMenuItem("Menu", "Exit", func = exit_program, shortcut = None, underline = -1)
app.setStretch("both")
app.addScale("Slider")
app.setScaleIncrement("Slider", 1)
app.setScaleRange("Slider", 0, 100)
app.setScaleChangeFunction("Slider", slide_video)


app.startLabelFrame("Player")
app.setFont(20)
app.addImage("Video", frame)
app.setImageSize("Video", w, h)
app.setStretch("both")


app.stopLabelFrame()
app.startLabelFrame("Controls")
app.setStretch("both")
app.startLabelFrame("")

app.stopLabelFrame()
app.addButton("Open", open_video, 6, 2, 2)
app.addButton("Pause", pause_video, 6, 4, 2)
app.addButton("Play", start_play_thread, 6, 6, 2)
app.addButton("Stop", stop_playing, 6, 8, 2)
app.addButton("Midi", start_play_thread_midi, 6, 10, 2)

app.addLabelScale("Volume", 6, 12)
app.setScaleVertical("Volume")
app.setScaleRange("Volume", 5, 20)
app.setScale("Volume", 10)
app.setScaleIncrement("Volume", 1)
app.stopLabelFrame()


initialize()
app.go()


        
