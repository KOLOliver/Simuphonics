import DCGANMODEL as gw
import tensorflow as tf
import os
import time
from threading import Thread
import gc

from kivy import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('kivy', 'exit_on_escape', '0')
from kivymd.app import MDApp
# from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.progressbar import MDProgressBar 
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from multiprocessing import Queue, Process
from kivy.properties import StringProperty 

os.environ["KIVY_VIDEO"] = "ffpyplayer"
# Window.fullscreen = "auto"
relpath = ""
epochs = 3500
sampling_iter = 50
verboseLogging = False

class DemoScreen(Screen):
    backButton = Button()
    continueButton = Button()
    playDogBarkButton = Button()
    playGeneratedButton = Button()
    dogBarkPB = ProgressBar()
    generatedPB = ProgressBar()
    isPlaying = False

    def updateDPB(self, dt):
        if self.dogBarkPB.value < self.dogBarkPB.max:
            self.dogBarkPB.value += 1
        else:
            self.isPlaying = False
            self.dogBarkPB.value = 0
            return False
        
    def updateGPB(self, dt):
        if self.generatedPB.value < self.generatedPB.max:
            self.generatedPB.value += 1
        else:
            self.isPlaying = False
            self.generatedPB.value = 0
            return False

    def build(self):
        def switchToInstructions(self):
            self.parent.parent.parent.transition.direction = "right"
            self.parent.parent.parent.current = "instruct"

        def switchToGenerate(self):
            self.parent.parent.parent.transition.direction = "left"
            self.parent.parent.parent.current = "generate"
        
        def playDogBark(self):
            sound = SoundLoader.load("dogbarkdemo.wav")
            if sound and not self.parent.parent.parent.isPlaying:
                self.parent.parent.parent.isPlaying = True
                self.dogBarkPB = MDProgressBar(max=sound.length)
                self.dogBarkPB.prog_ev = Clock.schedule_interval(self.parent.parent.parent.updateDPB, 1.0/44100) 
                sound.play()

            else:
                print("Error: Already playing!")
        
        def playGenerated(self):
            sound = SoundLoader.load("generateddemo.wav")
            if sound and not self.parent.parent.parent.isPlaying:
                self.parent.parent.parent.isPlaying = True
                self.generatedPB = MDProgressBar(max=sound.length)
                self.generatedPB.prog_ev = Clock.schedule_interval(self.parent.parent.parent.updateGPB, 1.0/44100) 
                sound.play()
            else:
                print("Error: Already playing!")
        
        self.backButton.bind(on_press=switchToInstructions)
        self.continueButton.bind(on_press=switchToGenerate)
        self.playDogBarkButton.bind(on_press=playDogBark)
        self.playGeneratedButton.bind(on_press=playGenerated)

class GenerationScreen(Screen):
    backButton = Button()
    startButton = Button()
    stopButton = Button()
    inputFileName = TextInput()
    validityText = Label(markup = True)
    statusLabel = Label(markup = True)
    continuelog = Queue()
    validFile = False
    hasStarted = False
    filename = ""

    def generateAudio(self):
        global relpath; global epochs; global sampling_iter; global verboseLogging
        
        self.hasStarted = True
        self.statusText.text = "Status: Preparing Tensorflow Prerequisites!"
        gc.collect()

        try:
            trainX, trainY = gw.createDataset(self.filename)
            self.statusText.text = "Status: Starting Generator Model Initialization!" 
        except FileNotFoundError:
            self.statusText.text = "Status: Something Went Wrong! Try a different file." 

        gen, disc = tf.keras.Sequential(), tf.keras.Sequential()
        gt = Thread(target=gw.create_generator, args=(gen,), kwargs={})
        gt.daemon = True
        gt.start()
        gt.join()
        self.statusText.text = "Status: Starting Discriminator Model Initialization!" 
        
        dt = Thread(target=gw.create_discriminator, args=(disc,), kwargs={})
        dt.daemon = True
        dt.start()
        dt.join()
        self.statusText.text = "Status: Saving Models!"
        
        try:
            os.remove('generator.keras')
        except FileNotFoundError:
            pass 
        try:
            os.remove('discriminator.keras')
        except FileNotFoundError:
            pass

        gen.save('generator.keras')
        disc.save('discriminator.keras')

        queue = Queue()
        try:
            if self.hasStarted:
                self.statusText.text = "Status: Beginning Training! This text will be replaced with training logs."
                if relpath:
                    outputFileName = self.filename.split("/")[-1]
                    generationProcess = Process(target=gw.train, args=(trainX, epochs, queue, self.continuelog, relpath + f"/{outputFileName}", sampling_iter, verboseLogging), kwargs={})
                else:
                    generationProcess = Process(target=gw.train, args=(trainX, epochs, queue, self.continuelog, self.filename, sampling_iter, verboseLogging), kwargs={})
                generationProcess.daemon = True
                generationProcess.start()

                if verboseLogging:
                    self.statusText.font_size = 20
                    self.statusText.text = "Status: Verbose logging is on! Font size reduced..."
                
                ct = 0
                while ct <= epochs:
                    if not self.hasStarted:
                        break
                    if not queue.empty(): 
                        self.statusText.text = str(queue.get())
                        ct += 1
                    time.sleep(0.05)
            
        except OSError:
            self.statusText.text = "Status: Error loading in the models."

        time.sleep(1)
        self.statusText.font_size = 30
        self.statusText.text = "Status: Finished!"
        time.sleep(1)
        self.hasStarted = False
        self.statusText.text = "Status: Not Started"

    def build(self):
        def forceStop(self):
            
            if not self.parent.parent.hasStarted:
                self.parent.parent.statusText.text = "Status: No Training to Stop! Please input a proper .wav file and then click the generation button."
            
            else:
                self.parent.parent.continuelog.put(False)
                time.sleep(0.5)
                self.parent.parent.statusText.text = "Generation has been force stopped!"
                self.parent.parent.hasStarted = False

        def switchToDemo(self):
            self.parent.parent.parent.transition.direction = "right"
            self.parent.parent.parent.current = "demo"

        def on_focus(self, value): #Check if inputted string is in file directory and is a .wav file
            if not value:
                self.parent.parent.filename = self.text
                self.parent.parent.validFile = os.path.isfile(self.parent.parent.filename + ".wav")
                if self.parent.parent.validFile:
                    self.parent.parent.validityText.text = f"{self.parent.parent.filename}.wav is a valid file!"
                    self.parent.parent.validityText.color = (59.0/255,177.0/255,67.0/255,1)
                else:
                    if self.text == "":
                        self.parent.parent.validityText.text = "No text input detected!"
                        self.parent.parent.validityText.color = (1,(36.0/255),0,1)
                    else:
                        self.parent.parent.validityText.text = f"{self.parent.parent.filename}.wav does not exist or is not a .wav formatted file!"
                        self.parent.parent.validityText.color = (1,(36.0/255),0,1)
        
        def on_release_gen(self):
            if not self.parent.parent.validFile:
                self.parent.parent.statusText.text = "Status: Please enter a valid file name!"
            else:
                if not self.parent.parent.hasStarted:
                    Thread(target=self.parent.parent.generateAudio, args=(), kwargs={}).start()                    

        self.validityText.text = "No text input detected!"
        self.validityText.color = (1,(36.0/255),0,1)
        self.statusText.text = "Status: Not Started"
        self.backButton.bind(on_press=switchToDemo)
        self.startButton.bind(on_release=on_release_gen)
        self.stopButton.bind(on_release=forceStop)
        self.inputFileName.bind(focus=on_focus)


class GenerationInstructionsScreen(Screen):

    backButton = Button()
    continueButton = Button()
    def build(self):
        def switchToMenu(self):
            self.parent.parent.parent.transition.direction = "right"
            self.parent.parent.parent.current = "menu"
        
        def switchToDemo(self):
            self.parent.parent.parent.transition.direction = "left"
            self.parent.parent.parent.current = "demo"
        
        self.backButton.bind(on_press=switchToMenu)
        self.continueButton.bind(on_press=switchToDemo)
    # pass

class CreditsScreen(Screen):
    titleLabel = Label(markup=True)
    backButton = Button()
    def build(self):
        def switchToMenu(self):
            self.parent.parent.parent.transition.direction = "right"
            self.parent.parent.parent.current = "menu"
        self.backButton.bind(on_press=switchToMenu)
        # pass

# class Menu(Widget):

class SettingsScreen(Screen):
    
    returnButton = Button()
    inputRelativePath = TextInput()
    inputEpochs = TextInput()
    inputSampling = TextInput()
    toggleVerboseLogging = CheckBox()
    validityTextOutfile = Label(markup = True)
    validityTextEpochs = Label(markup = True) 
    validityTextSampling = Label(markup = True) 
    statusTextVL = Label(markup=True)

    def build(self):
        def switchToMenu(self):
            self.parent.parent.parent.transition.direction = "right"
            self.parent.parent.parent.current = "menu"

        def on_focus_outfile(self, value): #Check if inputted string is a valid path to a folder
            global relpath
            if not value:
                if os.path.isdir(self.text):
                    relpath = self.text
                    self.parent.parent.validityTextOutfile.text = f"{relpath} is a valid directory!"
                    self.parent.parent.validityTextOutfile.color = (59.0/255,177.0/255,67.0/255,1)
                else:
                    if self.text == "":
                        self.parent.parent.validityTextOutfile.text = "Using input location as output location."
                        self.parent.parent.validityTextOutfile.color = (59.0/255,177.0/255,67.0/255,1)
                    else:
                        self.parent.parent.validityTextOutfile.text = f"{self.text} is not a valid directory!"
                        self.parent.parent.validityTextOutfile.color = (1,(36.0/255),0,1)

        def on_focus_epochs(self, value): #Check if inputting string is positive int and thus a valid epoch count
            global epochs
            global sampling_iter
            if not value:
                if self.text.isdigit() and int(self.text) > 0:
                    epochs = int(self.text)
                    self.parent.parent.validityTextEpochs.text = f"{self.text} is a valid epoch count!"
                    self.parent.parent.validityTextEpochs.color = (59.0/255,177.0/255,67.0/255,1)
                    if epochs < 50:
                        sampling_iter = int(epochs/2)
                        self.parent.parent.validityTextSampling.text = f"Using {sampling_iter} as sampling number."
                        self.parent.parent.validityTextSampling.color = (59.0/255,177.0/255,67.0/255,1)
                else:
                    if self.text == "":
                        epochs = 3500
                        self.parent.parent.validityTextEpochs.text = "Using 3500 as epoch count."
                        self.parent.parent.validityTextEpochs.color = (59.0/255,177.0/255,67.0/255,1)
                    else:
                        self.parent.parent.validityTextEpochs.text = f"{self.text} is not a positive integer!"
                        self.parent.parent.validityTextEpochs.color = (1,(36.0/255),0,1)
        
        def on_focus_sampling(self, value): #Check if inputting string is positive int and is less than epoch count
            global epochs
            global sampling_iter
            if not value:
                if self.text.isdigit() and int(self.text) > 0 and int(self.text) < epochs:
                    sampling_iter = int(self.text)
                    self.parent.parent.validityTextSampling.text = f"{self.text} is a valid number of iterations!"
                    self.parent.parent.validityTextSampling.color = (59.0/255,177.0/255,67.0/255,1)
                else:
                    if self.text == "":
                        if epochs >= 50:
                            sampling_iter = 50
                            self.parent.parent.validityTextSampling.text = "Using 50 as sampling number."
                            self.parent.parent.validityTextSampling.color = (59.0/255,177.0/255,67.0/255,1)
                        else:
                            # sampling_iter = epochs/2
                            self.parent.parent.validityTextSampling.text = f"Using {sampling_iter} as sampling number."
                            self.parent.parent.validityTextSampling.color = (59.0/255,177.0/255,67.0/255,1)
                    else:
                        self.parent.parent.validityTextSampling.text = f"{self.text} is not a valid sampling number!"
                        self.parent.parent.validityTextSampling.color = (1,(36.0/255),0,1)

        def on_toggle_VL(self, value):
            global verboseLogging
            verboseLogging = value
            if value:
                self.parent.parent.statusTextVL.text = f"Verbose logging is on!"
                self.parent.parent.statusTextVL.color = (59.0/255,177.0/255,67.0/255,1)
            else:
                self.parent.parent.statusTextVL.text = f"Verbose logging is off!"
                self.parent.parent.statusTextVL.color = (1,(36.0/255),0,1)

        self.validityTextOutfile.text = "Using input location as output location."
        self.validityTextOutfile.color = (59.0/255,177.0/255,67.0/255,1)
        self.validityTextEpochs.text = "Using 3500 as epoch count."
        self.validityTextEpochs.color = (59.0/255,177.0/255,67.0/255,1)
        self.validityTextSampling.text = "Using 50 as sampling number."
        self.validityTextSampling.color = (59.0/255,177.0/255,67.0/255,1)
        self.statusTextVL.text = f"Verbose logging is off!"
        self.statusTextVL.color = (1,(36.0/255),0,1)
        self.returnButton.bind(on_press=switchToMenu)
        self.inputRelativePath.bind(focus=on_focus_outfile)
        self.inputEpochs.bind(focus=on_focus_epochs)
        self.inputSampling.bind(focus=on_focus_sampling)
        self.toggleVerboseLogging.bind(active=on_toggle_VL)



class MenuScreen(Screen):
    
    startButton = Button()
    creditsButton = Button()
    settingsButton = Button()
    
    def build(self):
        def switchToInstructions(self):
            self.parent.parent.parent.transition.direction = "left"
            self.parent.parent.parent.current = "instruct"

        def switchToCredits(self):
            self.parent.parent.parent.transition.direction = "left"
            self.parent.parent.parent.current = "credits"
        
        def switchToSettings(self):
            self.parent.parent.parent.transition.direction = "left"
            self.parent.parent.parent.current = "settings"

        self.startButton.bind(on_press=switchToInstructions)
        self.creditsButton.bind(on_press=switchToCredits)
        self.settingsButton.bind(on_press=switchToSettings)
        # pass

class DCGANApp(MDApp):
    def build(self):

        sm = ScreenManager() #initialize the screen manager

        #Add all screens into the manager
        sm.add_widget(MenuScreen(name="menu"))
        sm.add_widget(CreditsScreen(name="credits"))
        sm.add_widget(GenerationInstructionsScreen(name="instruct"))
        sm.add_widget(DemoScreen(name="demo"))
        sm.add_widget(GenerationScreen(name="generate"))
        sm.add_widget(SettingsScreen(name="settings"))
        
        #Build each screen
        sm.get_screen("menu").build()
        sm.get_screen("credits").build()
        sm.get_screen("instruct").build()
        sm.get_screen("demo").build()
        sm.get_screen("generate").build()
        sm.get_screen("settings").build()
        
        #Set current screen then return
        sm.current = "menu"
        return sm
    
if __name__ == '__main__':
    DCGANApp().run()
