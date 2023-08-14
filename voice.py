import pyttsx3  
# initialize Text-to-speech engine  
engine = pyttsx3.init()  
# convert this text to speech  
text = "Hi how are you"  
engine.say(text)  
# play the speech  
engine.runAndWait() 