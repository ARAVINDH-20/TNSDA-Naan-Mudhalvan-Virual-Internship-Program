import speech_recognition as sr  #speech to text
import pyttsx3 as pt    #text to speech
import datetime
import webbrowser
import random

engine = pt.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice",voices[1].id)

class Twinkle:
    def speak(self,audio):
        engine.say(audio)
        engine.runAndWait()

    def wishme(self):
        hour = int(datetime.datetime.now().hour)
        if hour>=0 and hour<12:
            self.speak("Good Morning")
        elif hour>=12 and hour<18:
            self.speak("Good Afternoon")
        else:
            self.speak("Good Evening")


        self.speak("I am Twinkle, please tell me how may i help you?")

    def takecommand(self):
        r=sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source,duration=1)
            print("Listening..")
            audio=r.listen(source)
        try:
            print("Recognizing..")
            query=r.recognize_google(audio,language="en-in")
            print(f"User Query: {query}")
        except:
            print("say that again please..")
            return "None"
        return query




class stone_paper_scissor_game(Twinkle):
    def __init__(self):
        self.user_score = 0
        self.computer_score = 0
        
    def play(self):
        while(True):
            print(f"Your Score: {self.user_score}, Computer Score: {self.computer_score}")
            
            user_choice = self.takecommand().lower()
            
            options = ["stone","paper","scissor"]

            if "quit game" in user_choice:
                break
            
            elif user_choice in options:
                computer_choice = random.choice(options)
                self.speak(f"Computer Choice: {computer_choice} ")

                if(user_choice == computer_choice):
                    print("Tie")
                    print("______________")
                elif(user_choice == "stone" and computer_choice == "scissor"):
                    self.user_score +=1
                    print("You Won👏")
                    print("______________")
                elif(user_choice == "paper" and computer_choice == "stone"):
                    self.user_score +=1
                    print("You Won👏")
                    print("______________")
                elif(user_choice == "scissor" and computer_choice == "paper"):
                    self.user_score +=1
                    print("You Won👏")
                    print("______________")
                else:
                    self.computer_score +=1
                    print("Computer Won👍")
                    print("______________")

                    
                if self.computer_score == 5:
                    print("Computer Won the Game🏆")
                    self.speak("Computer Won the Game🏆")
                    break
                elif self.user_score == 5:
                    print("Congrats🥳🎉, You Won the Game🏆")
                    self.speak("Congrats🥳🎉, You Won the Game🏆")
                    break
            else:
                print("INVALID ENTRY😤")
                self.speak("INVALID ENTRY😤")
                print("______________")
    



ai = Twinkle()

ai.wishme()
while True:
    query=ai.takecommand().lower()

    if "who are you" in query:
        ai.speak("I am Your Virtual Assistant sir, My name is Twinkle")

    elif "what can you do" in query:
        ai.speak("I can search google, open Instagram, Tell joke, Tell time")

    elif "leave" in query:
        break

    elif "open linkedin" in query:
        webbrowser.open("linkedin.com")

    elif "open google" in query:
        webbrowser.open("google.com")
        ai.speak("Do you want me to search")
        a=ai.takecommand().lower()
        if "yes" in a:
            while True:
                    ai.speak("Listening..")
                    s=ai. takecommand().lower()

                    if "close" in s:
                        ai.speak("exiting google..")
                        break
                    elif "none" not in s:
                        webbrowser.open_new_tab(f"https://www.google.com/search?q={s}")

    elif "time now" in query:
        time_now = datetime.datetime.now().strftime("%H:%M:%S")
        ai.speak(f"time now is{time_now}")

    elif "play game" in query:
        ai.speak("Let's Play Stone,Paper,Scissors!!")
        game=stone_paper_scissor_game()
        game.play()
