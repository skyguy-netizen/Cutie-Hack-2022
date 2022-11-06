from tkinter import *
import os
from faceMask import FaceTracker

window = Tk()
# button = tk.Button(
#     text="Click me!",
#     width=25,
#     height=5,
#     bg="blue",
#     fg="yellow",
# )
b1 = Button(window,text = "Face Tracker",command = FaceTracker,activeforeground = "red",activebackground = "pink",pady=20) 
b2 = Button(window,text = "Some fun cartoon pics",command = FaceTracker,activeforeground = "red",activebackground = "pink",pady=20) 
b1.pack()   
b2.pack()

window.mainloop()