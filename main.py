from tkinter import *
import os
from faceMask import FaceTracker
from face_recognition import Face_Recognition

window = Tk()
# button = tk.Button(
#     text="Click me!",
#     width=25,
#     height=5,
#     bg="blue",
#     fg="yellow",
# )
b1 = Button(window,text = "Mask Detection",command = FaceTracker,activeforeground = "red",activebackground = "pink",pady=20) 
b2 = Button(window,text = "Celebrity Lookalike",command = Face_Recognition,activeforeground = "red",activebackground = "pink",pady=20) 
b1.pack()   
b2.pack()

window.mainloop()