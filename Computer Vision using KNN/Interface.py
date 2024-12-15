import tkinter as tk
import os
import subprocess
import urllib.request  
import webbrowser  

root = tk.Tk()
root.title('Projet Machine Learning Ahmed Jedidi D-IITWM')
root.geometry("500x550")
root.iconbitmap('face.ico')

def image(smp):
    img = tk.PhotoImage(file="image.png")
    img = img.subsample(smp, smp)
    return img
 
def add_new_face():
    os.system("add_new_face.py") 

def face_regonition():
    os.system("face_recognition.py") 
def open_site():
    #webUrl = urllib.request.urlopen('https://jedidi.me')  
    webbrowser.open_new_tab('https://ahmed-jedidi.github.io/')  

but = tk.Button(
    root,
    bd=0,
    compound=tk.CENTER,
    bg="white",
    fg="steel blue",
    activeforeground="pink",
    activebackground="white",
    font="arial 30",
    text="Add New Face",
    pady=10,
    command=add_new_face
    )
but1 = tk.Button(
    root,
    bd=0,
    relief="groove",
    compound=tk.CENTER,
    bg="white",
    fg="steel blue",
    activeforeground="pink",
    activebackground="white",
    font="arial 30",
    text="Face recognition",
    pady=10,
    command=face_regonition
    # width=300
    )
but2 = tk.Button(
    root,
    bd=0,
    relief="groove",
    compound=tk.CENTER,
    bg="white",
    fg="steel blue",
    activeforeground="pink",
    activebackground="white",
    font="arial 28",
    text="Visite my website",
    pady=10,
    # width=300
    command=open_site
    )

 
img = image(1) # 1=normal, 2=small, 3=smallest
but.config(image=img)
but.pack()
but1.config(image=img)
but1.pack()
but2.config(image=img)
but2.pack()

root.mainloop()