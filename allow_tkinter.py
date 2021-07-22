# import tkinter as tk
from os import replace, times
from tkinter import *
from PIL import Image,ImageTk

import tkinter.ttk as exTk
class Demo1:
    def __init__(self, master=None):
        self.master = master
        self.frame = Frame(self.master).pack()
        # create object
        self.lable1 = exTk.Label(self.frame,text="Im bao \n gia bao",background="red", font="Times 30", relief=RAISED,borderwidth=5,anchor=CENTER,justify=CENTER)
        self.img = exTk.Label(self.frame,image="", anchor=CENTER)
        #position
        self.lable1.place(height=100,width=1000,x=10,y=0)
        self.img.place(height=500,width=1000,x=0,y=120)
        self.abc = 5
    def replace_img(self):
        print("225")
        image2 = ImageTk.PhotoImage(file="/home/bao/Desktop/DoAnTotNghiep/main/image/tech.png")
        self.img.configure(image=image2)
        return self.img
if __name__ == '__main__':
    try:
        #set up window
        win =Tk()
        win.title("window")
        win.geometry("1920x1080")
        win.resizable(width=False,height=False)
        brimg = ImageTk.PhotoImage(file="/home/bao/Desktop/DoAnTotNghiep/main/image/tech.png")
        br = Label(win,image=brimg).place(x=0,y=0,relheight=1,relwidth=1)

        app = Demo1(win)
     
        a = ImageTk.PhotoImage(file="/home/bao/Desktop/DoAnTotNghiep/main/image/hcmute.png")
        app.img = a
        app.img.configure(img = a)
    finally:
        win.mainloop()