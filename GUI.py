import tkinter as tk
root = tk.Tk()
root.title("Prediction_interactable_window")
root.geometry('500x500')
btn = tk.Button(root, text = "Predict" ,
             fg = "black").pack()
root.mainloop()