from __future__ import print_function
from tkinter import Tk

root = Tk()
root.withdraw() # don't show the GUI window
root.after(1000, print, 'foo') # print foo in a second
root.after(0, print, 'bar') # print bar in a jiffy
root.after(2000, root.destroy) # exit mainloop in 2 seconds
root.mainloop()
print("done")
