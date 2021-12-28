import threading

def hello():
    print('Hello')
    a=1

t = threading.Timer(5.0, hello)
t.start()

a=0
while(a==0):
   print("you're here")
print('Good By')
