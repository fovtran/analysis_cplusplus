import threading

# Given a function, f, thread it like this:
threading.Thread(target=f).start()
# To pass arguments to f
threading.Thread(target=f, args=(a,b,c)).start()