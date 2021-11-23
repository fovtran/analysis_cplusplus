import math

# input voltage divider for preamp
v = 23.0
r1 = 1000000
r2 = 2200000
r = r2+r1
x = (v/r)*1000  # output in mA
s = f"{x:5.12f}"
v1 = (v*r2) / (r1+r2)
print(s, v1)
