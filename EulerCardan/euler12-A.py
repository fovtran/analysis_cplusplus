import math

def factorCount (n):
    square = math.sqrt (n)
    isquare = int (square)
    count = -1 if isquare == square else 0
    for candidate in range (1, isquare + 1):
        if not n % candidate: count += 2
    return count

triangle = 1
index = 1
while factorCount (triangle) < 1001:
    index += 1
    triangle += index

print (triangle)
