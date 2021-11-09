def linearsolver(A,b):
  n = len(A)
  M = A

  i = 0
  for x in M:
   x.append(b[i])
   i += 1

  for k in range(n):
   print "Iteration ", k
   for i in range(k,n):
     if abs(M[i][k]) > abs(M[k][k]):
        M[k], M[i] = M[i],M[k]
     else:
        pass

   # Show the matrix after swapping rows
   for row in M:
     print row
   print

   for j in range(k+1,n):
       q = M[j][k] / M[k][k]
       for m in range(k, n+1):
          M[j][m] +=  q * M[k][m]

   # Show matrix after multiplying rows
   for row in M:
     print row
   print

  x = [0 for i in range(n)]

  print "n = ", n
  print "x = ", x
  for row in M: