def RGB2lambda(R, G, B):
   """Returns 0 if indeciferable"""
   # selects range by maximum component
   # if max is blue - range is 380 - 489
   # if max is green - range is 490 - 579
   # if max is red - range is 580 - 645

   # which colour has highest intensity?
   high = float(R)
   highind = 1
   if G > high:
       high = float(G)
       highind = 2
   if B > high:
       high = float(B)
       highind = 3

   # normalize highest to 1.0
   RGBnorm = [R / high, G / high, B / high]

   # start decoding
   RGBlambda = 0
   if highind == 1: # red is highest
       if B >= G: # there is more blue than green
           return 0 # max red and more blue than green shouldn't happen
       # wavelength linearly changes from 645 to 580 as green increases
       RGBlambda = 645 - RGBnorm[1] * (645. - 580.)

   elif highind == 2: # green is max, range is 510 - 579
       if R > B: # range is 510 - 579
           RGBlambda = 510 + RGBnorm[0] * (580 - 510)
       else: # blue is higher than red, range is 490 - 510
           RGBlambda = 510 - RGBnorm[2] * (510 - 490)

   elif highind == 3: # blue is max, range is 380 - 490
       if G > R: # range is 440 - 490
           RGBlambda = RGBnorm[1] * (490 - 440) + 440
       else: # there is more red than green, range is 380 - 440
           RGBlambda = 440 - RGBnorm[0] * (440 - 380)

   return RGBlambda
