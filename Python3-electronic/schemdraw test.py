# -*- coding: utf-8 -*-

import SchemDraw as schem
import SchemDraw.elements as e
import matplotlib.pyplot as plt
plt.xkcd() 
    
Vin = 5 #volts
R1 = 990 # R103
R2 = 1010 # R103
R3 = 990 # R103
Rg = 1010.0  #ohms

d = schem.Drawing()
d.fontsize = 9

V1 = d.add( e.SOURCE_V, label='%.1fV' % Vin )
d.add( e.LINE, d='right', to=V1.start )
d.add( e.LINE, d='right', move_cur=True)

net1_A = d.add( e.LINE, d='right', move_cur=True)
net2_A = d.add( e.LINE, d='right', move_cur=True)
net0_A = d.add(e.DOT_OPEN, label='A')

d.push()
R1 = d.add( e.RES, xy=net1_A.start, d='down', label='R1=%i$\Omega$' % R1)
C = d.add(e.DOT_OPEN, label='C')
R2 = d.add( e.RES, xy=R1.end, d='down', label='R2=%i$\Omega$' % R2)


d.pop()
R3 = d.add( e.RES, xy=net2_A.start, d='down', label='R3=%i$\Omega$' % R3)
d.add(e.DOT_OPEN, label='D')
d.add( e.METER_V, label='V', to=R1.end)
d.push()

Rg = d.add( e.RES, xy=R3.end, d='down', label='Rg=%.2f$\Omega$' % Rg)
net2_A = d.add( e.LINE, d='right', move_cur=True)
B = d.add(e.DOT_OPEN, label='B')

d.add(e.LINE, to=R2.end)
d.add( e.LINE, d='left', move_cur=True)
d.add( e.LINE, d='left', move_cur=True)
d.add(e.LINE, to=V1.start)
d.draw()

Vin =5 
R1 = 22
R2 = 10000000
Rload = 22

plt.xkcd(True)
d = schem.Drawing()
d.fontsize = 9

V1 = d.add( e.BATTERY, d='up', label='%.1fV' % Vin )
d.add( e.LINE, d='right', to=V1.start )
d.add( e.LINE, d='right', move_cur=True)

R1 = d.add( e.RES, d='down', label='R1=%.2f$\Omega$' % R1)
net1_A = d.add( e.LINE, d='right', move_cur=True)
d.push()

C = d.add(e.DOT_OPEN, label='C')
R2 = d.add( e.RES, xy=R1.end, d='down', label='R2=%.2f$\Omega$' % R2)

d.pop()
d.add(e.DOT_OPEN, label='A')
d.add( e.LINE, d='right', move_cur=True)
d.add( e.METER_V, label='V=', d='down')
d.push()

Rload = d.add( e.RES, xy=R3.end, d='down', label='Rload')
#net2_A = d.add( e.LINE, d='right', move_cur=True)
B = d.add(e.DOT_OPEN, label='B')
d.add( e.LINE, d='right', move_cur=True)

d.add(e.LINE, to=R2.end)
d.add( e.LINE, d='left', move_cur=True)
d.add( e.LINE, d='left', move_cur=True)
d.add(e.LINE, to=V1.start)
d.draw()