# -*- coding: utf-8 -*-
import sys
import SchemDraw as schem
import SchemDraw.elements as e
import matplotlib.pyplot as plt
import sympy as sy
init_printing()
#sys.displayhook = display

#Sympy definitions
I, IR1, IR2, Vin, Vout, R1, R2, R3, R4, Rg, Rload = sy.symbols('I, IR1, IR2, Vin, Vout, R1, R2, R3, R4, Rg, RLoad')

Vout = Vin * ( IR2 / (I*(R1 + R2)))
display(Vout)

Vout = Vin * ((Vin* ((R2*Rload) / (R2+Rload))  ) / (R1 + ((R2*Rload) / (R2+Rload))))  #With load
display(Vout)

### PLOTS AREA
def plotxy(x,y):
    plt.xkcd(False)
    fig2 = plt.figure()
    #plt.xlim(0.005,1.5)
    #plt.ylim(0.01,4)
    plt.title("Voltage mean Resistor/Gauge")
    plt.plot(y,x, 'r-')
    ax= fig2.add_subplot(222)
    ax.spines['right'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(False)
    plt.yscale('log')
    plt.plot(x,y, 'g-')
    plt.show()
        
def Schematic(Vin, R1, R2, Rload, Rload_current, Vout):
    plt.xkcd(True)
    d = schem.Drawing()
    d.fontsize = 9
    
    V1 = d.add( e.BATTERY, d='up', label='%.1fV' % Vin )
    d.add( e.LINE, d='right', to=V1.start )
    d.add( e.LINE, d='right', move_cur=True)
    
    R1 = d.add( e.RES, d='down', label='R1=%.2f' % R1)
    net1_A = d.add( e.LINE, d='right', move_cur=True)
    d.push()
    
    C = d.add(e.DOT_OPEN, label='C')
    R2 = d.add( e.RES, xy=R1.end, d='down', label='R2=%.2f $\Omega$' % R2)
    
    d.pop()
    d.add(e.DOT_OPEN, label='A')
    Rload = d.add( e.RES, d='down', label='Rload=%.2f'% Rload, move_cur=False)
    d.add( e.LINE, d='right', move_cur=True)
    d.add( e.METER_V, label='V=%.2f' % Vout, d='down')
    d.add( e.LINE, d='left', move_cur=True)
    B = d.add(e.DOT_OPEN, label='B')    
    d.push()
    
    #net2_A = d.add( e.LINE, d='right', move_cur=True)

#    d.add( e.LINE, d='right', move_cur=True)
    d.add(e.LINE, to=R2.end)
    d.add( e.LINE, d='left', move_cur=True)
    d.add( e.LINE, d='left', move_cur=True)
    d.add(e.LINE, to=V1.start)
    d.draw()

### FUNCTION AREA

Vin = 220                #volts
R1 = 10000000 # R103
R2 = 220 # R103
Rload = 200

R2_Rload = (R2*Rload) / (R2+Rload)

# Vout = Vin * ( IR2 / I*(R1 + R2))
Vout = Vin * ((Vin*R2) / (R1 + R2))  #Without load
#Vout = Vin * ((Vin* ((R2*Rload) / (R2+Rload))  ) / (R1 + ((R2*Rload) / (R2+Rload))))  #With load

Rload_current = Vout / Rload

Schematic(Vin, R1, R2, Rload, Rload_current, Vout)
print("Current through R1:",Vin / (R1+R2))
print("Current through R2:",Vout / R2)

