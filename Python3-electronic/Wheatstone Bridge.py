# -*- coding: utf-8 -*-
# Wheatstone calc
#
import SchemDraw as schem
import SchemDraw.elements as e
import matplotlib.pyplot as plt
import sympy as sy
init_printing()
#sys.displayhook = display

#Sympy definitions
I, IR1, IR2, Vin, Vout, R1, R2, R3, R4, Rg, Rload = sy.symbols('I, IR1, IR2, Vin, Vout, R1, R2, R3, R4, Rg, RLoad')

### PLOTS AREA
def plotxy(x,y):
    plt.xkcd(False)
    fig2 = plt.figure()
    #plt.xlim(-0.01,.200)
    #plt.ylim(-0.5,5)
    plt.title("Voltage mean Resistor/Gauge")
    plt.plot(y,x, 'r*-')
    ax= fig2.add_subplot(222)
    ax.spines['right'].set_position('center')
    ax.spines['left'].set_color('green')
    ax.spines['top'].set_color('none')    
    ax.spines['bottom'].set_color('black')    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True)
    plt.xscale('log')
    ax.plot(y[:6],x[:6], 'g-*')
    plt.show()
        
def Schematic(R1, R2, R3, Rg, Vmeas):
    plt.xkcd(True)
    d = schem.Drawing()
    d.fontsize = 9
    
    V1 = d.add( e.BATTERY, d='up', label='%.1fV' % Vin )
    d.add( e.LINE, d='right', to=V1.start )
    d.add( e.LINE, d='right', move_cur=True)
    
    net1_A = d.add( e.LINE, d='right', move_cur=True)
    net2_A = d.add( e.LINE, d='right', move_cur=True)
    net0_A = d.add(e.DOT_OPEN, label='A')
    
    d.push()
    R1 = d.add( e.RES, xy=net1_A.start, d='down', label='R1=%.2f$\Omega$' % R1)
    C = d.add(e.DOT_OPEN, label='C')
    R2 = d.add( e.RES, xy=R1.end, d='down', label='R2=%.2f$\Omega$' % R2)
    
    
    d.pop()
    R3 = d.add( e.RES, xy=net2_A.start, d='down', label='R3=%.2f$\Omega$' % R3)
    d.add(e.DOT_OPEN, label='D')
    d.add( e.METER_V, label='V=%.3f' % Vmeas, to=R1.end)
    d.push()
    
    Rg = d.add( e.RES, xy=R3.end, d='down', label='Rg=%.2f$\Omega$' % Rg)
    net2_A = d.add( e.LINE, d='right', move_cur=True)
    B = d.add(e.DOT_OPEN, label='B')
    
    d.add(e.LINE, to=R2.end)
    d.add( e.LINE, d='left', move_cur=True)
    d.add( e.LINE, d='left', move_cur=True)
    d.add(e.LINE, to=V1.start)
    d.draw()

### FUNCTION AREA
# IDEAL Vin=8  R1,R3=390  R2=10 ==> 200mV-140mV
# IDEAL Vin=15  R1,R3=390  R2=3.9 ==> 150mV-100mV
# IDEAL Vin=15  R1,R3=100  R2=10 ==> 1365mV-1200mV
Vin = 15               #volts
R1 = 220 # R103
R2 = 39 # R103
R3 = 220  # R103
Rg = 0.005  #ohms

I1 = Vin / (R1+R2)  # offset current
I2 = Vin / (R3+Rg)
display(I1)
display(I2)
print("R1=%.2f  R2=%.2f  R3=%.2f Rg=%.2f" % (R1, R2, R3, Rg))
print("Current for node I1 is:",I1)
print("Current for node I2 is:",I2)

C_v = Vin*(R2/(R1+R2))  #V3 = I1*R3 = A
D_v = Vin*(Rg/(R3+Rg))  #Vg = I2*Rs = A
print("Voltage at point C:",C_v)
print("Voltage at point D:",D_v)

# Vmeas = C_v - D_v 
Vmeas = Vin* (R2/(R1+R2) - Rg/(R3+Rg) )
print("Voltage differential",Vmeas)

def calcVmeas(R1,R2,R3,Rg):
    return Vin* (R2/(R1+R2) - Rg/(R3+Rg) )
    
Rgs = [0.001, 0.005, 0.01, 0.05, 0.08, 0.1, 0.2, 0.5, 1, 1.2]

Rsx = []
Rsy = []
for Rg1 in Rgs:
    Rsx.append( calcVmeas(R1, R2, R3, Rg1))
    Rsy.append(Rg1)

Schematic(R1, R2, R3, Rg, Vmeas)
plotxy(Rsx,Rsy)

plt.close('all')