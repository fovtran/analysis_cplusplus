# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
import os
import random
from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import SchemDraw as schem
import SchemDraw.elements as e

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

Vin = 5                #volts
R1 = 220 # R103
R2 = 100 # R103
R3 = 50 # R103
Rg = 0.1  #ohms

I1 = Vin / (R1+R2)  # offset current
I2 = Vin / (R3+Rg)

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

import gc
gc.enable()

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=7, height=5, dpi=72):
        plt.xkcd(True)
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)        
        self.axes.set_title('Voltage mean Resistor/Gauge')
        self.axes.spines['right'].set_position('center')
        self.axes.spines['right'].set_color('none')
        self.axes.spines['top'].set_color('none')    
        self.axes.xaxis.set_ticks_position('bottom')
        self.axes.yaxis.set_ticks_position('left')
        #plt.xlim(0.005,1.5)
        #plt.ylim(0.01,4)
        #plt.yscale('log')
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        #FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""

    def compute_initial_figure(self):
        Rgs = [0.01, 0.05, 0.08, 0.1, 0.2, 0.5, 1, 1.2, 1.5, 2, 2.5, 3.9]
        Rsx = []
        Rsy = []
        for Rg1 in Rgs:
            Rsx.append( calcVmeas(R1, R2, R3, Rg1))
            Rsy.append(Rg1)
    
        self.axes.plot(Rsx,Rsy, 'r--')


class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")
        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtGui.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtGui.QWidget(self)
        
        l = QtGui.QVBoxLayout(self.main_widget)
        sc = MyStaticMplCanvas(self.main_widget, width=8, height=4, dpi=72)
        l.addWidget(sc)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtGui.QMessageBox.about(self, "About",
                                """embedding_in_qt4.py example Copyright 2005 Florent Rougon, 2006 Darren Dale"""
                                )


qApp = QtGui.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % progname)
aw.show()
sys.exit(qApp.exec_())