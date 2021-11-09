import ahkab
import pylab

osc = ahkab.Circuit('MOS COLPITTS OSCILLATOR')

# models need to be defined before the devices that use them
osc.add_model('ekv', 'nmos', dict(TYPE='n', VTO=.4, KP=10e-6))

osc.add_vsource('vdd', n1='dd', n2=osc.gnd, dc_value=3.3)

# Ql = 33 at 3GHz
osc.add_inductor('l1', n1='dd', n2='nd', value=5e-9, ic=-1e-9)
osc.add_resistor('r0', n1='nd', n2='dd', value=3.5e3)

# n = 0.5, f0 = 3GHz
osc.add_capacitor('c1', n1='nd', n2='ns', value=1.12e-12)
osc.add_capacitor('c2', n1='ns', n2=osc.gnd, value=1.12e-12)

osc.add_mos('m1', nd='nd1', ng='bias', ns='ns', nb='ns',
            model_label='nmos', w=600e-6, l=100e-9)
# voltage source as a current probe
osc.add_vsource('vtest', n1='nd', n2='nd1', dc_value=0)

# Bias
osc.add_vsource('vbias', n1='bias', n2=osc.gnd, dc_value=2.)
osc.add_isource('ib', n1='ns', n2=osc.gnd, dc_value=1.3e-3)

# calculate an Operating Point (OP) to initialize the transient
# analysis
op = ahkab.new_op()
res = ahkab.run(osc, op)

# modify the OP to give the circuit a little kick to start the
# oscillation
x0 = res['op'].asarray()
l1vdei = osc.find_vde_index('l1')
l1i = len(osc.nodes_dict) - 1 + l1vdei
x0[l1i, 0] += -1e-9

# Setup and run a transient analysis with the modified x0 as start point
tran = ahkab.new_tran(tstart=0., tstop=20e-9, tstep=.01e-9, method='trap',
                      x0=x0)
res = ahkab.run(osc, tran)['tran']

# plot the results!
pylab.subplot(211)
pylab.hold(True)
pylab.plot(res.get_x(), res['vnd'], label='ND')
pylab.plot(res.get_x(), res['vns'], label='NS')
pylab.plot(res.get_x(), res['vbias'], label='BIAS')
pylab.legend()
pylab.subplot(212)
pylab.plot(res.get_x(), res['i(vtest)'], label='I(VTEST)')
pylab.legend()
pylab.show()
