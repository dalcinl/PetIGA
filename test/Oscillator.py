import numpy as np
import matplotlib.pylab as mpl
filename = 'Oscillator.out'

state = np.loadtxt(filename);
t = state[:, 0]
x = state[:, 1]
v = state[:, 2]

mpl.figure()
mpl.plot(t,x,'b')
mpl.title("Single DOF Oscillator")
mpl.xlabel("Time")
mpl.ylabel("Displacement")

mpl.figure()
mpl.plot(t,v,'r')
mpl.title("Single DOF Oscillator")
mpl.xlabel("Time")
mpl.ylabel("Velocity")

mpl.show()
