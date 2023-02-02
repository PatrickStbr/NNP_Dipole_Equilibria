#Thermal equilibrium of a non-neutral plasma in a magnetic dipole trap

The thermal equilibria that can be calculated with this code are either local, along a magnetic field line or global. 
A global thermal equilibrium requires the total number of particles, rotation frequency and temperature of the plasma as input.
On top of that one has to make sure that the trap provides a potential well in the respective rotating frame of refference.
A local thermal equilibrium is the more general case and does not require a potential well. 
It requires the number of particles and temperature on each field line as input parameter. 
Alternatively one can specify the density across a certain line perpendicalus to the magnetic field lines as input.
The theory behind this and a description of the numerical methods was published in "?"

This repository contains the package Dipole_Tools with some helpfull functions. This package is used by the acompanied notebooks. 
If you run them, note that results for the global thermal equilibrium (GTE) serve as input for the local thermal equilibrium (LTE).
So you want to run them in the following order: 1. GTE.ipynb, 2. GTE2LTE.ipynb, 3. LTE.ipynb, 4. LTE_Cold.ipynb
Otherwise you need to specify the input parameters yourself.
The notebooks produce all results and figures published in "?"
