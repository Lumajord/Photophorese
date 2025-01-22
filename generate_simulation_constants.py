# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:56:57 2015

@author: Lucas
"""

import numpy as np
from scipy import special as sc


def define(outfile, string, num, comment):
    outfile.write("#define " + string + " " + str(num) + "  // " + comment + "\n")
    
def define2(outfile, string, num, comment):
    outfile.write("#define " + string + " " + str(num) + comment + "\n")

"""
boundary:
1 = open volume with inflow boundaries
2 = closed volume with specular boundaries
3 = closed volume with diffuse boundaries
"""
boundary = 3

kB = 1.38064852e-23 # boltzmann constant


numSteps = 500000 # number of timesteps
generatorSeed = 123 # seed for cuda's XORWOW random generator


# grid dimensions
nx = np.int(64)
nz = np.int(nx/2)
ny = np.int(nx/2)

numGas = np.int(1.5e4)		#Number of gas particles
	

accomodationCoefficient = 1.0 # 0.0 for specular reflection and 1.0 for diffuse reflection



Kn = np.append(np.exp(np.linspace(np.log(1000), np.log(10), 3)), np.exp(np.linspace(np.log(10), np.log(0.5), 8))[1:]) # Knudsen number
#Kn = np.append(Kn, np.exp(np.linspace(np.log(0.10985605), np.log(0.01), 4)))
Kn = Kn[9]
Kn = 0.1
Kn = 1.e3

particleRadius = 1.0e-5 # radius of the spherical test body
edgeLength = 2.5*particleRadius # the size of the test volume is given in units of the test body's radius
edgeLengthD2 = edgeLength/2 # half the length of the simulation volume

# edge length of the collision cells
dx = edgeLength/nx
dz = edgeLength/(2.0*nz)
dy = edgeLength/(2.0*ny)

ax = 5.05 # mean flow velocity of the gas, only relevant for open boundaries




gasTemperature = 300.0		# Temperature of the gas

## reference parameters for hydrogen taken from G.A. Bird : The DSMC Method
T_ref = 273.0
d_ref = 2.92e-10
gasMass = 3.34e-27   
nu = 0.0#17
w = nu+0.5
## end reference parameters



simVol = edgeLength*edgeLengthD2**2 # volume of the simulation volume
bodyVol = 4/3*np.pi*particleRadius**3 / 4 # volume of the test body



numCells = nx*nz*ny # total number of cells


a = np.random.poisson(numGas/numCells, numCells)
print(np.min(a))

mfp = 2*Kn*particleRadius

#particleRadius = 0.0

vT = np.sqrt(2.0*gasTemperature*kB/gasMass) # most probable speed of the maxwell boltzmann distribution


particleDensity = 1/(np.sqrt(2)*np.pi * d_ref**2 * mfp * (T_ref/gasTemperature)**(w-0.5)) # particle density
rho = gasMass * particleDensity # particle mass density


# different average relative speed to the power of 2*nu
# see G.A. Bird : The DSMC Method
crP2nu = np.sqrt(16.0*kB*gasTemperature/(np.pi*gasMass))**(2*nu)
crP2nu2 = (4*kB*T_ref/gasMass)**(w-0.5)/sc.gamma(5/2-w)

sigma_ref = np.pi*d_ref**2*(crP2nu2/crP2nu) # reference cross section of the gas atoms



numParticles = particleDensity*simVol # number of theoretical particles inside the simulation volume
                                        # do not mix up the number of simulation particles!

## caltulate the timestep for the simulation
# the timestep is the minimum of :
# the 1/10 of the average mean collision time
# and 1/4 of the average time it takes a atom to cross a collision cell
cellLength = min(dz,dx,dy)
VxMean = np.sqrt(2.0*gasTemperature*kB/(np.pi*gasMass)) #mean of abs(velocity) in x direction  wolframalpha>> integrate x/sqrt(pi)*sqrt(c)*e^(-x^2*c) from 0 to infinity
timestep = 1/(10*4*d_ref**2 * particleDensity*np.sqrt(np.pi * kB * T_ref / gasMass)*(gasTemperature/T_ref)*(1-w)) # Bird1994 4.64
print(timestep, cellLength/(4.0*(VxMean+ax*vT)))
timestep = min(timestep, cellLength/(4.0*(VxMean+ax*vT)))

# only relevant for open boundaries:
# number of particles entering the simulation volume per timestep from different direction
inflowFast = particleDensity*edgeLength/2*edgeLength/2*timestep*vT/(2.0*np.sqrt(np.pi)) * (np.exp(-ax*ax) + ax*np.sqrt(np.pi)*(1.0 + sc.erf(ax)))
inflowSlow = particleDensity*edgeLength/2*edgeLength/2*timestep*vT/(2.0*np.sqrt(np.pi)) * (np.exp(-ax*ax) - ax*np.sqrt(np.pi)*(1.0 + sc.erf(-ax)))
inflowNormal = particleDensity*edgeLength/2*edgeLength*timestep*vT/(2.0*np.sqrt(np.pi))
inflowTotal = inflowFast + inflowSlow + 2*inflowNormal

# define the size of the particle reservoir so that there are always enough particles for the open boundaries to flow inside the volume
partVol = 0.6#975#(numParticles - 35*inflowTotal)/numParticles# part of gas atoms in the Volume, rest is in the reservoir
partVolAdjusted = partVol*simVol /(simVol - bodyVol)
partVolAdjusted = 1
partVol = 1
numSim = numParticles/(numGas*partVolAdjusted)







vRel = np.sqrt(16.0*kB*gasTemperature/(gasMass*np.pi)) #averare relative speed of the gas atoms
vRelMax = 7.0*vRel + ax*vT # the maximum relative speed is given by this constant. Otherwise you would need to measure it during the simulation

crossSection = sigma_ref # simulation particles cross section

mCandParameter = 0.5*numSim*sigma_ref*vRelMax*timestep # number of collision candidates choosen each timestep

# constant for the variable hard sphere (VHS) model
sigma_T_constant = d_ref*d_ref*np.pi*((4.0*kB*T_ref)/gasMass)**(w-0.5) / ((sc.gamma(2.5-w)) * (sigma_ref*vRelMax))

print("Kn = ", Kn, " ax = ", ax, "partVol = ", partVol, " num outside = ", (1-partVol)*100)

particleDensity *= simVol/(simVol - bodyVol)
collRate = 4*d_ref**2*particleDensity*np.sqrt(np.pi * kB * T_ref/gasMass)*(gasTemperature/T_ref)**(1-w)

print("Expected collisions per timestep",  0.5*((simVol - bodyVol)/simVol)*simVol*collRate*particleDensity/numSim*timestep)
print("Expected collisions per timestep",  0.5*((simVol - bodyVol)/simVol)*simVol/numSim* (rho/gasMass)**2 * (sigma_ref*vRel**(2*nu))*(2.0/np.sqrt(np.pi))*sc.gamma((1-2*nu+3)/2)*(4*gasTemperature*kB/gasMass)**((1-2*nu)/2))

# Open a outfile
outfile = open("simulation_constants.h", "w")
   
outfile.write("/*\n")
outfile.write(" * simulation_constants.h\n")
outfile.write(" *\n")
outfile.write(" *  Created on: Oct 27, 2015\n")
outfile.write(" *      Author: jordan\n")
outfile.write("*/\n\n")

outfile.write("#ifndef SIMULATION_CONSTANTS_H_\n")
outfile.write("#define SIMULATION_CONSTANTS_H_\n\n")

outfile.write("namespace simconstants")
outfile.write("\n{\n\n\n")

#define(outfile, "pi", np.pi, "pi")
define(outfile, "pi2", 2.0*np.pi, "2*pi")
define(outfile, "kB", kB, "boltzmann consatnt")
define(outfile, "m2kBdM", -2.0*kB/gasMass, "minus two times the boltzmann consatnt over mass of a gas particle")

outfile.write("\n\n")

define(outfile, "numSteps", numSteps, "number of timesteps") 
define(outfile, "generatorSeed", generatorSeed, "seed for cuda's XORWOW random generator")
define(outfile, "partVol", partVol, "part of gas atoms in the Volume, rest is in the reservoir")
define(outfile, "partVolAdjusted", partVolAdjusted, "part of gas atoms in the Volume, rest is in the reservoir. Adjusted the number considering the volume the sphere takes away")

outfile.write("\n\n")

define(outfile, "numGas", numGas, "Number of gas particles")
define(outfile, "gasDiameter", d_ref, "reference diameter of the gas molecules")
define(outfile, "gasMassMolecule", gasMass, "mass of the gas molecules")
define(outfile, "gasMass", gasMass*numSim, "mass of the gas molecules times number of molecules on particle represents")
outfile.write("\n\n")

define(outfile, "nz", nz, "Number of collision cells in z direction")
define(outfile, "nzM1", nz-1, "nz - 1")

define(outfile, "nx", nx, "Number of collision cells in x direction")
define(outfile, "nxM1", nx-1, "nx - 1")

define(outfile, "ny", ny, "Number of collision cells in y direction")
define(outfile, "nyM1", ny-1, "ny - 1")

define(outfile, "nxny", nx*ny, "Number of collision cells in x and y direction")


define(outfile, "numCells", numCells, "total number of collision cells")
define(outfile, "mCandParameter", mCandParameter, "parameter for calculating the number of potential collisions")

outfile.write("\n")
define(outfile, "nu", nu, "exponent for the variable hard sphere (VHS) model")
define(outfile, "twom2omega", 2.0-2.0*w, "exponent for the variable hard sphere (VHS) model")
define(outfile, "sigma_T_constant", sigma_T_constant, "constant for the variable hard sphere (VHS) model")

outfile.write("\n")

define(outfile, "particleRadius", particleRadius, "radius of the Sphere")
define(outfile, "particleRadiusSq", particleRadius*particleRadius, "particleRadius^2")
define(outfile, "Kn", Kn, "Knudsen number")

outfile.write("\n")
define(outfile, "gasTemperature", gasTemperature, "Temperature of the gas")
define(outfile, "accomodationCoefficient", accomodationCoefficient, "0.0 for specular reflection and 1.0 for diffuse reflection")



outfile.write("\n\n\n\n\n")


define(outfile, "ax", ax, "ratio of the gas flow speed over the most probable thermal velocity")

define(outfile, "vFlow",  ax*vT, "average gas flow velocity in x direction for epstein")

outfile.write("\n")

define(outfile, "edgeLength", edgeLength, "length of the edges for the simulation volume")
define(outfile, "halfEdgeLength", edgeLength/2.0, "half edge length of the simulation volume")
define(outfile, "dx", dx, "length of a collision cell in x-direction")
define(outfile, "invdx", 1/dx, "inverse length of a collision cell in x-direction")

define(outfile, "dz", dz, "length of a collision cell in z-direction")
define(outfile, "invdz", 1/dz, "inverse length of a collision cell in z-direction")

define(outfile, "dy", dy, "length of a collision cell in y-direction")
define(outfile, "invdy", 1/dy, "inverse length of a collision cell in y-direction")

outfile.write("\n")


outfile.write("\n")
define(outfile, "timestep", timestep,  "Size of one timestep is set as 1/4.7 * mean free time")

outfile.write("\n")
define(outfile, "vT", vT, "v_T = most probable speed of the maxwell boltzmann distribution")
define(outfile, "vTDsqrt2", vT/np.sqrt(2), "most probable speed / sqrt(2)")

outfile.write("\n")
outfile.write("\n")


inflowFast = rho/gasMass*edgeLength/2*edgeLength/2 /numSim*timestep*vT/(2.0*np.sqrt(np.pi)) * (np.exp(-ax*ax) + ax*np.sqrt(np.pi)*(1.0 + sc.erf(ax)))
inflowSlow = rho/gasMass*edgeLength/2*edgeLength/2 /numSim*timestep*vT/(2.0*np.sqrt(np.pi)) * (np.exp(-ax*ax) - ax*np.sqrt(np.pi)*(1.0 + sc.erf(-ax)))
inflowNormal = rho/gasMass*edgeLength/2*edgeLength/numSim*timestep*vT/(2.0*np.sqrt(np.pi))
inflowTotal = inflowFast + inflowSlow + 2*inflowNormal


define(outfile, "inflowFast", inflowFast, "number of particles entering each timestep with the flow")
define(outfile, "inflowSlow", inflowSlow, "number of particles entering each timestep against the flow")
define(outfile, "inflowNormal", inflowNormal, "number of particles entering each timestep without flow")
define(outfile, "inflowTotal", inflowTotal, "total number of particles entering each timestep")


outfile.write("\n")
outfile.write("\n")
define(outfile, "inflow_in_const1", ax*np.sqrt(np.pi)/(ax*np.sqrt(np.pi) + 1.0 + ax*ax), "constant for maxwell inflow distribution generation")
define(outfile, "inflow_in_const2", (ax*np.sqrt(np.pi) + 1.0)/(ax*np.sqrt(np.pi )+ 1.0 + ax*ax), "constant for maxwell inflow distribution generation")
define(outfile, "axSq", ax*ax, "constant for maxwell inflow distribution generation")

ax = -ax
z_a = 0.5*(ax - np.sqrt(ax*ax + 2.0))
b_a = ax - (1.0 - ax)*(ax - z_a)
inflow_ag_const3 = np.exp(- b_a*b_a)/(np.exp(-b_a*b_a) + 2.0*(ax - z_a)*(ax - b_a)*np.exp(-z_a*z_a))
define(outfile, "z_a", z_a, "constant for maxwell inflow distribution generation")
define(outfile, "b_a", b_a, "constant for maxwell inflow distribution generation")
define(outfile, "inflow_ag_const3", inflow_ag_const3, "constant for maxwell inflow distribution generation")

ax = -ax
define(outfile, "inflow_in_const3", 1.0/(2*ax*np.sqrt(np.pi) + 1.0), "constant for maxwell inflow distribution generation")

m_a = np.exp(-ax*ax) + ax*(1.0 + sc.erf(ax))


define2(outfile, "boundary", boundary, "/* boundary:\n                  1 for inflow boundaries \n                  2 for specular boundaries \n                  3 for diffusive boundaries \n                  */")

if ax != 0.0:
    define(outfile, "meanflow", 1, "1 for mean flow velocity != 0 and 0 for mean flow velocity == 0")
else:
    define(outfile, "meanflow", 0, "1 for mean flow velocity != 0 and 0 for mean flow velocity == 0")
    
outfile.write("\n\n")
outfile.write("} /* namespace simconstants */ \n")
outfile.write("#endif /* SIMULATION_CONSTANTS_H_ */")

outfile.flush()
outfile.close()















