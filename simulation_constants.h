/*
 * simulation_constants.h
 *
 *  Created on: Oct 27, 2015
 *      Author: jordan
*/

#ifndef SIMULATION_CONSTANTS_H_
#define SIMULATION_CONSTANTS_H_

namespace simconstants
{


#define pi2 6.283185307179586  // 2*pi
#define kB 1.38064852e-23  // boltzmann consatnt
#define m2kBdM -8267.356407185627  // minus two times the boltzmann consatnt over mass of a gas particle


#define numSteps 500000  // number of timesteps
#define generatorSeed 123  // seed for cuda's XORWOW random generator
#define partVol 1  // part of gas atoms in the Volume, rest is in the reservoir
#define partVolAdjusted 1  // part of gas atoms in the Volume, rest is in the reservoir. Adjusted the number considering the volume the sphere takes away


#define numGas 15000  // Number of gas particles
#define gasDiameter 2.92e-10  // reference diameter of the gas molecules
#define gasMassMolecule 3.34e-27  // mass of the gas molecules
#define gasMass 1.14803379673e-25  // mass of the gas molecules times number of molecules on particle represents


#define nz 32  // Number of collision cells in z direction
#define nzM1 31  // nz - 1
#define nx 64  // Number of collision cells in x direction
#define nxM1 63  // nx - 1
#define ny 32  // Number of collision cells in y direction
#define nyM1 31  // ny - 1
#define nxny 2048  // Number of collision cells in x and y direction
#define numCells 65536  // total number of collision cells
#define mCandParameter 1.29887657131e-24  // parameter for calculating the number of potential collisions

#define nu 0.0  // exponent for the variable hard sphere (VHS) model
#define twom2omega 1.0  // exponent for the variable hard sphere (VHS) model
#define sigma_T_constant 3.91466781189e-05  // constant for the variable hard sphere (VHS) model

#define particleRadius 1e-05  // radius of the Sphere
#define particleRadiusSq 1.0000000000000002e-10  // particleRadius^2
#define Kn 1000.0  // Knudsen number

#define gasTemperature 300.0  // Temperature of the gas
#define accomodationCoefficient 1.0  // 0.0 for specular reflection and 1.0 for diffuse reflection





#define ax 5.05  // ratio of the gas flow speed over the most probable thermal velocity
#define vFlow 7953.07971997  // average gas flow velocity in x direction for epstein

#define edgeLength 2.5e-05  // length of the edges for the simulation volume
#define halfEdgeLength 1.25e-05  // half edge length of the simulation volume
#define dx 3.90625e-07  // length of a collision cell in x-direction
#define invdx 2560000.0  // inverse length of a collision cell in x-direction
#define dz 3.90625e-07  // length of a collision cell in z-direction
#define invdz 2560000.0  // inverse length of a collision cell in z-direction
#define dy 3.90625e-07  // length of a collision cell in y-direction
#define invdy 2560000.0  // inverse length of a collision cell in y-direction


#define timestep 1.10450837084e-11  // Size of one timestep is set as 1/4.7 * mean free time

#define vT 1574.86727128  // v_T = most probable speed of the maxwell boltzmann distribution
#define vTDsqrt2 1113.59932699  // most probable speed / sqrt(2)


#define inflowFast 52.7054587482  // number of particles entering each timestep with the flow
#define inflowSlow 4.60185386816e-13  // number of particles entering each timestep against the flow
#define inflowNormal 5.88829125184  // number of particles entering each timestep without flow
#define inflowTotal 64.4820412518  // total number of particles entering each timestep


#define inflow_in_const1 0.252469268961  // constant for maxwell inflow distribution generation
#define inflow_in_const2 0.280675314845  // constant for maxwell inflow distribution generation
#define axSq 25.502499999999998  // constant for maxwell inflow distribution generation
#define z_a -5.14714130054  // constant for maxwell inflow distribution generation
#define b_a -5.63770486827  // constant for maxwell inflow distribution generation
#define inflow_ag_const3 0.0422622087791  // constant for maxwell inflow distribution generation
#define inflow_in_const3 0.0529050594166  // constant for maxwell inflow distribution generation
#define boundary 3/* boundary:
                  1 for inflow boundaries 
                  2 for specular boundaries 
                  3 for diffusive boundaries 
                  */
#define meanflow 1  // 1 for mean flow velocity != 0 and 0 for mean flow velocity == 0


} /* namespace simconstants */ 
#endif /* SIMULATION_CONSTANTS_H_ */