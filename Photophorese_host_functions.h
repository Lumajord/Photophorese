
#ifndef __PHOTOPHORESE_HOST_FUNCTIONSH_H__
#define __PHOTOPHORESE_HOST_FUNCTIONSH_H__

#include "cuda_runtime.h"
#include "simulation_constants.h"


#include <random>

/*
 * initialize the gas within the simulation volume
*/
void init_cpu_gas(double4* h_pos,	// input: array for positions		// output: starting positions
				  double3* h_vel);	// input: array for velocities		// ourput: starting velocities


/*
 * calculates the number of collision cells that are not within the test sphere
*/
unsigned int calculateNumNonEmptyCells();


/*
 * calculates the inverse volume of the collisions cells
*/
void writeInverseCellVolume(double* cellInvVolume); // output: inverse volume of the collision cells

#endif