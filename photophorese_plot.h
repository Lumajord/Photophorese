#ifndef PHOTOPHORESE_PLOT_H
#define PHOTOPHORESE_PLOT_H


#include "cuda_runtime.h"

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

/*
 * update the VTK plot with new positions and velocities
*/
extern "C"
void update_plot(	double4 *r_new,		// input: positions 
					double3 *vel_new,	// input: velocities
					unsigned int N);	// input: number of particles


/*
 * initializes a VTK plot
*/
extern "C"
void plot(	double a,			// input: radius of the test body
			double4 *atoms,		// input: positions of the particles
			unsigned int N);	// input: number of particles




#endif