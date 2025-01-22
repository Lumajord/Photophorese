#ifndef __PHOTOPHORESE_DEVICE_FUNCTIONSH_H__
#define __PHOTOPHORESE_DEVICE_FUNCTIONSH_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <curand.h>
#include "curand_kernel.h"

#include <random>
#include <cmath>

typedef unsigned int uint;

#include "thrust/device_ptr.h"
#include "thrust/sort.h"


#include "simulation_constants.h"


inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ void operator-=(double3 &a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}


/*
 * update Normal:
 * generates poisson distributed numbers for the inflow boundary function and loads them intro constant device memory
 * @param gen:				marsenne twister random number generator
 * @param distInflowSlow:	poisson distribution for the number of particles entering the simulation volume against the flow direction
 * @param distInflowFast:	poisson distribution for the number of particles entering the simulation volume with the flow direction
 * @param distInflowNormal:	poisson distribution for the number of particles entering the simulation volume perpendicular to the flow direction
 * @param hNormal:			host array to store the random numbers
*/
void updateNormal(std::mt19937 &gen,
				  std::poisson_distribution<int> &distInflowSlow,
				  std::poisson_distribution<int> &distInflowFast,
				  std::poisson_distribution<int> &distInflowNormal,
				  unsigned int* hNormal);



/*
 * lets the expected number of particle pairs within a collision cell collide randomly
 */
void collide(double4		*dPos,				// input: sorted positions			output: positions after particle collisions
			 double3		*dVel,				// input: sorted velocities			output: velocities after particle collisions
			 double			*dInvCellVolume,	// input: inverse volume of the collision cells
             uint			*dCellStart,		// input: cell start index
             uint			*dCellEnd,			// input: cell end index
             curandState	*dStates,			// input: cuda random number states
			 uint*			dCountColl,			// output: number of particle collisions
			 uint			gridSize,
			 uint			blockSize);


/* 
 * returns the sum of a array
 */
double summation(int	n,			// input: array length
				 double *d_idata);	// input: data to sum




/*
 * rearrange particle data into sorted order, and find the start of each cell
 * in the sorted hash array
 */
void reorderDataAndFindCellStart(uint		*dCellStart,			// output: cell start index
                                 uint		*dCellEnd,				// output: cell end index
                                 double4	*&dSortedPos,			// input : work array
                                 double3	*&dSortedVel,			// input : work array
                                 uint		*dGridParticleHash,		// input : sorted grid hashes
                                 uint		*dGridParticleIndex,	// input : sorted particle indices
                                 double4	*&dOldPos,				// input : position array	output : sorted position array
                                 double3	*&dOldVel,				// input : velocity array	output : sorted velocity array
								 uint		gridSize,
								 uint		blockSize);


/*
 * calculate grid hash value for each particle and sorts the particle's hash values
 */
void calcHash(uint		*dGridParticleHash,		// output: sorted grid hashes
              uint		*dGridParticleIndex,	// output: sorted particle indices
              double4	*dPos,					// input : positions
			  uint		gridSize,
			  uint		blockSize);


/*
 * initializes N instances of the XORWOW random generator
 */
void initGenerators(curandState*	state,		// input: cuda random states		output: initialized cuda random states
					uint			N,			// input: array length
					uint			offset,		// input: offset for initializing random states
					uint			gridSize,
					uint			blockSize);


/*
 * advamces all particles with the euler integration scheme and does boundary checking and checks collision with the test sphere
 * also gives particles in the reservoir have a chance to enter the simulation volume
 */
void euler(double4		*pos,			// input: sorted positions			output: positions advanced one timestep
		   double3		*vel,			// input: sorted velocities			output: velocities advanced one timestep
		   double		*momentum,		// output: momentum transfer on the test body
		   uint			*cellStart,		// input: cell start index
		   uint			*cellEnd,		// input: cell end index
		   curandState	*globalState,	// input: cuda random states
		   uint			*dCountCollP,	// output: number of particles colliding with test body
		   uint			gridSize,
		   uint			blockSize);

/*
 * Sort the particle velocities into bins to visualize the velocity distribution
*/
void sortBins(double3*	dVel,			// input: positions
			  double4*	dPos,			// input: velocities
			  uint*		bins,			// output: bins representing the velocity distribution
			  uint		numBins,		// input: number of bins
			  uint		gridSize,
			  uint		blockSize);


#endif