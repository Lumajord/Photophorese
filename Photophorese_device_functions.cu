#include "Photophorese_device_functions.h"


__constant__  unsigned int dNormal[4]; // constant device memory array to store numbers for the inflow boundary function

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
				  unsigned int* hNormal)
{

	hNormal[0] = distInflowSlow(gen);
	hNormal[1] = distInflowFast(gen);
	hNormal[2] = distInflowNormal(gen);
	hNormal[3] = distInflowNormal(gen);

	cudaMemcpyToSymbol (dNormal, hNormal, 4*sizeof(unsigned int) );

}






/*
 * helper function for the reduce6 kernel taken from the cuda samples
*/
#if (__CUDA_ARCH__ >= 300 )
/*
 * shuffle down function to shuffle down double3 elements
*/
__device__ inline
double3 __shfl_down(double3 var, unsigned int srcLane, int width=32) {

	var.x = __shfl_down(var.x, srcLane);
	var.y = __shfl_down(var.y, srcLane);
	var.z = __shfl_down(var.z, srcLane);

	return var;
}
#endif

/*
	This Reduce algorithm is taken from NVIDIA cuda samples.

    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/

__global__
void reduce6(double *g_idata, double *g_odata, int n)
{
    extern __shared__ double sdata[];    // blockSize + 1 elements

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    int gridSize = blockSize*2*gridDim.x;

    double mySum = 0.0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
    	if(tid < 256)
    	{
    		sdata[tid] = mySum = mySum + sdata[tid + 256];
    	}
    	__syncthreads();
    }



    if (blockSize >= 256)
    {
    	if(tid < 128)
    	{
    		sdata[tid] = mySum = mySum + sdata[tid + 128];
    	}

    	__syncthreads();
    }



    if (blockSize >= 128)
    {
    	if(tid <  64)
    	{
    		sdata[tid] = mySum = mySum + sdata[tid +  64];
    	}

    	__syncthreads();
    }


#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if(tid < 32)
    {
		if (blockSize >=  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 32];
		}

		__syncthreads();

		if (blockSize >=  32)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 16];
		}

		__syncthreads();

		if (blockSize >=  16)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  8];
		}

		__syncthreads();

		if (blockSize >=   8)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  4];
		}

		__syncthreads();

		if (blockSize >=   4)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  2];
		}

		__syncthreads();

		if (blockSize >=   2)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  1];
		}

		__syncthreads();
    }
#endif


    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}


/*
 * helper function for the reduce6 kernel taken from the cuda samples
*/
unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction kernel
// For kernel 6, we observe the maximum specified number of blocks, because each thread
// in that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);

    blocks = min(maxBlocks, blocks);
}


////////////////////////////////////////////////////////////////////////////////
// This function performs a summation of the input data and returns the result, input data is set to 0 afterwards
////////////////////////////////////////////////////////////////////////////////
double summation(int  n, double *d_idata)
{


	int maxThreads = 512;
    int maxBlocks = 1024;
    int numBlocks = 0;
    int numThreads = 0;

	double gpu_result = 0.0;


    getNumBlocksAndThreads(n, maxBlocks, maxThreads, numBlocks, numThreads);

	// execute the kernel
    int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(double) : numThreads * sizeof(double);
	reduce6<<< numBlocks, numThreads, smemSize >>>(d_idata, d_idata, n);

	// sum partial block sums on GPU
	int s=numBlocks;

	while (s > 1)
	{
		getNumBlocksAndThreads(s, maxBlocks, maxThreads, numBlocks, numThreads);
		smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(double) : numThreads * sizeof(double);
		reduce6<<< numBlocks, numThreads, smemSize >>>(d_idata, d_idata, s);

		s = (s + (numThreads*2-1)) / (numThreads*2);
	}


    // copy final sum from device to host
    cudaMemcpy(&gpu_result, d_idata, sizeof(double), cudaMemcpyDeviceToHost);

	cudaMemset(d_idata, 0, n*sizeof(double));
	cudaDeviceSynchronize();

    return gpu_result;
}




// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint    *cellStart,			// output: cell start index
                                  uint    *cellEnd,				// output: cell end index
                                  double4 *sortedPos,			// output: sorted positions
                                  double3 *sortedVel,			// output: sorted velocities
                                  uint    *gridParticleHash,	// input : sorted grid hashes
                                  uint    *gridParticleIndex,	// input : sorted particle indices
                                  double4 *oldPos,				// input : sorted position array
                                  double3 *oldVel)				// input : sorted velocity array
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (tid < numGas)
    {
        hash = gridParticleHash[tid];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (tid > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[tid-1];
        }
    }

    __syncthreads();

    if (tid < numGas)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (tid == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = tid;

            if (tid > 0)
                cellEnd[sharedHash[threadIdx.x]] = tid;
        }

        if (tid == numGas - 1)
        {
            cellEnd[hash] = tid + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[tid];
        sortedPos[tid] = oldPos[sortedIndex];
        sortedVel[tid] = oldVel[sortedIndex];


    }

}



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
								 uint		blockSize)
{

    // set all cells to empty
    cudaMemset(dCellStart, 0xffffffff, (numCells+1)*sizeof(uint));

    uint sMem = sizeof(uint)*(blockSize+1);
    reorderDataAndFindCellStartD<<<gridSize, blockSize, sMem>>>(
        dCellStart,
        dCellEnd,
        dSortedPos,
        dSortedVel,
        dGridParticleHash,
        dGridParticleIndex,
        dOldPos,
        dOldVel);


	cudaDeviceSynchronize();

	double3* dSwapVel;			 // used to swap gpu and gpu_sorted pointers
	dSwapVel = dOldVel;
	dOldVel = dSortedVel;
	dSortedVel = dSwapVel;

	double4* dSwap;				 // used to swap gpu and gpu_sorted pointers
	dSwap = dOldPos;
	dOldPos = dSortedPos;
	dSortedPos = dSwap;


}



































// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash2(double4 pos)	// input : grid position
											// return: flattened grid position
{

	if(pos.w == 0.0)
	{

		int3 gridPos;
		gridPos.x = (pos.x+halfEdgeLength)*invdx;
		gridPos.y = pos.y*invdy;
		gridPos.z = pos.z*invdz;

		gridPos.x = min(gridPos.x, nxM1);
		gridPos.y = min(gridPos.y, nyM1);
		gridPos.z = min(gridPos.z, nzM1);

		gridPos.x = max(gridPos.x, 0);
		gridPos.y = max(gridPos.y, 0);
		gridPos.z = max(gridPos.z, 0);

		return gridPos.z*nxny + gridPos.y*nx +  gridPos.x;
	}
	else
	{
		return numCells;
	}

}




// calculate grid hash value for each particle
__global__
void calcHashD(uint    *gridParticleHash,	// output: sorted grid hashes
               uint    *gridParticleIndex,	// output: sorted particle indices
               double4 *pos)				// input : positions
{
	for(uint index = blockIdx.x * blockDim.x + threadIdx.x;
		index < numGas;
		index += blockDim.x * gridDim.x)
	{

		// store grid hash and particle index
		gridParticleHash[index] = calcGridHash2(pos[index]);
		gridParticleIndex[index] = index;

	}

}



/*
 * calculate grid hash value for each particle and sorts the particle hash values
 */
void calcHash(uint		*dGridParticleHash,		// output: sorted grid hashes
              uint		*dGridParticleIndex,	// output: sorted particle indices
              double4	*dPos,					// input : positions
			  uint		gridSize,
			  uint		blockSize)
{


	calcHashD<<<gridSize, blockSize, 0>>>(dGridParticleHash,
										  dGridParticleIndex,
										  dPos);

	cudaDeviceSynchronize();

	thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
					thrust::device_ptr<uint>(dGridParticleHash + numGas),
					thrust::device_ptr<uint>(dGridParticleIndex));

	cudaDeviceSynchronize();

}























/*
 * initGenerators device Kernel
 * initializes N instances of the XORWOW random generator
 */
__global__
void initGeneratorsD(	curandState* state, // input: random state array			output: random state array with initialized random states
						uint N,				// input: number of random states
						uint offset)		// input: offset for the random states
{


	for(uint index = blockDim.x*blockIdx.x + threadIdx.x;
		index < N;
		index += blockDim.x * gridDim.x)
	{
		curand_init (generatorSeed, index + offset, 0, &state[index] );
	}
}



/*
 * initGenerators C wrapper
 * initializes N instances of the XORWOW random generator
 */
void initGenerators(	curandState* state,	// input: random state array				output: random state array with initialized random states
						uint N,				// input: number of random states
						uint offset,		// input: offset for the random states
						uint gridSize,		// input: grid size
						uint blockSize)		// input: block size
{

	// call the cuda Kernel to initialize the random states
	initGeneratorsD <<<gridSize, blockSize, 0>>>(state, N, offset);


}


















/*
 * calculates the temperature based on the particles position on the shpere
*/
__device__ inline
double calc_temp(	double4 pos)	// input: position of the particle that is colliding with the testbody
{
		return 300 - pos.x*15.0;
}



/*
 * diffuse_reflection:
 * algorithm is taken from: Ching Shen, Rarefied Gas Dynamics	Fundamentals, Simulations and Micro Flows
 * returns a new velocity for the particle after a diffuse reflection with the test body
*/
__device__
double3 diffuse_reflection(double4 n,			// input: position of the particle colliding with the test body
						   curandState &state)	// input: cuda random state
{

	// normalize the position of the particle
	double invLn = 1.0/sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
	n.x *= invLn;
	n.y *= invLn;
	n.z *= invLn;

	// generate first tangential vector t1 = n x e_y  if  n  is similar to e_z else  t1 = n x e_z
	double3 t1;

	if(n.z*n.z > particleRadiusSq)
	{
		t1.x = -n.z;
		t1.y =  0.0;
		t1.z =  n.x;
	}
	else
	{
		t1.x =  n.y;
		t1.y = -n.x;
		t1.z = 0.0;
	}

	// normalize the first tangential vector
	double invLt1 = 1.0/sqrt(t1.x*t1.x + t1.y*t1.y + t1.z*t1.z);
	t1.x *= invLt1;
	t1.y *= invLt1;
	t1.z *= invLt1;

	// generate second tangetial vector t2 = n x t1
	double3 t2;
	t2.x = n.y*t1.z - n.z*t1.y;
	t2.y = n.z*t1.x - n.x*t1.z;
	t2.z = n.x*t1.y - n.y*t1.x;

	// normalize the second tangential vector
	double invLt2 = 1.0/sqrt(t2.x*t2.x + t2.y*t2.y + t2.z*t2.z);
	t2.x *= invLt2;
	t2.y *= invLt2;
	t2.z *= invLt2;


	double temp = calc_temp(n);	// calculate the temperature of the test body's surface at the position of the particle


	// generate a diffuse reflected velocity for the particle 
	double vn = sqrt(m2kBdM*temp*log(curand_uniform_double(&state)));

	double phi = curand_uniform_double(&state)*pi2;

	double v1;
	double v2;

	sincos(phi, &v1, &v2);

	double v = sqrt(m2kBdM*temp*log(curand_uniform_double(&state)));

	v1 *= v;
	v2 *= v;

	return make_double3(n.x*vn + t1.x*v1 + t2.x*v2, n.y*vn + t1.y*v1 + t2.y*v2, n.z*vn + t1.z*v1 + t2.z*v2);
}



/*
 * specular_reflection:
 * specularly reflects the velocity at the particle's impact point on the test body
*/
__device__
double3 specular_reflection(double4 n,	// input: position of the particle colliding with the test body
							double3 v)	// input: velocity of the particle
{
	double norm = 1.0/sqrt(n.x*n.x + n.y*n.y + n.z*n.z); // 1/sqrt(|n * n|)
	n.x *= norm;
	n.y *= norm;
	n.z *= norm;
	double nv = v.x*n.x + v.y*n.y + v.z*n.z; // n*v
	v.x = v.x - 2.0*nv*n.x;
	v.y = v.y - 2.0*nv*n.y;
	v.z = v.z - 2.0*nv*n.z;

	return make_double3(v.x, v.y, v.z);
}





/*
 * collision_gas_sphere:
 * advances the particle in time with the euler-integrator and checks for a collision between particle and testbody,
 * if there is a collision between the particle and the spherical test body it calculates the point of impact,
 * and scatters the gas atom at this point
 */
__device__
void collision_gas_sphere(double &momentum,		// input: storage for the momentum transfer					output: updated momentum transfer 
							double4	  &p1,		// input: position of the particle							output: updated position
							double	  t,		// input: part of the timestep
							double3	 &vel,		// input: velocity of the particle							output: updated velocity
							uint	*countColl,	// input: number of particles colliding with the testbody	output: incremented count in case of collision between particle and test body
							curandState &state)	// input: random state
{
	/*
	 * this algorithm performs a simple sphere - line segment collision test
	 * see Christer Ericson: Real-Time Collision Detection
	*/

	double3 d; // distance the particle moves  d = p2 - p1
	d.x = t*vel.x*timestep;
	d.y = t*vel.y*timestep;
	d.z = t*vel.z*timestep;

	// calculate wether the particle collides with the sphere
	double a = d.x*d.x +  d.y*d.y +  d.z*d.z;
	double b = p1.x*d.x + p1.y*d.y + p1.z*d.z;
	double c = fma(p1.x, p1.x, p1.y*p1.y) + fma(p1.z,p1.z, -particleRadiusSq); // c = p1^2 - R^2

	if(c > 0.0 && b > 0.0)
	{
		// the particle moves the full distance if it does not collide with the sphere
		p1.x += d.x;
		p1.y += d.y;
		p1.z += d.z;
		return;
	}

	double discr = b*b - a*c;

	if(discr < 0.0)
	{
		// the particle moves the full distance if it does not collide with the sphere
		p1.x += d.x;
		p1.y += d.y;
		p1.z += d.z;
		return;
	}

	t = (-b - sqrt(discr))/a; // reuse of variable t

	if(t < 0.0 || t > 1.0)
	{
		// the particle moves the full distance if it does not collide with the sphere
		p1.x += d.x;
		p1.y += d.y;
		p1.z += d.z;
		return;
	}

	// if the code reaches this point, then the particle is colliding with the test sphere 


	momentum = gasMass*vel.x; // momentum transfer from particle to sphere on particle impact on sphere

	// the particle only moves until it collides with the sphere
	p1.x = fma(t, d.x, p1.x);
	p1.y = fma(t, d.y, p1.y);
	p1.z = fma(t, d.z, p1.z);

	// calculate new particle velocity
	if(curand_uniform_double(&state) <= accomodationCoefficient)
	{
		vel = diffuse_reflection(p1, state);
	}
	else
	{
		vel = specular_reflection(p1, vel);
	}

	// the particle moves for the rest of the timestep with the new velocity
	p1.x += timestep*(1.0-t)*vel.x;
	p1.y += timestep*(1.0-t)*vel.y;
	p1.z += timestep*(1.0-t)*vel.z;

	momentum -= gasMass*vel.x; // momentum transfer from particle to sphere on particle reflection from sphere

	atomicAdd(&countColl[0], 1);

	return;
}












/*
 * imposes the boundary conditions on the particles
 * depending on the constant "boundary" you can choose between:
 * boundary = 1 : open boundaries ( particle is moved to the reservoir if it leaves the volume )
 * boundary = 2 : specular boundaries ( particles are reflected specularly at the edges of the volume )
 * boundary = 3 : diffusive boundaries ( particles are reflected diffusively at the edges of the volume )
 */
__device__
void boundaries(	double4 &pos,	// input: particle position		output: updated particle position
					double3 &vel	// input: particle velocity		output: updated particle velocity
#if boundary == 3
					,curandState &state	// input: cuda random state
#endif
				)
{

	if(pos.x < -halfEdgeLength)
	{
#if boundary == 1
		pos.w = 1.0; // move particle to reservoir
#endif


#if boundary == 2
		// reflect particle on wall
		pos.x = -edgeLength-pos.x;
		vel.x = -vel.x;
#endif


#if boundary == 3

		// generate the velocity for a diffuse reflected particle in x-direction (see Ching Shen, Rarefied Gas Dynamics	Fundamentals, Simulations and Micro Flows)
		vel.x = vT*sqrt(-log(curand_uniform_double(&state)));
		double phi = curand_uniform_double(&state)*pi2;
		double v = vT*sqrt(-log(curand_uniform_double(&state)));
		sincos(phi, &vel.y, &vel.z);
		vel.y *= v;
		vel.z *= v;

		
												
#endif
	}

	if(pos.x > halfEdgeLength)
	{
#if boundary == 1
		pos.w = 1.0; // move particle to reservoir
#endif
		

#if boundary == 2
		// reflect particle on wall
		pos.x = edgeLength-pos.x;
		vel.x = -vel.x;
#endif


#if boundary == 3

		// generate the velocity for a diffuse reflected particle in -x-direction (see Ching Shen, Rarefied Gas Dynamics	Fundamentals, Simulations and Micro Flows)
		vel.x = -vT*sqrt(-log(curand_uniform_double(&state)));
		double phi = curand_uniform_double(&state)*pi2;
		double v = vT*sqrt(-log(curand_uniform_double(&state)));
		sincos(phi, &vel.y, &vel.z);
		vel.y *= v;
		vel.z *= v;


#endif
	}




	if(pos.y < 0.0)
	{
#if boundary == 1
		// reflect particle on wall
		pos.y = -pos.y;
		vel.y = -vel.y;
#endif
		

#if boundary == 2
		// reflect particle on wall
		pos.y = -pos.y;
		vel.y = -vel.y;
#endif


#if boundary == 3


		// generate the velocity for a diffuse reflected particle in y-direction (see Ching Shen, Rarefied Gas Dynamics	Fundamentals, Simulations and Micro Flows)
		vel.y = vT*sqrt(-log(curand_uniform_double(&state)));
		double phi = curand_uniform_double(&state)*pi2;
		double v = vT*sqrt(-log(curand_uniform_double(&state)));
		sincos(phi, &vel.x, &vel.z);
		vel.x *= v;
		vel.z *= v;



#endif
	}

	if(pos.y > halfEdgeLength)
	{
#if boundary == 1
		pos.w = 1.0; // move particle to reservoir
#endif
		

#if boundary == 2
		// reflect particle on wall
		pos.y = edgeLength-pos.y;
		vel.y = -vel.y;
#endif


#if boundary == 3

		// generate the velocity for a diffuse reflected particle in -y-direction (see Ching Shen, Rarefied Gas Dynamics	Fundamentals, Simulations and Micro Flows)
		vel.y = -vT*sqrt(-log(curand_uniform_double(&state)));
		double phi = curand_uniform_double(&state)*pi2;
		double v = vT*sqrt(-log(curand_uniform_double(&state)));
		sincos(phi, &vel.x, &vel.z);
		vel.x *= v;
		vel.z *= v;


#endif
	}




	if(pos.z < 0.0)
	{
#if boundary == 1
		// reflect particle on wall
		pos.z = -pos.z;
		vel.z = -vel.z;
#endif
		

#if boundary == 2
		// reflect particle on wall
		pos.z = -pos.z;
		vel.z = -vel.z;
#endif


#if boundary == 3


		// generate the velocity for a diffuse reflected particle in z-direction (see Ching Shen, Rarefied Gas Dynamics	Fundamentals, Simulations and Micro Flows)
		vel.z = vT*sqrt(-log(curand_uniform_double(&state)));
		double phi = curand_uniform_double(&state)*pi2;
		double v = vT*sqrt(-log(curand_uniform_double(&state)));
		sincos(phi, &vel.x, &vel.y);
		vel.x *= v;
		vel.y *= v;


#endif
	}

	if(pos.z > halfEdgeLength)
	{
#if boundary == 1
		pos.w = 1.0; // move particle to reservoir
#endif
		

#if boundary == 2
		// reflect particle on wall
		pos.z = edgeLength-pos.z;
		vel.z = -vel.z;
#endif


#if boundary == 3


		// generate the velocity for a diffuse reflected particle in -z-direction (see Ching Shen, Rarefied Gas Dynamics	Fundamentals, Simulations and Micro Flows)
		vel.z = -vT*sqrt(-log(curand_uniform_double(&state)));
		double phi = curand_uniform_double(&state)*pi2;
		double v = vT*sqrt(-log(curand_uniform_double(&state)));
		sincos(phi, &vel.x, &vel.y);
		vel.x *= v;
		vel.y *= v;


#endif
	}

	return;

}








#if boundary == 1

#if meanflow != 0

/*
 * maxwell_inflow_ag (recommended for -0.4 < ax < 1.3):
 * (see Garcia & Wagner 2006	Generation of the Maxwellian inflow distribution)
 * returns a random velocity from the maxwellian inflow distribution for a <= 0
 * where a is flow speed divided by average thermal speed
 *
 * @param state :	cuda random generator state
 */
__device__
double maxwell_inflow_ag(curandState &state)
{

		start:
		double z_ = -sqrt(axSq - log(curand_uniform_double(&state)));

		if((-ax-z_)/(-z_) > curand_uniform_double(&state))
		{
			return z_;
		}
		else
		{
			goto start;
		}
}





/*
 * maxwell_inflow_in (recommended for -0.4 < ax < 1.3):
 * (see Garcia & Wagner 2006	Generation of the Maxwellian inflow distribution)
 * returns a random velocity from the maxwellian inflow distribution for a > 0
 * where a is flow speed divided by average thermal speed
 *
 * @param state :	cuda random generator state
 */
__device__
double maxwell_inflow_in(curandState &state)
{
	start:
	double u = curand_uniform_double(&state);

	if(inflow_in_const1 > u)
		return -abs(curand_normal_double(&state))*0.7071067811865475244;

	else if(inflow_in_const2 > u)
		return -sqrt(-log(curand_uniform_double(&state)));

	else
	{
		double z_ = ax*(1.0-sqrt(curand_uniform_double(&state)));
		if(exp(-z_*z_) > curand_uniform_double(&state))
		{
			return z_;
		}
		else
		{
			goto start;
		}
	}
	
}




/*
 * maxwell_inflow_ag2 (recommended for high speeds ax):
 * (see Garcia & Wagner 2006	Generation of the Maxwellian inflow distribution)
 * returns a random velocity from the maxwellian inflow distribution for a < 0
 * where a is flow speed divided by average thermal speed
 *
 * @param state :	cuda random generator state
 */
__device__
double maxwell_inflow_ag2(curandState &state)
{

	while (true)
	{
	if(inflow_ag_const3 > curand_uniform_double(&state))
		{
			double z_ = -sqrt(b_a*b_a - log(curand_uniform_double(&state)));
			if((z_ + ax)/z_ > curand_uniform_double(&state))
				return z_;
		}
		else
		{
			double z_ = b_a + (-ax - b_a)*curand_uniform_double(&state);
			if((-ax - z_)/(-ax - z_a)*exp(z_a*z_a - z_*z_) > curand_uniform_double(&state))
				return z_;
		}
	}

}





/*
 * maxwell_inflow_in2 (recommended for high speeds ax):
 * (see Garcia & Wagner 2006	Generation of the Maxwellian inflow distribution)
 * returns a random velocity from the maxwellian inflow distribution for a >= 0
 * where a is flow speed divided by average thermal speed
 *
 * @param state :	cuda random generator state
 */
__device__
double maxwell_inflow_in2(curandState &state)
{

	double z_;

	while(true)
	{
		if(inflow_in_const3 > curand_uniform_double(&state))
			z_ = - sqrt(-log(curand_uniform_double(&state)));
		else
			z_ = curand_normal_double(&state)*0.7071067811865475244;

		if((ax - z_)/ax > curand_uniform_double(&state))
			return z_;
	}
	
}

#endif

/*
 * enter_cube:
 * puts particles from the reservoir on the edges of the simulation volume with a maxwell inflow velocity
*/
__device__
double enter_cube(uint		index,
					 double4		&oldPos,			// input: sorted positions
					 double3		&vel,               // input: sorted velocities
					 uint			*cellStart,			// input: cell start index
					 uint			*cellEnd,			// input: cell end index
					 curandState	&state)				// input: cuda random number state
{
    
    uint startIndex = cellStart[numCells];	// get start of bucket for reservoir cell

    if (startIndex == 0xffffffff)      // cell is empty
    {
    	printf("Warning: reservoir empty\n");
    	return 0.0;
    }

	uint endIndex = cellEnd[numCells];	// get end of bucket for this cell

	uint cellSize = endIndex - startIndex;	// numbers of particles in this cell

	index -= startIndex;	// shift the index of the particles entering the volume to zero
							// this way it is easier to decide from which side the particle enters the volume


	if(cellSize < inflowTotal)	// check whether enough particles are inside the reservoir
	{
		if(index == 0)
		{
    		printf("Warning: too few particles in reservoir\n");
		}

		return 0.0;
	}

	uint maxIndex = dNormal[0]; // the number of particles entering the volume are normal distributed around the expected value
								// this is important to avoid boundary effects (see Tysander & Garcia 2005	Non-equilibrium behaviour of equilibrium reservoirs in molecular simulations)

	// put new particles on each open edge of the volume and perform a random sized timestep into the volume
	// the generation of the values for the particles entering the volume is taken from  Garcia & Wagner 2006 Generation of the Maxwellian inflow distribution

	if(index < maxIndex)	// inflow from -x-direction
	{

#if meanflow == 0
		// generate velocity without mean flow speed
		vel.x = -vT*sqrt(-log(curand_uniform_double(&state)));
		vel.y = curand_normal_double(&state)*vTDsqrt2;
		vel.z = curand_normal_double(&state)*vTDsqrt2;
#else
		// generate velocity with mean flow speed
		vel.x = fma(maxwell_inflow_ag(state), vT, vFlow);
		vel.y = curand_normal_double(&state)*vTDsqrt2;
		vel.z = curand_normal_double(&state)*vTDsqrt2;
#endif

		// particle is evenly distributed on the wall of the simulation volume
		oldPos.x = halfEdgeLength;
		oldPos.y = curand_uniform_double(&state)*halfEdgeLength;
		oldPos.z = curand_uniform_double(&state)*halfEdgeLength;

		oldPos.w = 0.0;	// 0.0 means the particle is now inside the volume
		return curand_uniform_double(&state);	// particle performs a random sized timestep upon entering the volume

	}


	maxIndex += dNormal[1];

	if(index < maxIndex)	// inflow from +x-direction
	{

#if meanflow == 0
		vel.x = vT*sqrt(-log(curand_uniform_double(&state)));
		vel.y = curand_normal_double(&state)*vTDsqrt2;
		vel.z = curand_normal_double(&state)*vTDsqrt2;
#else
		vel.x = fma(maxwell_inflow_in(state), -vT, vFlow);
		vel.y = curand_normal_double(&state)*vTDsqrt2;
		vel.z = curand_normal_double(&state)*vTDsqrt2;
#endif
		// particle is evenly distributed on the wall of the simulation volume
		oldPos.x = -halfEdgeLength;
		oldPos.y = curand_uniform_double(&state)*halfEdgeLength;
		oldPos.z = curand_uniform_double(&state)*halfEdgeLength;

		oldPos.w = 0.0;	// 0.0 means the particle is now inside the volume
		return curand_uniform_double(&state);	// particle performs a random sized timestep upon entering the volume

	}

	maxIndex += dNormal[2];

	if(index < maxIndex)	// inflow from -y-direction
	{

#if meanflow == 0

		vel.x = curand_normal_double(&state)*vTDsqrt2;
		vel.y = -vT*sqrt(-log(curand_uniform_double(&state)));
		vel.z = curand_normal_double(&state)*vTDsqrt2;
#else
		vel.x = fma(curand_normal_double(&state), vTDsqrt2, vFlow);
		vel.y = -vT*sqrt(-log(curand_uniform_double(&state)));
		vel.z = curand_normal_double(&state)*vTDsqrt2;
#endif

		// particle is evenly distributed on the wall of the simulation volume
		oldPos.x = fma(curand_uniform_double(&state), edgeLength, -halfEdgeLength);
		oldPos.y = halfEdgeLength;
		oldPos.z = curand_uniform_double(&state)*halfEdgeLength;

		oldPos.w = 0.0;	// 0.0 means the particle is now inside the volume
		return curand_uniform_double(&state);	// particle performs a random sized timestep upon entering the volume

	}



	maxIndex += dNormal[3];

	if(index < maxIndex) // inflow from -z-direction
	{

#if meanflow == 0
		vel.x = curand_normal_double(&state)*vTDsqrt2;
		vel.y = curand_normal_double(&state)*vTDsqrt2;
		vel.z = -vT*sqrt(-log(curand_uniform_double(&state)));
#else
		vel.x = fma(curand_normal_double(&state), vTDsqrt2, vFlow);
		vel.y = curand_normal_double(&state)*vTDsqrt2;
		vel.z = -vT*sqrt(-log(curand_uniform_double(&state)));
#endif

		// particle is evenly distributed on the wall of the simulation volume
		oldPos.x = fma(curand_uniform_double(&state), edgeLength, -halfEdgeLength);
		oldPos.y = curand_uniform_double(&state)*halfEdgeLength;
		oldPos.z = halfEdgeLength;

		oldPos.w = 0.0;	// 0.0 means the particle is now inside the volume
		return curand_uniform_double(&state);	// particle performs a random sized timestep upon entering the volume

	}

	return 0.0;

}



#endif




__global__
void eulerD(double4		*pos,			// input: sorted position			output: updated positions
		   double3		*vel,			// input: sorted velocities			output: updated velocities
		   double		*momentum,		// output: momentum transfer on test body
		   uint			*cellStart,		// input: cell start index
		   uint			*cellEnd,		// input: cell end index
		   uint			*countColl,		// output: number of particles colliding with test body
		   curandState	*globalState)	// input: cuda random number state
{

	uint startId = blockDim.x*blockIdx.x + threadIdx.x;
	curandState st = globalState[startId];

	for(uint index = startId;
		index < numGas;
		index += blockDim.x * gridDim.x)
	{

		double4 position = pos[index];


		double3 velocity;
		double t = 1.0; // part of the timestep a particles moves
						// for particles inside the volume t = 1.0
						// for particles entering the volume t is a uniform random number in (0.0, 1.0]
		
#if boundary == 1
		if(position.w == 1.0) // check if particle is inside the reservoir
		{
			// particles inside the reservoir have a chance to enter the simulation volume

			t = enter_cube(index,
							position,
							vel[index],     
							cellStart,
							cellEnd,
							st);
		}
#endif

		if(position.w == 0.0) // if particle is inside the simulation volume check collision with sphere and move particel
							  // and check again if it is still inside the simulation volume
		{

			velocity = vel[index];

			collision_gas_sphere(momentum[index], position, t, velocity, countColl, st); // check collision with sphere and move particel

			boundaries(	position,
						velocity
#if boundary == 3
						,st
#endif
						);


		}

		pos[index] = position;
		vel[index] = velocity;
		

	}

	globalState[startId] = st;
}




/*
 * advamces all particles with the euler integration scheme and does boundary checking and checks collision with the test sphere
 * also gives particles in the reservoir a chance to enter the simulation volume
 */
void euler(double4		*dPos,			// input: sorted position			output: updated positions
		   double3		*dVel,			// input: sorted velocities			output: updated velocities
		   double		*dMomentum,		// output: momentum transfer on test body
		   uint			*dCellStart,	// input: cell start index
		   uint			*dCellEnd,		// input: cell end index
		   curandState	*dGlobalState,	// input: cuda random number state
		   uint			*countCollP,	// output: number of particles colliding with test body
		   uint			gridSize,
		   uint			blockSize)
{


	eulerD<<<gridSize, blockSize, 0>>>( dPos,
										dVel,
										dMomentum,
										dCellStart,
										dCellEnd,
										countCollP,
										dGlobalState);

	cudaDeviceSynchronize();


}
























/*
 * Sort the particle velocities into bins to visualize the velocity distribution
*/
__global__
void sortBinsD(double3* dVel, double4* dPos, uint* bins, uint numBins)
{
	for(uint index = blockDim.x*blockIdx.x + threadIdx.x;
	index < numGas;
	index += blockDim.x * gridDim.x)
	{
		if(index >= numGas)
			return;

		if(dPos[index].w == 1.0)
			return;

		double3 vel = dVel[index];



		double v = sqrt((vel.x*vel.x + vel.y*vel.y + vel.z*vel.z)*gasMassMolecule/kB);

		uint bin = uint(v+0.5);

		if(bin < numBins)
			atomicAdd(&bins[bin],1);
	}

}


/*
 * Sort the particle velocities into bins to visualize the velocity distribution
*/
void sortBins(double3* dVel, double4* dPos, uint* bins, uint numBins, uint gridSize, uint blockSize)
{

	sortBinsD<<<gridSize, blockSize, 0>>>(dVel, dPos, bins, numBins);

}









































// collide two spheres
__device__
void collideSpheres(double4      posA,    double4  posB,	// input : positions
					double3     &velA,    double3 &velB,	// in\out: velocities
					curandState &state,						// input : random state
					uint*countColl)
{

	double vRel = sqrt((velB.x-velA.x)*(velB.x-velA.x) + (velB.y-velA.y)*(velB.y-velA.y) + (velB.z-velA.z)*(velB.z-velA.z));
	double r = curand_uniform_double(&state);
	double p = sigma_T_constant*pow(vRel, twom2omega);

	if(p > 1.0)
	{
		printf("vRel = %f, p = %f", vRel, p);
		printf("	v1 = %f	%f	%f	%f", velA.x, velA.y, velA.z, posA.w);
		printf("	v2 = %f	%f	%f	%f\n", velB.x, velB.y, velB.z, posB.w);
		printf("Error: crossSectionRefDivCrossSectionVRelMax too big\n");
	}

	if(r < p)
	{

		vRel *= 0.5;
		atomicAdd(&countColl[0],1);
		
		// center of mass velocity
		velB.x = 0.5*(velB.x + velA.x);
		velB.y = 0.5*(velB.y + velA.y);
		velB.z = 0.5*(velB.z + velA.z);


		double cos_theta = fma(curand_uniform_double(&state), -2.0, 1.0);
		double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
		double phi = curand_uniform_double(&state)*pi2;

		sincos(phi, &velA.y, &velA.x);

		// 0.5*relative velocity
		velA.x *= sin_theta*vRel;
		velA.y *= sin_theta*vRel;
		velA.z = cos_theta*vRel;

		// v_B = v_cm - 0.5*v_r
		velB -= velA;

		// v_A = v_B + 2*0.5*v_r
		//	   = v_cm - 0.5*v_r + 2*0.5*v_r
		//     = v_cm + 0.5*v_r
		velA.x = fma(velA.x, 2.0, velB.x);
		velA.y = fma(velA.y, 2.0, velB.y);
		velA.z = fma(velA.z, 2.0, velB.z);

		return;
	}
	return;
}


// choose collision pairs within a cell and give them a chance to collide
__global__
void collideD(double4     *dPos,				// input: sorted positions
			  double3     *dVel,				// input: sorted velocities
			  double	  *dInvCellVolume,		// input: inverse cell volume of the collision cells
              uint        *cellStart,			// input: cell start index
              uint        *cellEnd,				// input: cell end index
              curandState *state,				// input: cuda random states
			  uint* countColl)					// output: number of particle collisions
{

	uint startId = blockDim.x*blockIdx.x + threadIdx.x;
	curandState st = state[startId];

	for(uint index = startId;
		index < numCells;
		index += blockDim.x * gridDim.x)
	{


		// get start of bucket for this cell
		uint startIndex = cellStart[index];
	
		if (startIndex != 0xffffffff)          // cell is not empty
		{

			// get end of bucket for this cell
			uint endIndex = cellEnd[index];

			// numbers of particles in this cell
			uint cellSize = endIndex - startIndex; 

			if(cellSize > 1)
			{

				// number of potential collinsions
				double m_cand = cellSize*(cellSize-1)*mCandParameter*dInvCellVolume[index];

				//if(index == 0)
				//	printf("%f %f %d\n", m_cand, crossSection, cellSize);
				while(m_cand >= 1.0)
				{
					m_cand -= 1.0;

					// indices of the collision candidates
					uint i = floor((1.0-curand_uniform_double(&st))*double(cellSize));
					uint j = floor((1.0-curand_uniform_double(&st))*double(cellSize));

					while(i == j) i = uint((1.0-curand_uniform_double(&st))*double(cellSize)); // if i == j : reroll i		

					i += startIndex;
					j += startIndex;


					if(i < 10)
					{
						
						uint id1 = calcGridHash2(dPos[i]);
						uint id2 = calcGridHash2(dPos[j]);

						//printf("id1 = %d	id2 = %d	i = %d	j = %d	start = %d	size = %d\n", id1, id2, i -startIndex, j-startIndex, startIndex, cellSize); 
					}
					collideSpheres(dPos[i], dPos[j],  dVel[i],  dVel[j], st, countColl);
				}



				if(curand_uniform_double(&st) < m_cand)
				{
					// indices of the collision candidates
					uint i = floor((1.0-curand_uniform_double(&st))*double(cellSize));
					uint j = floor((1.0-curand_uniform_double(&st))*double(cellSize));

					while(i == j) i = floor((1.0-curand_uniform_double(&st))*double(cellSize));  // if i == j : reroll i	

					i += startIndex;
					j += startIndex;

					collideSpheres(dPos[i], dPos[j],  dVel[i],  dVel[j], st, countColl);
				}

			}
        
		}
		
	}


	state[startId] = st;
	
}





/*
 * wrapper function for collideD:
 * choose collision pairs within a cell and give them a chance to collide
 */
void collide(double4     *dPos,					// input: sorted positions
			 double3     *dVel,					// input: sorted velocities
			 double		 *dInvCellVolume,		// input: inverse cell volume of the collision cells
             uint        *dCellStart,			// input: cell start index
             uint        *dCellEnd,				// input: cell end index
             curandState *dStates,				// input: cuda random states
			 uint* dCountColl,					// output: number of particle collisions
			 uint gridSize,
			 uint blockSize)
{

	collideD<<<gridSize, blockSize>>>(dPos,          
									 dVel,           
									 dInvCellVolume,
									 dCellStart,
									 dCellEnd,
									 dStates,
									 dCountColl);

	cudaDeviceSynchronize();
		

}
