// the plot function requires VTK installed and was only tested for VTK 6.2 & 6.3
#define DO_PLOT 1

#if DO_PLOT
#include "photophorese_plot.h" // for Plotting
#endif

#include "helper_functions\open_folder.h" //for writing to file

#include "Photophorese_device_functions.h"
#include "Photophorese_host_functions.h"



#include<cmath>

#include<chrono>

using namespace std;
typedef unsigned int uint;


double kinE(uint N, double3* vel)
{
	double T = 0.0;

	for(uint i = 0; i < N; ++i)
	{
		T += 0.5*gasMass*(vel[i].x*vel[i].x + vel[i].y*vel[i].y + vel[i].z*vel[i].z);
	}

	return T;
}


int main(int argc, char *argv[])
{
	uint devId = 0;
	if(argc > 1)
	{
		devId = stoi(argv[1]);

	}

	printf("Using Device %d\n", devId);
	cudaSetDevice(devId);

	myfoldercreator::result_folder folder_for_testresults("data", 2);

	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
	cout << numSMs << endl;



	int reorder_threads_per_block = 128;										// Number of threads per block
	const uint reorderBlockSize = (uint)min(numGas,reorder_threads_per_block);					// Blocksize
	const uint reorderGridSize  = (uint)ceil((double)numGas/(double)reorder_threads_per_block); // Number of blocks

	cout << reorderBlockSize << "	" << reorderGridSize << endl;

	int init_threads_per_block = 256;										// Number of threads per block
	int initBlockSize = (uint)min(numGas,init_threads_per_block);					// Blocksize
	const uint initGridSize  = 96*numSMs; // Number of blocks

	int euler_threads_per_block = 256;												// Number of threads per block
	const uint eulerBlockSize = (uint)min(numGas,euler_threads_per_block);					// Blocksize
	const uint eulerGridSize  = 96*numSMs; // Number of blocks

	int collide_threads_per_block = 256;												// Number of threads per block
	const uint collideBlockSize = (uint)min(numCells,collide_threads_per_block);					// Blocksize
	const uint collideGridSize  = 96*numSMs; // Number of blocks

	cout << collideBlockSize << "	" << collideGridSize << endl;


	double momentum = 0.0;	// momentum transfer from gas to test body

	double4* hPos;	cudaMallocHost((void **)&hPos, numGas*sizeof(double4)); // CPU-Positions of the gas atoms
	double3* hVel;	cudaMallocHost((void **)&hVel, numGas*sizeof(double3));// CPU-Velocities of the gas atoms

	uint* hCellStart;	cudaMallocHost((void **)&hCellStart, (numCells+1)*sizeof(uint));
	uint* hCellEnd;		cudaMallocHost((void **)&hCellEnd, (numCells+1)*sizeof(uint));
	
	double4* dPos;				 // GPU-Positions of the gas atoms
	double3* dVel;				 // GPU-Velocities of the gas atoms
	double4* dSortedPos;		 // GPU-Positions of the gas atoms
	double3* dSortedVel;		 // GPU-Velocities of the gas atoms
	double * dMomentum;			 // GPU-momentum the gas induces on the particle

	uint*	dGridParticleHash;
	uint*	dGridParticleIndex;

	uint*   dCellStart;
	uint*   dCellEnd;

	uint hCountColl;	// number of collisions between particles
	uint* dCountColl;
	cudaMalloc((void**)&dCountColl, sizeof(uint));
	cudaDeviceSynchronize();
	cudaMemset(dCountColl, 0, sizeof(uint));

	uint hCountCollP;	// number of particles colliding with the test body
	uint* dCountCollP;
	cudaMalloc((void**)&dCountCollP, sizeof(uint));
	cudaDeviceSynchronize();
	cudaMemset(dCountCollP, 0, sizeof(uint));


	cudaMalloc((void **)&dPos, numGas*sizeof(double4));
	cudaMalloc((void **)&dVel, numGas*sizeof(double3));
	cudaMalloc((void **)&dSortedPos, numGas*sizeof(double4));
	cudaMalloc((void **)&dSortedVel, numGas*sizeof(double3));
	cudaMalloc((void **)&dMomentum, numGas*sizeof(double));
	cudaMemset(dMomentum, 0, numGas*sizeof(double));

	cudaMalloc((void **)&dGridParticleHash, numGas*sizeof(uint));
	cudaMalloc((void **)&dGridParticleIndex, numGas*sizeof(uint));

	cudaMalloc((void **)&dCellStart, (numCells+1)*sizeof(uint));
	cudaMalloc((void **)&dCellEnd, (numCells+1)*sizeof(uint));
	

	cout << "Initializing gas...";
	cout.flush();
	cudaDeviceSynchronize();
	init_cpu_gas(hPos, hVel);
	cudaDeviceSynchronize();
	cout << "  done." << endl;
	cout.flush();

	cudaMemcpy(dPos, hPos, numGas*sizeof(double4), cudaMemcpyHostToDevice);
	cudaMemcpy(dVel, hVel, numGas*sizeof(double3), cudaMemcpyHostToDevice);
	cudaMemcpy(dSortedVel, hVel, numGas*sizeof(double3), cudaMemcpyHostToDevice);


    curandState* eulerStates;
	//curandState* collideStates;
	cudaMalloc(&eulerStates, (eulerBlockSize*eulerGridSize)*sizeof(curandState));
	//cudaMalloc(&collideStates, (collideBlockSize*collideGridSize)*sizeof(curandState));
	// setup generator states
    cout << "Initializing random generators...";
    cout.flush();
	cudaDeviceSynchronize();
    initGenerators (eulerStates, eulerBlockSize*eulerGridSize, 0, initGridSize, initBlockSize);
	cudaDeviceSynchronize();
	//initGenerators (collideStates, collideBlockSize*collideGridSize, eulerBlockSize*eulerGridSize, initGridSize, initBlockSize);
    cudaDeviceSynchronize();
    cout << "  done." << endl;
	cout.flush();


	// calculate the inverse of the cells volume
	cout << "Calculating cell volume...";
    cout.flush();
	double *hCellVolume = new double[numCells];
	double* dCellVolume; cudaMalloc((void **)&dCellVolume, numCells*sizeof(double));
	uint nonEmptyCells = calculateNumNonEmptyCells();
	writeInverseCellVolume(hCellVolume);
	cudaDeviceSynchronize();
	cudaMemcpy(dCellVolume, hCellVolume, numCells*sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cout << "  done." << endl;
	cout.flush();

	const uint numBins = 100;
	uint* hBins = new uint[numBins];
	uint* dBins;
	cudaMalloc((void**)&dBins, numBins*sizeof(uint));
	cudaDeviceSynchronize();
	cudaMemset(dBins, 0, numBins*sizeof(uint));


	// normal number generator for inflow
	// each timestep a poisson distributed number with the expected number of inflowing particles as mean is choosen for the inflow boundaries 
	uint hNormal[4];
	std::poisson_distribution<int> distInflowSlow(inflowSlow);
	std::poisson_distribution<int> distInflowFast(inflowFast);
	std::poisson_distribution<int> distInflowNormal(inflowNormal);
	std::mt19937 normal_gen (7*generatorSeed);
	cudaDeviceSynchronize();
	updateNormal(normal_gen,  distInflowSlow, distInflowFast, distInflowNormal, hNormal);
	cudaDeviceSynchronize();
	//// open file
	ofstream file;
    ostringstream filename;
	filename << "momentum_Kn=" << Kn << "alpha=" << accomodationCoefficient << "_a=" << ax << "_nx=" << nx << "_L=" << (edgeLength/particleRadius) << ".dat";		  // name of file
	folder_for_testresults.open_file(file, filename); // open file 


    file << numSteps << endl;
    file << numSteps/100 << endl;
    file << particleRadius << endl;
    file << accomodationCoefficient << endl;
    file << edgeLength << endl;
    file << gasTemperature << endl;
    file << timestep << endl;
    file << numGas << endl;
	file << gasDiameter << endl;
	file << gasMassMolecule << endl;
	file << nu << endl;
    file << partVolAdjusted << endl;
    file << ax << endl;
	file << Kn << endl;
	file << kB << endl;

	ofstream file2;
    ostringstream filename2;
    filename2 << "bins_Kn=" << Kn << "alpha=" << accomodationCoefficient << "_a=" << ax << "_nx=" << nx << "_L=" << (edgeLength/particleRadius) << ".dat";		  // name of file
	folder_for_testresults.open_file(file2, filename2);  // open file


	ofstream file3;
    ostringstream filename3;
    filename3 << "rho_Kn=" << Kn << "alpha=" << accomodationCoefficient << "_a=" << ax << "_nx=" << nx << "_L=" << (edgeLength/particleRadius) << ".dat";		  // name of file
	folder_for_testresults.open_file(file3, filename3);  // open file
	///////////////

	file3 << nx << endl;

	// Write cell Volime to file
	for(int iz = 0; iz < nz; ++iz)
	{
		const unsigned int iy = 0;

			for(int ix = 0; ix < nx; ++ix)
			{
				file3 << hCellVolume[iz*nx*ny + iy*nx + ix] << "	";
			
			}
			file3 << endl;
	}
	



	double* hDensity = new double[(nx*nz*ny)];
	for(int iz = 0; iz < nz; ++iz)
	{
		for(int iy = 0; iy < ny; ++iy)
		{
			for(int ix = 0; ix < nx; ++ix)
			{
				hDensity[iz*nx*ny + iy*nx + ix] = 0.0;
			}
		}
	}


    /////////////////////////////////////////////////
	// start the simulation
	////////////////////////////////////////////////

	printf("Starting simulation...\n grid size = %d\n blocksize = %d\n collide grid size = %d\n blocksize = %d\n", reorderGridSize, reorderBlockSize, eulerGridSize, eulerBlockSize);
	printf("Size of Volume = %f, 	dx = %f,	dr = %f\n", edgeLength, dx, dz);
	printf("particle Gridsize: nx = %d	nr =	%d	number of Cells = %d\n", nx, nz, numCells);
	printf("average particles per cell = %f\n", (float)numGas/(float(numCells)));

	double totalParticlesInside = 0.0;
	double totalParticlesOutside = 0.0;

	uint step = 0;

	uint divisor = 1000; // every 1000 timesteps the programm prints out some information about the simulation
	uint mainloop = numSteps/divisor; 
	uint restloop = numSteps%divisor;

	printf("number of timesteps = %d\n", mainloop*divisor + restloop);


	calcHash(dGridParticleHash, dGridParticleIndex, dPos, eulerGridSize, eulerBlockSize);

	reorderDataAndFindCellStart(dCellStart, dCellEnd, dSortedPos, dSortedVel, dGridParticleHash, dGridParticleIndex, dPos, dVel, reorderGridSize, reorderBlockSize);


	#if DO_PLOT
			plot(particleRadius, hPos, numGas);
	#endif

	sortBins(dVel, dPos, dBins, numBins, eulerGridSize, eulerBlockSize);
	cudaDeviceSynchronize();
	cudaMemcpy(hBins, dBins, numBins*sizeof(uint), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaMemset(dBins, 0, numBins*sizeof(uint));

	file2 << 0.0 << "	" << 0.0 << "	" << 0.0 << "	" << (3.0*gasTemperature*kB/gasMassMolecule)*0.5*gasMass*numGas << "	";
	for(uint i = 0; i < numBins; i++)
	{
		file2 << double(hBins[i])/double(numGas)/partVol << "	";
	}
	file2 << endl;

	cudaMemcpy(hVel, dVel, numGas*sizeof(double3), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	file2 << 0.0 << "	" << 0.0 << "	" << 0.0 << "	" << kinE(numGas, hVel) << "	";

	for(uint i = 0; i < numBins; i++)
	{
		file2 << double(hBins[i])/double(numGas)/partVol << "	";
	}
	file2 << endl;

	

	for(uint i = 0; i < restloop; i++)
	{
		step++;

		euler(dPos, dVel, dMomentum, dCellStart, dCellEnd, eulerStates, dCountCollP, eulerGridSize, eulerBlockSize);

		momentum = summation(numGas, dMomentum);
		file << momentum << endl;

		calcHash(dGridParticleHash, dGridParticleIndex, dPos, eulerGridSize, eulerBlockSize);

		reorderDataAndFindCellStart(dCellStart, dCellEnd, dSortedPos, dSortedVel, dGridParticleHash, dGridParticleIndex, dPos, dVel, reorderGridSize, reorderBlockSize);

		updateNormal(normal_gen,  distInflowSlow, distInflowFast, distInflowNormal, hNormal);

		collide(dPos, dVel, dCellVolume, dCellStart, dCellEnd, eulerStates, dCountColl, collideGridSize, collideBlockSize);


#if DO_PLOT
		cudaMemcpy(hPos, dPos, numGas*sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(hVel, dVel, numGas*sizeof(double3), cudaMemcpyDeviceToHost);
		update_plot(hPos, hVel, numGas);
#endif

	}

	auto start = chrono::steady_clock::now();

	for(uint loop = 0; loop < divisor; loop++)
	{

		// Measure time 
		auto end = chrono::steady_clock::now();

		auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end- start);

		uint hours = (delta.count()*(divisor-loop))/3600000;
		uint minutes = ((delta.count()*(divisor-loop)) % 3600000)/60000;
		uint seconds = ((delta.count()*(divisor-loop)) % 60000)/1000;

		printf("delta = %d\n", delta.count());
		printf("%.1f%% done.	Estimated remaining time : %d h	%d m	%d s\n", float(loop)/10.0, hours, minutes, seconds);


		cudaMemcpy(hCellStart, dCellStart, (numCells+1)*sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(hCellEnd, dCellEnd, (numCells+1)*sizeof(uint), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		uint maxCellSize = 0;
		uint averageCellSize = 0;

		for(uint iz = 0; iz < nz; ++iz)
		{
			for(uint iy = 0; iy < ny; ++iy)
			{
				for(uint ix = 0; ix < nx; ++ix)
				{

					uint cellSize = 0;

					uint l = iz*nx*ny + iy*nx + ix;

					if (hCellStart[l] != 0xffffffff)
					{
						cellSize = hCellEnd[l] - hCellStart[l];
						if(cellSize > 1)
						{
							averageCellSize += cellSize;
						}

						if (cellSize > maxCellSize)
						{
							maxCellSize = cellSize;
						}
					}

					if(loop > 400)
					{
					if(iy == 0)
						hDensity[iz*nx*ny + iy*nx + ix] += double(cellSize)/600.0;
				
					}
				}
			}
		}


		int numOutside = 0;

		if(hCellStart[numCells] != 0xffffffff)
		{
			numOutside = hCellEnd[numCells] - hCellStart[numCells]; 
		}

		totalParticlesInside += double(averageCellSize);
		totalParticlesOutside += double(numOutside);

		printf("maximum particles per cell = %d\n", maxCellSize);
		printf("# of particles in Volume  = %d	# outside = %d	sum = %d\n", averageCellSize, numOutside, averageCellSize+numOutside);
		//printf("%% of particles in Volume  = %f	%% outside = %f\n", 100.0*double(averageCellSize)/double(numGas), 100.0*double(numOutside)/double(numGas));
		printf("average particles per cell = %f\n", double(averageCellSize)/double(nonEmptyCells));


		sortBins(dVel, dPos, dBins, numBins, eulerGridSize, eulerBlockSize);


		cudaMemcpy(hBins, dBins, numBins*sizeof(uint), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaMemset(dBins, 0, numBins*sizeof(uint));


		cudaMemcpy(&hCountColl, dCountColl, sizeof(uint), cudaMemcpyDeviceToHost); 
		cudaDeviceSynchronize();
		cudaMemcpy(&hCountCollP, dCountCollP, sizeof(uint), cudaMemcpyDeviceToHost); 
		cudaDeviceSynchronize();

		printf("num Coll = %f	particle Coll = %f\n\n", double(hCountColl)/double(mainloop), double(hCountCollP)/double(mainloop));
		cudaMemset(dCountColl, 0, sizeof(uint));
		cudaMemset(dCountCollP, 0, sizeof(uint));

		cudaMemcpy(hVel, dVel, numGas*sizeof(double3), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		file2 << double(hCountColl)/double(mainloop) << "	" << double(hCountCollP)/double(mainloop) << "	" << delta.count() << "	" << kinE(numGas, hVel) << "	";
		
		for(uint i = 0; i < numBins; i++)
		{
			file2 << double(hBins[i])/double(averageCellSize) << "	";
		}
		file2 << endl;

		start = chrono::steady_clock::now();

		for(uint i = 0; i < mainloop; i++)
		{
		//////////////////////////////////
		// actuall simulation work is done here
		////////////////////////////////
			step++;

			euler(dPos, dVel, dMomentum, dCellStart, dCellEnd, eulerStates, dCountCollP, eulerGridSize, eulerBlockSize);

			momentum = summation(numGas, dMomentum);
			file << momentum << endl;

			calcHash(dGridParticleHash, dGridParticleIndex, dPos, eulerGridSize, eulerBlockSize);

			reorderDataAndFindCellStart(dCellStart, dCellEnd, dSortedPos, dSortedVel, dGridParticleHash, dGridParticleIndex, dPos, dVel, reorderGridSize, reorderBlockSize);

			updateNormal(normal_gen,  distInflowSlow, distInflowFast, distInflowNormal, hNormal);

			collide(dPos, dVel, dCellVolume, dCellStart, dCellEnd, eulerStates, dCountColl, collideGridSize, collideBlockSize);

#if DO_PLOT
			cudaMemcpy(hPos, dPos, numGas*sizeof(double4), cudaMemcpyDeviceToHost);
			cudaMemcpy(hVel, dVel, numGas*sizeof(double3), cudaMemcpyDeviceToHost);
			update_plot(hPos, hVel, numGas);
#endif


		}

	}



	// Write particle density to file
	for(int iz = 0; iz < nz; ++iz)
	{
		const unsigned int iy = 0;

			for(int ix = 0; ix < nx; ++ix)
			{
				file3 << hDensity[iz*nx*ny + iy*nx + ix] << "	";
			
			}
			file3 << endl;
	}
	


	cout << "Resulting force	x = " << momentum/(double(numSteps*timestep)) << endl << endl;
	cout << "On average " << totalParticlesInside/double(numGas) << " % of the particles where inside the simulation volume and " <<  totalParticlesOutside/double(numGas) << "% outside." << endl;

	cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dMomentum);

    cudaFree(eulerStates);
	//cudaFree(collideStates);

    cudaFree(dSortedPos);
    cudaFree(dSortedVel);
    cudaFree(dGridParticleHash);
    cudaFree(dGridParticleIndex);
    cudaFree(dCellStart);
    cudaFree(dCellEnd);

	cudaFree(dCellVolume);

	cudaFreeHost(hPos);
	cudaFreeHost(hVel);
	cudaFreeHost(hCellStart);
	cudaFreeHost(hCellEnd);

	cudaFree(dCountColl);
	cudaFree(dCountCollP);

	cudaFree(dBins);
	delete[] hBins;
	delete[] hCellVolume;
	delete[] hDensity;

	cudaDeviceReset();

	return 0;
	}