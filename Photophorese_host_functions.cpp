
#include "Photophorese_host_functions.h"

/*
 * calculates the number of collision cells that are not within the test sphere
*/
unsigned int calculateNumNonEmptyCells()
{

	unsigned int nonEmptyCells = nx*nz*ny;

	for(unsigned int ix = 0; ix  < nx; ++ix)
	{

		for(unsigned int iy = 0; iy  < ny; ++iy)
		{

			for(unsigned int iz = 0; iz < nz; ++iz)
			{

					double x0 = ix*dx - halfEdgeLength;
					double y0 = iy*dy;
					double z0 = iz*dz;

					double x1 = (ix+1)*dx - halfEdgeLength;
					double y1 = (iy+1)*dy;
					double z1 = (iz+1)*dz;

					if(sqrt(x0*x0 + y1*y1 + z1*z1) <= particleRadius && sqrt(x1*x1 + y1*y1 + z1*z1) <= particleRadius)
					{
						nonEmptyCells -= 1;
					}

				}
		}
	}

	return nonEmptyCells;
}






/*
 * initialize the gas within the simulation volume
*/
void init_cpu_gas(double4* h_pos, double3* h_vel)
{

	const double a =  -halfEdgeLength; 
	const double b =  halfEdgeLength;

	std::mt19937 gen (1);
	std::uniform_real_distribution<double> uniform(a, b);
	std::uniform_real_distribution<double> uniformyz(0.0, b);
	std::uniform_real_distribution<double> uniform_phi(0.0, pi2);
	std::uniform_real_distribution<double> uniform_cos_theta(-1.0, 1.0);
	std::normal_distribution<double> normal(0.0, 1.0);

	const double v = sqrt(3.0*gasTemperature*kB/gasMassMolecule);

	unsigned int i = 0;

	unsigned int N = (unsigned int)((double(numGas)*partVol));

	while(i < N)
	{

		double4 p;
		p.x = uniform(gen);
		p.y = uniformyz(gen);
		p.z = uniformyz(gen);
		p.w = 0.0;

		if(sqrt(p.x*p.x + p.z*p.z + p.y*p.y) > particleRadius)
		{
			h_pos[i] = p;

			
			double cos_theta = uniform_cos_theta(gen);
			double sin_theta = sqrt(1.0-cos_theta*cos_theta);
			double phi = uniform_phi(gen);
			double cos_phi = cos(phi);
			double sin_phi = sin(phi);

#if boundary == 1
			h_vel[i].x = sin_theta*cos_phi*v + vFlow;
#else
			h_vel[i].x = sin_theta*cos_phi*v;
#endif
			h_vel[i].y = sin_theta*sin_phi*v;
			h_vel[i].z = cos_theta*v;
			



			/*
			h_vel[i].x = normal(gen)*vTDsqrt2;
			h_vel[i].y = normal(gen)*vTDsqrt2;
			h_vel[i].z = normal(gen)*vTDsqrt2;
			

			h_vel[i].x += vFlow;
			*/


			++i;
		}
	}

	while(i < numGas)
	{
		h_pos[i].w = 1.0;
		i++;
	}
}










/*
 * calculates the volume of the cubic cells regarding the sphere in the center
*/
void writeInverseCellVolume(double* cellInvVolume)
{
	/*
	 * the function that performs the particle - particle collisions needs the value 1/cellVolume 
	 * to calculate the number of potential collisions.
	 * So we calculate 1/cellVolume beforehand so we can multiply by 1/cellVolume instead of dividing by cellVolume, since divisions are more expensive than multiplications
	 * the volume of the collision cells that are overlapped by the test body is  calculated with a monte carlo integration
	*/
	std::mt19937 gen(123);

	const unsigned int N = 100000; // number of random points in the cube, more points for better accuracy

	for(unsigned int iz = 0; iz < nz; ++iz)
	{
		for(unsigned int iy = 0; iy < ny; ++iy)
		{
			for(unsigned int ix = 0; ix < nx; ++ix)
			{

				double volume = 0.0;

				int out = 0; // number of points outside the Sphere

				double x0 = ix*dx - halfEdgeLength;
				double y0 = iy*dy;
				double z0 = iz*dz;

				double x1 = (ix+1)*dx - halfEdgeLength;
				double y1 = (iy+1)*dy;
				double z1 = (iz+1)*dz;


				bool t = true;

				if(sqrt(x0*x0 + y1*y1 + z1*z1) <= particleRadius && sqrt(x1*x1 + y1*y1 + z1*z1) <= particleRadius)
				{
					volume = 0.0;
					t = false;
				}

				if(sqrt(x0*x0 + y0*y0 + z0*z0) > particleRadius && sqrt(x1*x1 + y0*y0 + z0*z0) > particleRadius)
				{
					volume = dz*dx*dy;
					t = false;
				}

				if(t)
				{

					std::uniform_real_distribution<double> randX(x0, x1);
					std::uniform_real_distribution<double> randY(y0, y1);
					std::uniform_real_distribution<double> randZ(z0, z1);


					for(int i = 0; i < N; i++)
					{
						double x = randX(gen);
						double y = randY(gen);
						double z = randZ(gen);


						if(sqrt(x*x + y*y + z*z) > particleRadius)
							out++;
					}

					volume = dz*dx*dy*std::max(0.001, double(out)/double(N)); // the volume of a cell shoudn't be to small, otherwise the inverse becomes too big

				}


				if(volume > 0.0)
					volume = 1.0/volume;
				else
					volume = 1.e100;


				cellInvVolume[iz*nxny + iy*nx + ix] = volume;

			
			}
		}
	}
}
