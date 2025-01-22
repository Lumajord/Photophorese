/* VTK Includes */
//#define vtkRenderingCore_AUTOINIT 4(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingFreeTypeOpenGL,vtkRenderingOpenGL)
//#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL)

 #include <vtkAutoInit.h>
 VTK_MODULE_INIT(vtkRenderingOpenGL);

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkProperty.h>
#include <vtkInteractorStyleTrackballCamera.h>
/* VTK Includes Ende*/


#include "simulation_constants.h"

#include "photophorese_plot.h"




vtkSmartPointer<vtkPoints> points;
vtkSmartPointer<vtkCellArray> lines;
vtkSmartPointer<vtkCellArray> triangles;
vtkSmartPointer<vtkPolyData> pointsPolydata;
vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter;
vtkSmartPointer<vtkPolyData> polydata;
vtkSmartPointer<vtkUnsignedCharArray> colors;
vtkSmartPointer<vtkPolyDataMapper> mapper;
vtkSmartPointer<vtkActor> actor;
vtkSmartPointer<vtkRenderer> renderer;
vtkSmartPointer<vtkRenderWindow> renderWindow;
vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor;

typedef unsigned int uint;

double temperature = gasTemperature;
double temp_min = 10.0;
double temp_max = 1000.0;

double v_max = sqrt(8.0*temp_max*kB/3.1415926535898/gasMassMolecule);
double v_min = sqrt(8.0*temp_min+kB/3.1415926535898/gasMassMolecule);


double calc_vel(double3 vel)
{
	double v = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
	return v;
}


double h_calc_temp(double4 p)
{

	if(p.x < 0.0)
		return 400.0;
	else
		return 400.0;
}

unsigned char clamp_temp(double x)
{
	if(x >= 1.0)
		return (unsigned char)(255);
	else if(x < 0.0)
		return (unsigned char)(0);

	return (unsigned char)(x*255.0);
}



/*
 * updates the plot with new positions and velocities
*/
extern "C"
void update_plot(	double4 *r_new,		// input: positions
					double3 *vel_new,	// input: velocities
					uint N)				// input: number of particles
{

	int step = 1;

	N /= step;

	// Koordinaten der Punkte neu schreiben
	for(uint i = 0; i < N; ++i)
	{

		unsigned char color[4];

		if(r_new[i*step].w == 0.0)
		{


			double p[3]  = {r_new[i*step].x, r_new[i*step].y, r_new[i*step].z};
			polydata->GetPoints()->SetPoint(i, p);

			double v = calc_vel(vel_new[i*step]);
			double coeff = (v - v_min)/(v_max-v_min);

			color[0] = clamp_temp(coeff);
			color[1] = clamp_temp(coeff - coeff*coeff)*4.0;
			color[2] = clamp_temp(1.0 - coeff);
			color[3] = 255;
		}
		else
		{

			double p[3]  = {0.0, 0.0, 0.0};
			polydata->GetPoints()->SetPoint(i, p);

			color[0] = 227;
			color[1] = 207;
			color[2] = 87;
			color[3] = 0;
		}
		//if(length(make_double3(r_new[i])) < 1.0)
		//	color[3] = 255;
		
		colors->SetTupleValue(i,color);


	}

	// Plotten
	polydata->Modified();
	renderWindow->Render();
	renderWindowInteractor->Start();

}


/*
 * initializes a VTK plot
*/
extern "C"
void plot(	double a,		// input: radius of the spherical test body
			double4 *atoms,	// input: positions
			uint N)			// input: number of particles
{

	int step = 1;

	N /= step;

    const int ball_points = 96;
    double4* r = new double4[ball_points*ball_points*ball_points];
    double l0 = 2.0f*a/(double)(ball_points-1);

	// Initialize Particle points
	for(int z = 0; z < ball_points; z++){
		for(int y = 0; y < ball_points; y++){
			for(int x = 0; x < ball_points; x++){
				int N = x+y*ball_points+z*ball_points*ball_points;
				r[N].x = l0*x - a;
				r[N].y = l0*y - a;
				r[N].z = l0*z - a;
				r[N].w = 0.0f;

				double len = sqrt(r[N].x*r[N].x + r[N].y*r[N].y + r[N].z*r[N].z); 

				if(len > a-0.99f*l0  && len <= a)
				{
					r[N].w = 2.0f;
				}
			}
		}
	}

	// Initialize Particle color
	static unsigned char point_color[ball_points*ball_points*ball_points*4];

	for(int z = 0; z < ball_points; z++){
		for(int y = 0; y < ball_points; y++){
			for(int x = 0; x < ball_points; x++){

				int N = x+y*ball_points+z*ball_points*ball_points;
				double4 p;
				p.x = (double)x/(double)(ball_points-1) - 0.5;
				p.y = (double)y/(double)(ball_points-1) - 0.5;
				p.z = (double)z/(double)(ball_points-1) - 0.5;
				p.w = 0.0;

				double temp = h_calc_temp(p);

				double coeff = (temp - temp_min)/(temp_max-temp_min);
				
				point_color[N*4 + 0] = clamp_temp(coeff);
				point_color[N*4 + 1] = clamp_temp(coeff);//clamp_temp(coeff - coeff*coeff)*4.0;
				point_color[N*4 + 2] = clamp_temp(coeff);//clamp_temp(1.0 - coeff);
				point_color[N*4 + 3] = 255;

				
			}
		}
	}

	dim3 dims((unsigned int)ball_points,(unsigned int)ball_points,(unsigned int)ball_points);
	const int Nx = ball_points;
	const int Ny = ball_points;
	const int Nz = ball_points;


	// farben
	colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(4);
    colors->SetName ("Colors");

	// Punkte in points speichern
	points = vtkSmartPointer<vtkPoints>::New();


	// Insert atoms and atom colors
	for(uint i = 0; i < N; i++)
	{
		if(atoms[i*step].w == 0.0)
		{
			points->InsertNextPoint(atoms[i*step].x, atoms[i*step].y, atoms[i*step].z);


			unsigned char color[4];
			color[0] = 227;
			color[1] = 207;
			color[2] = 87;
			color[3] = 255;

			colors->InsertNextTupleValue(color);
		}
		else
		{
			points->InsertNextPoint(0.0, 0.0, 0.0);


			unsigned char color[4];
			color[0] = 227;
			color[1] = 207;
			color[2] = 87;
			color[3] = 0;

			colors->InsertNextTupleValue(color);
		}
	}


	// Insert particle
	for(uint z = 0; z < Nz; z++)
	{
		for(uint y = 0; y < Ny; y++)
		{
			for(uint x = 0; x < Nx; x++)
			{

				uint N = x+y*Nx+z*Nx*Ny;

				double4 point = r[N];

				// farbe der Punkte speichern
				if(point.w > 0.5f)
				{
					points->InsertNextPoint(point.x, point.y, point.z);
					unsigned char color[4];

					color[0] = point_color[4*N + 0];
					color[1] = point_color[4*N + 1];
					color[2] = point_color[4*N + 2];
					color[3] = point_color[4*N + 3];
					colors->InsertNextTupleValue(color);
				}

			}
		}

	}


	// Create a polydata to store points
	pointsPolydata = vtkSmartPointer<vtkPolyData>::New();

	// Add the points to the dataset
    pointsPolydata->SetPoints(points);

	vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	vertexFilter->SetInputData(pointsPolydata);
    vertexFilter->Update();

    triangles = vtkSmartPointer<vtkCellArray>::New();


	polydata = vtkSmartPointer<vtkPolyData>::New();

	// Add the points to the dataset
	polydata->DeepCopy(vertexFilter->GetOutput());

    polydata->GetCellData()->SetScalars(colors);

	// Visualization
	mapper = vtkSmartPointer<vtkPolyDataMapper>::New();

	mapper->SetInputData(polydata);

	actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetPointSize(5);

	renderer = vtkSmartPointer<vtkRenderer>::New();
	renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	renderer->AddActor(actor);

	renderWindow->Render();

	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style =
	vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	renderWindowInteractor->SetInteractorStyle( style );

	renderWindowInteractor->Start();
}




