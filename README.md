# Photophorese  
Simulation to compute the photophoretic force felt by a spherical particle with a temperature gradient inside a rarefied gas. The code is based on the cuda sample Simulations/particles and implements the direct simulation monte-carlo (DSMC) method by [G.A. Bird](http://www.gab.com.au/).
The simulation can also be visualized with [VTK](https://vtk.org/).

![alt text](https://github.com/Lumajord/Photophorese/blob/main/sphere.png "Spherical particle with temperature gradient inside a rarefied gas.")

## How to use  
Change physical and numerical parameters inside "generate_simulation_constants.py" to the desired values and run the python script.  
Then build the project with 'make' and run the executable.  
The program produces plain text output files containing gas densities and transfered momentum data.  
