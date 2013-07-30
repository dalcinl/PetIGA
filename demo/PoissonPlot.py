"""
This python script shows the usage of igakit
(https://bitbucket.org/dalcinl/igakit) to post-process results
obtained using the demo code:

   ./demo/Poisson.c 

When running the C-code with the -save option, two data files are
generated. One contains the geometry and discretization information,
the other is the solution vector.
"""
from igakit.io import PetIGA,VTK
from numpy import linspace

# read in discretization info and potentially geometry
nrb = PetIGA().read("PoissonGeometry.dat")

# read in solution vector as a numpy array
sol = PetIGA().read_vec("PoissonSolution.dat",nrb)

# write a function to sample the nrbs object (100 points from beginning to end)
uniform = lambda U: linspace(U[0], U[-1], 100)

# write a binary VTK file
VTK().write("PoissonVTK.vtk",       # output filename
            nrb,                    # igakit NURBS object
            fields=sol,             # sol is the numpy array to plot 
            sampler=uniform,        # specify the function to sample points
            scalars={'solution':0}) # adds a scalar plot to the VTK file
