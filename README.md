This repository contains the run scripts for the paper "Simulation of reactive transport in fractured porous media" by Shin Irgens Banshoya, Inga Berre and Eirik Keilegavlen (all at Dept. of Mathematics, UiB).

The implementation uses version 1.5.0 of PorePy. Additionally, some functionalities were modified. To get these, pull the branch modified_discretization_matrices, commit dd37db2c9c403ba9d0eca734eacfc40031bbac02. 

The heavier simulations (refinement levels 3-5 and the 3D case) require some computer memory, since various matrices are large. Running these cases on your own machine is at your own risk. Moreover, an older version of PorePy is used. Installing the latest version and reverting back to the older commit may cause technical issues. If you encounter a technical problem, please create an issue at PorePy's GitHub page. 
