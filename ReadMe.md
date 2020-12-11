The MATLAB code package implements the stochastic variational inference algorithm to learn spatio-temporal extreme-value graphical models as described in the following paper:

H. Yu, and J. Dauwels, ''Modeling Spatio-Temporal Extreme Events using Graphical Models,'' to be appear in IEEE Trans. Signal Process.

The main script is mainfull.m. To test your own datasets, please follow the steps below:

1. Replace the artifical data "artiData(16x16)_GSL" by your own block maxima data set when loading data and store the data in the nxp matrix XDat, where n is the no. of time series and p is the no. of locations.  

2. For regular grids where the measuring sites are evenly distributed, please input the values of pc and pr, where pr and pc are respectively the no. of rows and columns of the lattice where we observe the block maxima.

3. For irregular grid, please provide the lattitudes and longitudes of the measuring sites as the input of variables Ltt and Lgt.

4. If the time series is periodic, you need to change the period pp accordingly. Otherwise, pp = 1.

5. You can tune pct (i.e, the percentage of measurements used to compute the stochastic gradients) in SVEM_final.m. Typically, pct should be larger than max(1/pt,1/ps) such that there is at least one measurement for every location and time point. Smaller pct value can usually speed up the algorithm with the accuracy unchanged. 



Yu Hang, NTU, May 2015.