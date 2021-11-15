# Animation P5: Cloth Simulation

In this project, I created a realistic cloth using physics based animation. I represented the cloth as a system of particles stored in various matrices and vectors, and did implicit Euler integration to solve for the new position of each particle at every step. Since I used implicit integration, the simulation remains stable even when using larger timesteps. Additionally, all physics matrices were stored and used as sparse matrices, which significantly increased performance. Created in C++ using OpenGL.

Usage after compilation:
./A5 RESOURCE_DIR