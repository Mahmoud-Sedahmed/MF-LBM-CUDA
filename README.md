
# MF-LBM C++/CUDA version

This is a C++/CUDA version of the [MF-LBM solver](https://github.com/lanl/MF-LBM).


### Screenshot
![Program Screenshot](/snapshot/PD_0.687.jpg)



## About the code
### [Original Version](https://github.com/lanl/MF-LBM)
MF-LBM is a high-performance lattice Boltzmann (LB) code for direct numerical simulation (DNS) of flow in porous media, 
primarily developed by Dr. Yu Chen (LANL), under the supervision of Prof. Albert Valocchi (UIUC), Dr. Qinjun Kang (LANL)
and Dr. Hari Viswananthan (LANL). 'MF' refers to microfluidics or 'Magic Find'. The code was first developed at University of Illinois at Urbana-Champaign 
based on a mainstream LB color-gradient multiphase model and further improved at Los Alamos National Laboratory by implementing the Continuum-Surface-Force
and geometrical wetting models to reduce spurious currents so that the inertial effects in scCO2 and brine displacement in porous media can be accounted
for. In addition, a single-phase flow solver is also provided for absolute permeability measurement or DNS of turbulent flow. Only double precision is supported.

### This version
The multiphase MF-LBM solver was rebuilt using the C++ language and the Nvidia C++/CUDA interface. It was developed by Dr. Mahmoud Sedahmed
(Alexandria University).
The new solver follows the same syntax and naming conventions of the original one to facilitate direct mapping between the two versions.
This version supports both single an double precisions. Single precision was added to benifit from the high computional power of Nvidia GPUs used in 
consumer grade computers. Such devices are becoming more powerful and equipped with many FP32 units.
The code was tested on both Linux and Windows (MS Visual Studio) operating systems.
## Authors

[Mahmoud-Sedahmed](https://github.com/Mahmoud-Sedahmed)


## Features

- Advanced LB multiphase model (CSF model + geometrical wetting model) ensuring relatively small spurious currents.
- The AA pattern streaming method is employed to significantly reduce memory access and memory consumption.
- The structure of arrays (SoA) data layout is used to achieve coalesced memory access and maximize vectorization.
- Single/Double precision solver option.
- Cross platform (Windows/Linux).
- Generic array indexing via preprocessor macros (starting at "1" not "0").
- Portable timing functions using the C++ Chrono library.
- Automatic code dependency generation in the makefile to reduce source files re-compilation.



## Prerequisites
- GCC compiler (tested on version 9.4.0)
- CUDA toolkit (tested on version 12.1)
- Nvidia GPU (tested on RTX 3090)
## Installation

1. Clone the repository
```bash
 $ git clone https://github.com/Mahmoud-Sedahmed/MF-LBM-CUDA
```
2. Change the current working directory to the MF-LBM-CUDA
```bash
$ cd MF-LBM-CUDA/
```
3. Edit the Makefile
 Use your preferred directory to modify the following parameters in the Makefile to match your CUDA toolkit installation and your GPU compute capability.
  ```makefile
  COMPUTE_CAPABILITY := 
  CUDA_LIBRARY_LOCATION := 
  CUDA_LNK_PATH := 
  ```
 Also, choose which type of build to be compiled (release/debug).
  ```makefile
   BUILD :=
  ```
  Optionally, add any additional flags to be passed to the GCC or the NVCC compilers
  ```makefile
  CPPFLAGS_ADD := 
  CUFLAGS_ADD := 
  ```
4. Select the solver precision
 Use your preferred editor to modify the following parameter in the header file "solver_precision.h"
```c++
/* Select solver precision: SINGLE_PRECISION / DOUBLE_PRECISION */
#define PRECISION (SINGLE_PRECISION)	
```
5. Compile the program using GNU Make
```bash
$ Make
```
6. Run the program
```bash
$ ./MF_LBM_CUDA
```
    
## Usage/Examples

- The file [simulation_control.txt]() is mainly used for customizing the simulations.
- Lattice units are used in the simulation control file parameters.
- The [original MF-LBM usage section](https://github.com/lanl/MF-LBM) is  advised for more details.
- The configuration files in the [original MF-LBM](https://github.com/lanl/MF-LBM) could be easily mapped to the [simulation_control.txt]() file.
- External geometry files are located in "input/geometry/". The path and name of the geometry required to be loaded in the solver must be specified in the file "Geometry_File_Path.txt". The geometry file's name suffix must not be written in "Geometry_File_Path.txt". The geometry file's name must end in a number.



## Ouptut files
One main directory will be created for the simulation results named "results", with three sub-directories:

- out1.output: bulk properties (i.e., saturation, flow rate) against time. See Monitor.cpp for more information.
- out2.checkpoint: checkpoint data used to restart simulation. See IO_multiphase.cpp for more information.
- out3.field_data: legacy vtk files for flow analysis. See IO_multiphase.cpp for more information. 

## Important Notes
- Limitations

    i. The contact angle in the control file must be less or equal to 90 degrees due to the particular numerical scheme used, meaning that fluid1 and fluid2 will always be the nonwetting phase and wetting phase, respectively. Drainage and imbibition can be completed by injecting fluid1 and fluid2 respectively.

    ii. Periodic boundary conditions can't be applied in the in X-direction.

- Potential future upgrades

    i. An alternative geometrical wetting model, e.g. [Akai et al., 2018](https://www.sciencedirect.com/science/article/pii/S030917081731028X?via%3Dihub) (recently added to the original MF-LBM solver).

    ii. Multi-GPU implementation

    iii. Support for large scale problems.

    IV. [SSS streaming technique](https://www.sciencedirect.com/science/article/pii/S0045793018308727) for direct addressing data structure.
    
 
## License

Distributed under the BSD-3 License. See [LICENSE](/LICENSE) for more information.

