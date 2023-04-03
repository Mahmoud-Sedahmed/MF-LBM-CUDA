#ifndef IO_multiphase_H
#define IO_multiphase_H

#include "Module_extern.h"
#include "utils.h"

//=========================================================================================================================== =
//----------------------Read input parameters----------------------
//=========================================================================================================================== =
void read_parameter_multi();
//=========================================================================================================================== =
//----------------------save data ----------------------
//=========================================================================================================================== =
// ******************************* save checkpoint data *************************************
void save_checkpoint(int save_option);

// ******************************* save data - phi *************************************
void save_phi(int nt);

// ******************************* save data - macro variables *************************************
void save_macro(int nt);

//void slice_output(int nt);

//=========================================================================================================================== =
//----------------------save data - vtk----------------------
//=========================================================================================================================== =

/* *********************** 3D legacy VTK writer *********************** */
void VTK_legacy_writer_3D(int nt, int vtk_type);
//void Binary_writer_3D(int nt);

//***********************2D legacy VTK writer * **********************
//used in 2D micromodel simulation by averaging phi along y(depth) direction
void VTK_phi_2d_micromodel(int nt);

/* ****************** save geometry VTK *********************** */
void VTK_walls_bin();



//===================================================================
//---------------------- Print a scalar array to a bindary file ----------------------
//=================================================================== 
 
template <typename T>
void printScalarBinary_gh(T* scalar_array, const string name, long long mem_size, const string path = "results/out3.field_data/") {
	// Get the name of the passed array
	ostringstream filepath;
	filepath << path << name << ".bin";
	string fns = filepath.str();
	const char* fnc = fns.c_str();
	FILE* binary_file = fopen(fnc, "wb+"); // open the file for the first time (create the file)
	if (binary_file == NULL) { ERROR("Could not create scalar binary file!"); }
	long long file_size = mem_size;
	fwrite(scalar_array, file_size, 1, binary_file);
	fclose(binary_file);
}



#endif