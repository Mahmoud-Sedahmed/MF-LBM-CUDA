#ifndef Geometry_preprocessing_H
#define Geometry_preprocessing_H

// =====================================================================================================================================
// ----------------------geometry_preprocessing new ----------------------
// process the geometry before the start of the main iteration : normal directions of the solid surface; extrapolation weights from
// different directions not recommended for relatively large domain - the processing time before each new or old simulation could be
// too long(not paralllelized on CPU for the GPU version code); also not optimal for Xeon Phi as the on - package memory is limited
// =====================================================================================================================================
void geometry_preprocessing_new();

//=====================================================================================================================================
//----------------------geometry_preprocessing based on existing preprocessed data----------------------
//load the geometry information from precomputed file.recommended for relatively large domain
//the geometry in the simulation must 100 % match the preprocessed geometry data!!!
//Due to the issue with different padding schemes used in different compilers for MPI_Type_create_struct,
//The compiler used to compute the geometry info must be the same with the compiler for simulation
//I.e., Intel - Intel, PGI - PGI, GCC - GCC
//=====================================================================================================================================
void geometry_preprocessing_load();

#endif
