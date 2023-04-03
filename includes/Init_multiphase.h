#ifndef Init_multiphase_H
#define Init_multiphase_H

/* initialization basic */
void initialization_basic_multi();
// ~~~~~~~~~~~~~~~~~~~~~open velocity inlet BC initialization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void initialization_open_velocity_inlet_BC();
/* initialization for new simulation - field variables */
void initialization_new_multi();
/* initialization for old simulation - field variables */
void initialization_old_multi();
/* initial particle distribution functions */
void initialization_new_multi_pdf();

/* memory allocate/deallocate */
/* geometry related memory allocate/deallocate */
void MemAllocate_geometry(int flag);
/* fluid flow related memory allocate/deallocate */
void MemAllocate_multi(int flag);

//===================================================================
//------------- initialize memory size -------------
//=================================================================== 
void initMemSize();


#endif