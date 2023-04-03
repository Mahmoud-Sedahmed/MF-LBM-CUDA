#ifndef Init_multiphase_GPU
#define Init_multiphase_GPU

/* initialization basic - CUDA */
void initialization_GPU();
/* memory allocate/deallocate */
/* geometry related memory allocate/deallocate */
void MemAllocate_geometry_GPU(int flag);
//************* fluid flow related memory allocate/deallocate ******************************
void MemAllocate_multi_GPU(int flag);

#endif