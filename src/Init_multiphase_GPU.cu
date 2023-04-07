#include "externLib.h"
#include "solver_precision.h"
#include "externLib_CUDA.cuh"
#include "Module_extern.h"
#include "utils_GPU.cuh"
#include "Init_multiphase_GPU.h"
#include "Fluid_singlephase_extern.h"
#include "Fluid_multiphase_extern.h"
#include "Idx_cpu.h"


/* initialization basic - CUDA */
void initialization_GPU() {
    cout << "***************************** GPU Specifications **********************************" << endl;
    const int kb = 1024;
    const int mb = kb * kb;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    cout << 0 << ": " << props.name << ": " << props.major << "." << props.minor << endl;
    cout << "  Global memory:   " << props.totalGlobalMem / mb << " mb" << endl;
    cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << " kb" << endl;
    cout << "  Constant memory: " << props.totalConstMem / kb << " kb" << endl;
    cout << "  Block registers: " << props.regsPerBlock << endl;
    cout << "  Number of SMs: " << props.multiProcessorCount << endl;
    cout << "  Clock frequencey: " << props.clockRate / 1e3 << " MHz" << endl;
    cout << "  Warp size:         " << props.warpSize << endl;
    cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
    cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
    cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
    cout << "***************************** GPU Specifications **********************************" << endl;

    // allocate memory for (walls) and copy data from host
    MemAllocate_geometry_GPU(1);
    //************* fluid flow related memory allocate/deallocate ******************************
    MemAllocate_multi_GPU(1);

    cout << "Total amount of global memory needed on GPU (GB) = " << (memory_gpu / double(1024 * 1024 * 1024)) << endl;

}


/* geometry related memory allocate/deallocate */
void MemAllocate_geometry_GPU(int flag) {
    if (flag == 1) {
        /* allocate memory on device */
        cudaErrorCheck(cudaMalloc(&walls_d, mem_size_s2_int)); memory_gpu += mem_size_s2_int;
        cudaErrorCheck(cudaMemcpy(walls_d, walls, mem_size_s2_int, cudaMemcpyHostToDevice));

        cudaErrorCheck(cudaMalloc(&walls_type_d, mem_size_s4_int)); memory_gpu += mem_size_s4_int;
        cudaErrorCheck(cudaMemcpy(walls_type_d, walls_type, mem_size_s4_int, cudaMemcpyHostToDevice));

        cudaErrorCheck(cudaMalloc(&s_nx_d, mem_size_s4_TP)); memory_gpu += mem_size_s4_TP;
        cudaErrorCheck(cudaMemcpy(s_nx_d, s_nx, mem_size_s4_TP, cudaMemcpyHostToDevice));

        cudaErrorCheck(cudaMalloc(&s_ny_d, mem_size_s4_TP)); memory_gpu += mem_size_s4_TP;
        cudaErrorCheck(cudaMemcpy(s_ny_d, s_ny, mem_size_s4_TP, cudaMemcpyHostToDevice));

        cudaErrorCheck(cudaMalloc(&s_nz_d, mem_size_s4_TP)); memory_gpu += mem_size_s4_TP;
        cudaErrorCheck(cudaMemcpy(s_nz_d, s_nz, mem_size_s4_TP, cudaMemcpyHostToDevice));
        
    }
    else {
        cudaErrorCheck(cudaFree(walls_d));
        cudaErrorCheck(cudaFree(walls_type_d));
        cudaErrorCheck(cudaFree(s_nx_d));
        cudaErrorCheck(cudaFree(s_ny_d));
        cudaErrorCheck(cudaFree(s_nz_d));
    }

}

// ************* fluid flow related memory allocate/deallocate ******************************
void MemAllocate_multi_GPU(int flag) {
    if (flag == 1) {
        //cudaErrorCheck(cudaMalloc(&u_d, mem_size_s1_TP)); memory_gpu += mem_size_s1_TP; cudaErrorCheck(cudaMemcpy(u_d, u, mem_size_s1_TP, cudaMemcpyHostToDevice));
        //cudaErrorCheck(cudaMalloc(&v_d, mem_size_s1_TP)); memory_gpu += mem_size_s1_TP; cudaErrorCheck(cudaMemcpy(v_d, v, mem_size_s1_TP, cudaMemcpyHostToDevice));
        //cudaErrorCheck(cudaMalloc(&w_d, mem_size_s1_TP)); memory_gpu += mem_size_s1_TP; cudaErrorCheck(cudaMemcpy(w_d, w, mem_size_s1_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&rho_d, mem_size_s1_TP)); memory_gpu += mem_size_s1_TP; cudaErrorCheck(cudaMemcpy(rho_d, rho, mem_size_s1_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&curv_d, mem_size_s1_TP)); memory_gpu += mem_size_s1_TP; cudaErrorCheck(cudaMemcpy(curv_d, curv, mem_size_s1_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&W_in_d, NXG1 * NYG1 * sizeof(T_P))); memory_gpu += NXG1 * NYG1 * sizeof(T_P); cudaErrorCheck(cudaMemcpy(W_in_d, W_in, NXG1 * NYG1 * sizeof(T_P), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&pdf_d, mem_size_f1_TP)); memory_gpu += mem_size_f1_TP; cudaErrorCheck(cudaMemcpy(pdf_d, pdf, mem_size_f1_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&cn_x_d, mem_size_s2_TP)); memory_gpu += mem_size_s2_TP; cudaErrorCheck(cudaMemcpy(cn_x_d, cn_x, mem_size_s2_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&cn_y_d, mem_size_s2_TP)); memory_gpu += mem_size_s2_TP; cudaErrorCheck(cudaMemcpy(cn_y_d, cn_y, mem_size_s2_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&cn_z_d, mem_size_s2_TP)); memory_gpu += mem_size_s2_TP; cudaErrorCheck(cudaMemcpy(cn_z_d, cn_z, mem_size_s2_TP, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMalloc(&c_norm_d, mem_size_s2_TP)); memory_gpu += mem_size_s2_TP; cudaErrorCheck(cudaMemcpy(c_norm_d, c_norm, mem_size_s2_TP, cudaMemcpyHostToDevice));
        //convective BC
        if (outlet_BC == 1) {
            cudaErrorCheck(cudaMalloc(&phi_convec_bc_d, NXG1 * NYG1 * sizeof(T_P))); memory_gpu += NXG1 * NYG1 * sizeof(T_P); cudaErrorCheck(cudaMemcpy(phi_convec_bc_d, phi_convec_bc, NXG1 * NYG1 * sizeof(T_P), cudaMemcpyHostToDevice));
            cudaErrorCheck(cudaMalloc(&g_convec_bc_d, NXG1 * NYG1 * 19 * sizeof(T_P))); memory_gpu += NXG1 * NYG1 * 19 * sizeof(T_P); cudaErrorCheck(cudaMemcpy(g_convec_bc_d, g_convec_bc, NXG1 * NYG1 * 19 * sizeof(T_P), cudaMemcpyHostToDevice));
            cudaErrorCheck(cudaMalloc(&f_convec_bc_d, NXG1 * NYG1 * 19 * sizeof(T_P))); memory_gpu += NXG1 * NYG1 * 19 * sizeof(T_P); cudaErrorCheck(cudaMemcpy(f_convec_bc_d, f_convec_bc, NXG1 * NYG1 * 19 * sizeof(T_P), cudaMemcpyHostToDevice));
        }
        cudaErrorCheck(cudaMalloc(&phi_d, mem_size_s4_TP)); memory_gpu += mem_size_s4_TP; cudaErrorCheck(cudaMemcpy(phi_d, phi, mem_size_s4_TP, cudaMemcpyHostToDevice));

    }
    else {
        //cudaErrorCheck(cudaFree(u_d));
        //cudaErrorCheck(cudaFree(v_d));
        //cudaErrorCheck(cudaFree(w_d));
        cudaErrorCheck(cudaFree(rho_d));
        cudaErrorCheck(cudaFree(curv_d));
        cudaErrorCheck(cudaFree(W_in_d));
        cudaErrorCheck(cudaFree(pdf_d));
        cudaErrorCheck(cudaFree(cn_x_d));
        cudaErrorCheck(cudaFree(cn_y_d));
        cudaErrorCheck(cudaFree(cn_z_d));
        cudaErrorCheck(cudaFree(c_norm_d));
        if (outlet_BC == 1) {
            cudaErrorCheck(cudaFree(phi_convec_bc_d));
            cudaErrorCheck(cudaFree(g_convec_bc_d));
            cudaErrorCheck(cudaFree(f_convec_bc_d));
        }
        cudaErrorCheck(cudaFree(phi_d));


    }
}
