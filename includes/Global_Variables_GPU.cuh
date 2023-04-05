#ifndef Global_Variables_GPU_cuh
#define Global_Variables_GPU_cuh
#include "externLib_CUDA.cuh"
#include "solver_precision.h"

__constant__ T_P lbm_gamma_d, force_z_d, la_nui1_d, la_nui2_d, mrt_e2_coef1_d, mrt_e2_coef2_d, mrt_omega_xx_d, mrt_coef1_d, mrt_coef2_d, mrt_coef3_d, mrt_coef4_d, lbm_beta_d
, w_equ_1_d, w_equ_2_d, RK_weight2_d, w_equ_d[19], ISO4_d[2], phi_inlet_d, relaxation_d, sa_inject_d, uin_avg_d;
__constant__ T_P cos_theta_d;
__constant__ int ex_d[19], ey_d[19], ez_d[19], Z_porous_plate_d, porous_plate_cmd_d;
__constant__ long long num_solid_boundary_d, num_fluid_boundary_d, nxGlobal_d, nyGlobal_d, nzGlobal_d;
#endif