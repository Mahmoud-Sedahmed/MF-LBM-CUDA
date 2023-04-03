#include "externLib.h"
#include "solver_precision.h"
#include "externLib_CUDA.cuh"
#include "main_iteration_GPU.h"
#include "Module_extern.h"
#include "utils.h"
#include "utils_GPU.cuh"
#include "Fluid_singlephase_extern.h"
#include "Fluid_multiphase_extern.h"
#include "Global_Variables_GPU.cuh"
#include "Idx_gpu.cuh"

/* copy constant data to GPU */
void copyConstantData() {
    cudaErrorCheck(cudaMemcpyToSymbol(lbm_gamma_d, &lbm_gamma, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(force_z_d, &force_z, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(la_nui1_d, &la_nui1, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(la_nui2_d, &la_nui2, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(mrt_e2_coef1_d, &mrt_e2_coef1, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(mrt_e2_coef2_d, &mrt_e2_coef2, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(mrt_omega_xx_d, &mrt_omega_xx, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(mrt_coef1_d, &mrt_coef1, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(mrt_coef2_d, &mrt_coef2, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(mrt_coef3_d, &mrt_coef3, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(mrt_coef4_d, &mrt_coef4, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(lbm_beta_d, &lbm_beta, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(w_equ_1_d, &w_equ_1, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(w_equ_2_d, &w_equ_2, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(RK_weight2_d, &RK_weight2, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(w_equ_d, w_equ, 19 * sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(ex_d, ex, 19 * sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(ey_d, ey, 19 * sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(ez_d, ez, 19 * sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(Z_porous_plate_d, &Z_porous_plate, sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(porous_plate_cmd_d, &porous_plate_cmd, sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(num_solid_boundary_d, &num_solid_boundary, sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(num_fluid_boundary_d, &num_fluid_boundary, sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(ISO4_d, ISO4, 2 * sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(nxGlobal_d, &nxGlobal, sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(nyGlobal_d, &nyGlobal, sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(nzGlobal_d, &nzGlobal, sizeof(int)));
    cudaErrorCheck(cudaMemcpyToSymbol(phi_inlet_d, &phi_inlet, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(relaxation_d, &relaxation, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(sa_inject_d, &sa_inject, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(uin_avg_d, &uin_avg, sizeof(T_P)));
    cudaErrorCheck(cudaMemcpyToSymbol(cos_theta_d, &cos_theta, sizeof(T_P)));
}



#pragma region (kernel_multiphase)
//=====================================================================================================================================
//----------------------odd step kernel----------------------
// complete two streaming steps
//=====================================================================================================================================
__global__ void kernel_odd_color_GPU(int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax,
    int* walls, T_P* pdf, T_P* phi, T_P* cn_x, T_P* cn_y, T_P* cn_z, T_P* curv, T_P* c_norm) {

    // Indexing (Thread)
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    T_P fx, fy, fz, omega, cnx, cny, cnz;
    T_P sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9;
    T_P m_rho, m_e, m_e2, m_jx, m_qx, m_jy, m_qy, m_jz, m_qz, m_3pxx, m_3pixx, m_pww, m_piww, m_pxy, m_pyz, m_pzx, m_tx, m_ty, m_tz;
    T_P ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12, ft13, ft14, ft15, ft16, ft17, ft18;
    T_P g1t0, g1t1, g1t2, g1t3, g1t4, g1t5, g1t6, g1t7, g1t8, g1t9, g1t10, g1t11, g1t12, g1t13, g1t14, g1t15, g1t16, g1t17, g1t18;
    T_P g2t0, g2t1, g2t2, g2t3, g2t4, g2t5, g2t6, g2t7, g2t8, g2t9, g2t10, g2t11, g2t12, g2t13, g2t14, g2t15, g2t16, g2t17, g2t18;
    T_P s_e, s_e2, s_q, s_nu, s_pi, s_t;  //relaxation parameters
    T_P ux1, uy1, uz1, den, u2, rho1, rho2, tmp, tmp1;

    if (walls[i_s2(i, j, k)] == 0 && i >= ixmin && i <= ixmax && j >= iymin && j <= iymax && k >= izmin && k <= izmax) {
        //++++++++ + -AA pattern pull step++++++++++++
        g1t0 = pdf[i_f1(i, j, k, 0, 0)];
        g1t1 = pdf[i_f1(i - 1, j, k, 1, 0)];
        g1t2 = pdf[i_f1(i + 1, j, k, 2, 0)];
        g1t3 = pdf[i_f1(i, j - 1, k, 3, 0)];
        g1t4 = pdf[i_f1(i, j + 1, k, 4, 0)];
        g1t5 = pdf[i_f1(i, j, k - 1, 5, 0)];
        g1t6 = pdf[i_f1(i, j, k + 1, 6, 0)];
        g1t7 = pdf[i_f1(i - 1, j - 1, k, 7, 0)];
        g1t8 = pdf[i_f1(i + 1, j - 1, k, 8, 0)];
        g1t9 = pdf[i_f1(i - 1, j + 1, k, 9, 0)];
        g1t10 = pdf[i_f1(i + 1, j + 1, k, 10, 0)];
        g1t11 = pdf[i_f1(i - 1, j, k - 1, 11, 0)];
        g1t12 = pdf[i_f1(i + 1, j, k - 1, 12, 0)];
        g1t13 = pdf[i_f1(i - 1, j, k + 1, 13, 0)];
        g1t14 = pdf[i_f1(i + 1, j, k + 1, 14, 0)];
        g1t15 = pdf[i_f1(i, j - 1, k - 1, 15, 0)];
        g1t16 = pdf[i_f1(i, j + 1, k - 1, 16, 0)];
        g1t17 = pdf[i_f1(i, j - 1, k + 1, 17, 0)];
        g1t18 = pdf[i_f1(i, j + 1, k + 1, 18, 0)];

        g2t0 = pdf[i_f1(i, j, k, 0, 1)];
        g2t1 = pdf[i_f1(i - 1, j, k, 1, 1)];
        g2t2 = pdf[i_f1(i + 1, j, k, 2, 1)];
        g2t3 = pdf[i_f1(i, j - 1, k, 3, 1)];
        g2t4 = pdf[i_f1(i, j + 1, k, 4, 1)];
        g2t5 = pdf[i_f1(i, j, k - 1, 5, 1)];
        g2t6 = pdf[i_f1(i, j, k + 1, 6, 1)];
        g2t7 = pdf[i_f1(i - 1, j - 1, k, 7, 1)];
        g2t8 = pdf[i_f1(i + 1, j - 1, k, 8, 1)];
        g2t9 = pdf[i_f1(i - 1, j + 1, k, 9, 1)];
        g2t10 = pdf[i_f1(i + 1, j + 1, k, 10, 1)];
        g2t11 = pdf[i_f1(i - 1, j, k - 1, 11, 1)];
        g2t12 = pdf[i_f1(i + 1, j, k - 1, 12, 1)];
        g2t13 = pdf[i_f1(i - 1, j, k + 1, 13, 1)];
        g2t14 = pdf[i_f1(i + 1, j, k + 1, 14, 1)];
        g2t15 = pdf[i_f1(i, j - 1, k - 1, 15, 1)];
        g2t16 = pdf[i_f1(i, j + 1, k - 1, 16, 1)];
        g2t17 = pdf[i_f1(i, j - 1, k + 1, 17, 1)];
        g2t18 = pdf[i_f1(i, j + 1, k + 1, 18, 1)];

        // let ft be the bulk PDF
        ft0 = g1t0 + g2t0;
        ft1 = g1t1 + g2t1;
        ft2 = g1t2 + g2t2;
        ft3 = g1t3 + g2t3;
        ft4 = g1t4 + g2t4;
        ft5 = g1t5 + g2t5;
        ft6 = g1t6 + g2t6;
        ft7 = g1t7 + g2t7;
        ft8 = g1t8 + g2t8;
        ft9 = g1t9 + g2t9;
        ft10 = g1t10 + g2t10;
        ft11 = g1t11 + g2t11;
        ft12 = g1t12 + g2t12;
        ft13 = g1t13 + g2t13;
        ft14 = g1t14 + g2t14;
        ft15 = g1t15 + g2t15;
        ft16 = g1t16 + g2t16;
        ft17 = g1t17 + g2t17;
        ft18 = g1t18 + g2t18;

        // order parameter
        rho1 = g1t0 + g1t1 + g1t2 + g1t3 + g1t4 + g1t5 + g1t6 + g1t7 + g1t8 + g1t9 + g1t10 + g1t11 + g1t12 + g1t13 + g1t14 + g1t15 + g1t16 + g1t17 + g1t18;
        rho2 = g2t0 + g2t1 + g2t2 + g2t3 + g2t4 + g2t5 + g2t6 + g2t7 + g2t8 + g2t9 + g2t10 + g2t11 + g2t12 + g2t13 + g2t14 + g2t15 + g2t16 + g2t17 + g2t18;

        T_P phi_loc = (rho1 - rho2) / (rho1 + rho2);
        phi[i_s4(i, j, k)] = phi_loc;

        cnx = cn_x[i_s2(i, j, k)];
        cny = cn_y[i_s2(i, j, k)];
        cnz = cn_z[i_s2(i, j, k)];

        tmp = prc(0.5) * lbm_gamma_d * curv[i_s1(i, j, k)] * c_norm[i_s2(i, j, k)];
        fx = tmp * cnx;
        fy = tmp * cny;
        fz = tmp * cnz + force_z_d;   //body force force_z along flow direction

        //++++++++++++ - MRT COLLISION++++++++++++ + -
        // select viscosity++++++++++++++++++++++ + -
        omega = prc(1.) / (prc(6.) / ((prc(1.0) + phi_loc) * la_nui1_d + (prc(1.0) - phi_loc) * la_nui2_d) + prc(0.5));
        //MRT PARAMETERS
        s_nu = omega;
#if (mrt==1)
        //************bounceback opt * ***********
        s_e = omega;
        s_e2 = omega;
        s_pi = omega;
        s_q = prc(8.) * (prc(2.) - omega) / (prc(8.) - omega);
        s_t = s_q;
#elif (mrt==2)
        //************original * ***********
        s_e = prc(1.19);
        s_e2 = prc(1.4);
        s_pi = prc(1.4);
        s_q = prc(1.2);
        s_t = prc(1.98);

#elif (mrt==3 )           
        //************SRT * ***********
        s_e = omega;
        s_e2 = omega;
        s_pi = omega;
        s_q = omega;
        s_t = omega;
#elif (mrt==4 )           
        //************advection opt * ***********
        s_e = omega;
        s_e2 = omega;
        s_pi = omega;
        s_q = (prc(6.) - prc(3.) * omega) / (prc(3.) - omega);
        s_t = omega;
#endif 
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MRT kernel, repeated part~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //++++++++ + -!calculate macroscopic variables++++++++++++
        den = rho1 + rho2;
        ux1 = ft1 - ft2 + ft7 - ft8 + ft9 - ft10 + ft11 - ft12 + ft13 - ft14 + prc(0.5) * fx;
        uy1 = ft3 - ft4 + ft7 + ft8 - ft9 - ft10 + ft15 - ft16 + ft17 - ft18 + prc(0.5) * fy;
        uz1 = ft5 - ft6 + ft11 + ft12 - ft13 - ft14 + ft15 + ft16 - ft17 - ft18 + prc(0.5) * fz;

        u2 = ux1 * ux1 + uy1 * uy1 + uz1 * uz1;

        //PDFs summations for computation efficiency purpose
        sum1 = ft1 + ft2 + ft3 + ft4 + ft5 + ft6;
        sum2 = ft7 + ft8 + ft9 + ft10 + ft11 + ft12 + ft13 + ft14 + ft15 + ft16 + ft17 + ft18;
        sum3 = ft7 - ft8 + ft9 - ft10 + ft11 - ft12 + ft13 - ft14;
        sum4 = ft7 + ft8 - ft9 - ft10 + ft15 - ft16 + ft17 - ft18;
        sum5 = ft11 + ft12 - ft13 - ft14 + ft15 + ft16 - ft17 - ft18;
        sum6 = prc(2.) * (ft1 + ft2) - ft3 - ft4 - ft5 - ft6;
        sum7 = ft7 + ft8 + ft9 + ft10 + ft11 + ft12 + ft13 + ft14 - prc(2.) * (ft15 + ft16 + ft17 + ft18);
        sum8 = ft3 + ft4 - ft5 - ft6;
        sum9 = ft7 + ft8 + ft9 + ft10 - ft11 - ft12 - ft13 - ft14;
        //PDF to moment
        m_rho = den;
        m_e = prc(-30.) * ft0 - prc(11.) * sum1 + prc(8.) * sum2;
        m_e2 = prc(12.) * ft0 - prc(4.) * sum1 + sum2;
        m_jx = ft1 - ft2 + sum3;
        m_qx = prc(-4.) * (ft1 - ft2) + sum3;
        m_jy = ft3 - ft4 + sum4;
        m_qy = prc(-4.) * (ft3 - ft4) + sum4;
        m_jz = ft5 - ft6 + sum5;
        m_qz = prc(-4.) * (ft5 - ft6) + sum5;
        m_3pxx = sum6 + sum7;
        m_3pixx = prc(-2.) * sum6 + sum7;
        m_pww = sum8 + sum9;
        m_piww = prc(-2.) * sum8 + sum9;
        m_pxy = ft7 - ft8 - ft9 + ft10;
        m_pyz = ft15 - ft16 - ft17 + ft18;
        m_pzx = ft11 - ft12 - ft13 + ft14;
        m_tx = ft7 - ft8 + ft9 - ft10 - ft11 + ft12 - ft13 + ft14;
        m_ty = -ft7 - ft8 + ft9 + ft10 + ft15 - ft16 + ft17 - ft18;
        m_tz = ft11 + ft12 - ft13 - ft14 - ft15 - ft16 + ft17 + ft18;

        // relaxtion in moment space
        m_e = m_e - s_e * (m_e - (prc(-11.0) * den + prc(19.0) * u2)) + (prc(38.) - prc(19.) * s_e) * (fx * ux1 + fy * uy1 + fz * uz1);                           //m1
        m_e2 = m_e2 - s_e2 * (m_e2 - (mrt_e2_coef1_d * den + mrt_e2_coef2_d * u2)) + (prc(-11.) + prc(5.5) * s_e2) * (fx * ux1 + fy * uy1 + fz * uz1);                     //m2
        m_jx = m_jx + fx;                                                                                                   //m3
        m_qx = m_qx - s_q * (m_qx - (prc(-0.666666666666666667) * ux1)) + (prc(-0.666666666666666667) + prc(0.333333333333333333) * s_q) * fx; //m4
        m_jy = m_jy + fy;                                                                                                   //m5
        m_qy = m_qy - s_q * (m_qy - (prc(-0.666666666666666667) * uy1)) + (prc(-0.666666666666666667) + prc(0.333333333333333333) * s_q) * fy; //m6
        m_jz = m_jz + fz;                                                                                                   //m7
        m_qz = m_qz - s_q * (m_qz - (prc(-0.666666666666666667) * uz1)) + (prc(-0.666666666666666667) + prc(0.333333333333333333) * s_q) * fz; //m8

        m_3pxx = m_3pxx - s_nu * (m_3pxx - (prc(3.) * ux1 * ux1 - u2)) + (prc(2.) - s_nu) * (prc(2.) * fx * ux1 - fy * uy1 - fz * uz1);                           //m9
        m_3pixx = m_3pixx - s_pi * (m_3pixx - mrt_omega_xx_d * (prc(3.) * ux1 * ux1 - u2)) + (prc(1.) - prc(0.5) * s_pi) * (prc(-2.) * fx * ux1 + fy * uy1 + fz * uz1);         //m10
        m_pww = m_pww - s_nu * (m_pww - (uy1 * uy1 - uz1 * uz1)) + (prc(2.) - s_nu) * (fy * uy1 - fz * uz1);                                     //m11
        m_piww = m_piww - s_pi * (m_piww - mrt_omega_xx_d * (uy1 * uy1 - uz1 * uz1)) + (prc(1.) - prc(0.5) * s_pi) * (-fy * uy1 + fz * uz1);                   //m12
        m_pxy = m_pxy - s_nu * (m_pxy - (ux1 * uy1)) + (prc(1.) - prc(0.5) * s_nu) * (fx * uy1 + fy * ux1);                                        //m13
        m_pyz = m_pyz - s_nu * (m_pyz - (uy1 * uz1)) + (prc(1.) - prc(0.5) * s_nu) * (fy * uz1 + fz * uy1);                                        //m14
        m_pzx = m_pzx - s_nu * (m_pzx - (ux1 * uz1)) + (prc(1.) - prc(0.5) * s_nu) * (fx * uz1 + fz * ux1);                                        //m15
        m_tx = m_tx - s_t * (m_tx); //m16
        m_ty = m_ty - s_t * (m_ty); //m17
        m_tz = m_tz - s_t * (m_tz); //m18

        // transform back to PDFs
        // coeffcients for performance
        m_rho = mrt_coef1_d * m_rho;            //1 / 19
        m_e = mrt_coef2_d * m_e;              //1 / 2394
        m_e2 = mrt_coef3_d * m_e2;              //1 / 252
        m_jx = prc(0.1) * m_jx;
        m_qx = prc(0.025) * m_qx;
        m_jy = prc(0.1) * m_jy;
        m_qy = prc(0.025) * m_qy;
        m_jz = prc(0.1) * m_jz;
        m_qz = prc(0.025) * m_qz;
        m_3pxx = prc(2.) * mrt_coef4_d * m_3pxx;      //1 / 36
        m_3pixx = mrt_coef4_d * m_3pixx;        //1 / 72
        m_pww = prc(6.) * mrt_coef4_d * m_pww;       //1 / 12
        m_piww = prc(3.) * mrt_coef4_d * m_piww;     //1 / 24
        m_pxy = prc(0.25) * m_pxy;
        m_pyz = prc(0.25) * m_pyz;
        m_pzx = prc(0.25) * m_pzx;
        m_tx = prc(0.125) * m_tx;
        m_ty = prc(0.125) * m_ty;
        m_tz = prc(0.125) * m_tz;
        sum1 = m_rho - prc(11.) * m_e - prc(4.) * m_e2;
        sum2 = prc(2.) * m_3pxx - prc(4.) * m_3pixx;
        sum3 = m_pww - prc(2.) * m_piww;
        sum4 = m_rho + prc(8.) * m_e + m_e2;
        sum5 = m_jx + m_qx;
        sum6 = m_jy + m_qy;
        sum7 = m_jz + m_qz;
        sum8 = m_3pxx + m_3pixx;
        sum9 = m_pww + m_piww;

        ft0 = m_rho - prc(30.) * m_e + prc(12.) * m_e2;
        ft1 = sum1 + m_jx - prc(4.) * m_qx + sum2;
        ft2 = sum1 - m_jx + prc(4.) * m_qx + sum2;
        ft3 = sum1 + m_jy - prc(4.) * m_qy - prc(0.5) * sum2 + sum3;
        ft4 = sum1 - m_jy + prc(4.) * m_qy - prc(0.5) * sum2 + sum3;
        ft5 = sum1 + m_jz - prc(4.) * m_qz - prc(0.5) * sum2 - sum3;
        ft6 = sum1 - m_jz + prc(4.) * m_qz - prc(0.5) * sum2 - sum3;
        ft7 = sum4 + sum5 + sum6 + sum8 + sum9 + m_pxy + m_tx - m_ty;
        ft8 = sum4 - sum5 + sum6 + sum8 + sum9 - m_pxy - m_tx - m_ty;
        ft9 = sum4 + sum5 - sum6 + sum8 + sum9 - m_pxy + m_tx + m_ty;
        ft10 = sum4 - sum5 - sum6 + sum8 + sum9 + m_pxy - m_tx + m_ty;
        ft11 = sum4 + sum5 + sum7 + sum8 - sum9 + m_pzx - m_tx + m_tz;
        ft12 = sum4 - sum5 + sum7 + sum8 - sum9 - m_pzx + m_tx + m_tz;
        ft13 = sum4 + sum5 - sum7 + sum8 - sum9 - m_pzx - m_tx - m_tz;
        ft14 = sum4 - sum5 - sum7 + sum8 - sum9 + m_pzx + m_tx - m_tz;
        ft15 = sum4 + sum6 + sum7 - sum8 * prc(2.) + m_pyz + m_ty - m_tz;
        ft16 = sum4 - sum6 + sum7 - sum8 * prc(2.) - m_pyz - m_ty - m_tz;
        ft17 = sum4 + sum6 - sum7 - sum8 * prc(2.) - m_pyz + m_ty + m_tz;
        ft18 = sum4 - sum6 - sum7 - sum8 * prc(2.) + m_pyz - m_ty + m_tz;
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MRT kernel, repeated part~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // ++++++++++++ - recoloring & streaming to opposite direction++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        tmp1 = rho1 / den;
        pdf[i_f1(i, j, k, 0, 0)] = tmp1 * ft0;
        pdf[i_f1(i, j, k, 0, 1)] = ft0 * (prc(1.) - tmp1);

        // R - K method
        tmp = rho1 * rho2 * lbm_beta_d / den;

        g1t1 = tmp1 * ft1 + w_equ_1_d * tmp * (cnx);
        g1t2 = tmp1 * ft2 + w_equ_1_d * tmp * (-cnx);
        g1t3 = tmp1 * ft3 + w_equ_1_d * tmp * (cny);
        g1t4 = tmp1 * ft4 + w_equ_1_d * tmp * (-cny);
        g1t5 = tmp1 * ft5 + w_equ_1_d * tmp * (cnz);
        g1t6 = tmp1 * ft6 + w_equ_1_d * tmp * (-cnz);
        g1t7 = tmp1 * ft7 + RK_weight2_d * tmp * (cnx + cny);
        g1t8 = tmp1 * ft8 + RK_weight2_d * tmp * (-cnx + cny);
        g1t9 = tmp1 * ft9 + RK_weight2_d * tmp * (cnx - cny);
        g1t10 = tmp1 * ft10 + RK_weight2_d * tmp * (-cnx - cny);
        g1t11 = tmp1 * ft11 + RK_weight2_d * tmp * (cnx + cnz);
        g1t12 = tmp1 * ft12 + RK_weight2_d * tmp * (-cnx + cnz);
        g1t13 = tmp1 * ft13 + RK_weight2_d * tmp * (cnx - cnz);
        g1t14 = tmp1 * ft14 + RK_weight2_d * tmp * (-cnx - cnz);
        g1t15 = tmp1 * ft15 + RK_weight2_d * tmp * (cny + cnz);
        g1t16 = tmp1 * ft16 + RK_weight2_d * tmp * (-cny + cnz);
        g1t17 = tmp1 * ft17 + RK_weight2_d * tmp * (cny - cnz);
        g1t18 = tmp1 * ft18 + RK_weight2_d * tmp * (-cny - cnz);

        g2t1 = ft1 - g1t1;
        g2t2 = ft2 - g1t2;
        g2t3 = ft3 - g1t3;
        g2t4 = ft4 - g1t4;
        g2t5 = ft5 - g1t5;
        g2t6 = ft6 - g1t6;
        g2t7 = ft7 - g1t7;
        g2t8 = ft8 - g1t8;
        g2t9 = ft9 - g1t9;
        g2t10 = ft10 - g1t10;
        g2t11 = ft11 - g1t11;
        g2t12 = ft12 - g1t12;
        g2t13 = ft13 - g1t13;
        g2t14 = ft14 - g1t14;
        g2t15 = ft15 - g1t15;
        g2t16 = ft16 - g1t16;
        g2t17 = ft17 - g1t17;
        g2t18 = ft18 - g1t18;

        // ++++++++ + -AA pattern push step++++++++++++
        pdf[i_f1(i + 1, j, k, 2, 0)] = g1t1;
        pdf[i_f1(i - 1, j, k, 1, 0)] = g1t2;
        pdf[i_f1(i, j + 1, k, 4, 0)] = g1t3;
        pdf[i_f1(i, j - 1, k, 3, 0)] = g1t4;
        pdf[i_f1(i, j, k + 1, 6, 0)] = g1t5;
        pdf[i_f1(i, j, k - 1, 5, 0)] = g1t6;
        pdf[i_f1(i + 1, j + 1, k, 10, 0)] = g1t7;
        pdf[i_f1(i - 1, j + 1, k, 9, 0)] = g1t8;
        pdf[i_f1(i + 1, j - 1, k, 8, 0)] = g1t9;
        pdf[i_f1(i - 1, j - 1, k, 7, 0)] = g1t10;
        pdf[i_f1(i + 1, j, k + 1, 14, 0)] = g1t11;
        pdf[i_f1(i - 1, j, k + 1, 13, 0)] = g1t12;
        pdf[i_f1(i + 1, j, k - 1, 12, 0)] = g1t13;
        pdf[i_f1(i - 1, j, k - 1, 11, 0)] = g1t14;
        pdf[i_f1(i, j + 1, k + 1, 18, 0)] = g1t15;
        pdf[i_f1(i, j - 1, k + 1, 17, 0)] = g1t16;
        pdf[i_f1(i, j + 1, k - 1, 16, 0)] = g1t17;
        pdf[i_f1(i, j - 1, k - 1, 15, 0)] = g1t18;

        pdf[i_f1(i + 1, j, k, 2, 1)] = g2t1;
        pdf[i_f1(i - 1, j, k, 1, 1)] = g2t2;
        pdf[i_f1(i, j + 1, k, 4, 1)] = g2t3;
        pdf[i_f1(i, j - 1, k, 3, 1)] = g2t4;
        pdf[i_f1(i, j, k + 1, 6, 1)] = g2t5;
        pdf[i_f1(i, j, k - 1, 5, 1)] = g2t6;
        pdf[i_f1(i + 1, j + 1, k, 10, 1)] = g2t7;
        pdf[i_f1(i - 1, j + 1, k, 9, 1)] = g2t8;
        pdf[i_f1(i + 1, j - 1, k, 8, 1)] = g2t9;
        pdf[i_f1(i - 1, j - 1, k, 7, 1)] = g2t10;
        pdf[i_f1(i + 1, j, k + 1, 14, 1)] = g2t11;
        pdf[i_f1(i - 1, j, k + 1, 13, 1)] = g2t12;
        pdf[i_f1(i + 1, j, k - 1, 12, 1)] = g2t13;
        pdf[i_f1(i - 1, j, k - 1, 11, 1)] = g2t14;
        pdf[i_f1(i, j + 1, k + 1, 18, 1)] = g2t15;
        pdf[i_f1(i, j - 1, k + 1, 17, 1)] = g2t16;
        pdf[i_f1(i, j + 1, k - 1, 16, 1)] = g2t17;
        pdf[i_f1(i, j - 1, k - 1, 15, 1)] = g2t18;

    }

}

//=====================================================================================================================================
//----------------------even step kernel----------------------
//no steaming steps
//=====================================================================================================================================

__global__ void kernel_even_color_GPU(int ixmin, int ixmax, int iymin, int iymax, int izmin, int izmax,
    int* walls, T_P* pdf, T_P* phi, T_P* cn_x, T_P* cn_y, T_P* cn_z, T_P* curv, T_P* c_norm) {

    // Indexing (Thread)
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    T_P cnx, cny, cnz, rho1, rho2, tmp1;   //color model
    T_P sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9;
    T_P m_rho, m_e, m_e2, m_jx, m_qx, m_jy, m_qy, m_jz, m_qz, m_3pxx, m_3pixx, m_pww, m_piww, m_pxy, m_pyz, m_pzx, m_tx, m_ty, m_tz;
    T_P ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12, ft13, ft14, ft15, ft16, ft17, ft18;
    T_P g1t0, g1t1, g1t2, g1t3, g1t4, g1t5, g1t6, g1t7, g1t8, g1t9, g1t10, g1t11, g1t12, g1t13, g1t14, g1t15, g1t16, g1t17, g1t18;
    T_P g2t0, g2t1, g2t2, g2t3, g2t4, g2t5, g2t6, g2t7, g2t8, g2t9, g2t10, g2t11, g2t12, g2t13, g2t14, g2t15, g2t16, g2t17, g2t18;
    T_P s_e, s_e2, s_q, s_nu, s_pi, s_t;  //relaxation parameters
    T_P fx, fy, fz, ux1, uy1, uz1, den, tmp, omega, u2;

    if (walls[i_s2(i, j, k)] == 0 && i >= ixmin && i <= ixmax && j >= iymin && j <= iymax && k >= izmin && k <= izmax) {
        // +++++++++- AA pattern pull step++++++++++++
        g1t0 = pdf[i_f1(i, j, k, 0, 0)];
        g1t2 = pdf[i_f1(i, j, k, 1, 0)];
        g1t1 = pdf[i_f1(i, j, k, 2, 0)];
        g1t4 = pdf[i_f1(i, j, k, 3, 0)];
        g1t3 = pdf[i_f1(i, j, k, 4, 0)];
        g1t6 = pdf[i_f1(i, j, k, 5, 0)];
        g1t5 = pdf[i_f1(i, j, k, 6, 0)];
        g1t10 = pdf[i_f1(i, j, k, 7, 0)];
        g1t9 = pdf[i_f1(i, j, k, 8, 0)];
        g1t8 = pdf[i_f1(i, j, k, 9, 0)];
        g1t7 = pdf[i_f1(i, j, k, 10, 0)];
        g1t14 = pdf[i_f1(i, j, k, 11, 0)];
        g1t13 = pdf[i_f1(i, j, k, 12, 0)];
        g1t12 = pdf[i_f1(i, j, k, 13, 0)];
        g1t11 = pdf[i_f1(i, j, k, 14, 0)];
        g1t18 = pdf[i_f1(i, j, k, 15, 0)];
        g1t17 = pdf[i_f1(i, j, k, 16, 0)];
        g1t16 = pdf[i_f1(i, j, k, 17, 0)];
        g1t15 = pdf[i_f1(i, j, k, 18, 0)];

        g2t0 = pdf[i_f1(i, j, k, 0, 1)];
        g2t2 = pdf[i_f1(i, j, k, 1, 1)];
        g2t1 = pdf[i_f1(i, j, k, 2, 1)];
        g2t4 = pdf[i_f1(i, j, k, 3, 1)];
        g2t3 = pdf[i_f1(i, j, k, 4, 1)];
        g2t6 = pdf[i_f1(i, j, k, 5, 1)];
        g2t5 = pdf[i_f1(i, j, k, 6, 1)];
        g2t10 = pdf[i_f1(i, j, k, 7, 1)];
        g2t9 = pdf[i_f1(i, j, k, 8, 1)];
        g2t8 = pdf[i_f1(i, j, k, 9, 1)];
        g2t7 = pdf[i_f1(i, j, k, 10, 1)];
        g2t14 = pdf[i_f1(i, j, k, 11, 1)];
        g2t13 = pdf[i_f1(i, j, k, 12, 1)];
        g2t12 = pdf[i_f1(i, j, k, 13, 1)];
        g2t11 = pdf[i_f1(i, j, k, 14, 1)];
        g2t18 = pdf[i_f1(i, j, k, 15, 1)];
        g2t17 = pdf[i_f1(i, j, k, 16, 1)];
        g2t16 = pdf[i_f1(i, j, k, 17, 1)];
        g2t15 = pdf[i_f1(i, j, k, 18, 1)];

        // ft: bulk PDF
        ft0 = g1t0 + g2t0;
        ft1 = g1t1 + g2t1;
        ft2 = g1t2 + g2t2;
        ft3 = g1t3 + g2t3;
        ft4 = g1t4 + g2t4;
        ft5 = g1t5 + g2t5;
        ft6 = g1t6 + g2t6;
        ft7 = g1t7 + g2t7;
        ft8 = g1t8 + g2t8;
        ft9 = g1t9 + g2t9;
        ft10 = g1t10 + g2t10;
        ft11 = g1t11 + g2t11;
        ft12 = g1t12 + g2t12;
        ft13 = g1t13 + g2t13;
        ft14 = g1t14 + g2t14;
        ft15 = g1t15 + g2t15;
        ft16 = g1t16 + g2t16;
        ft17 = g1t17 + g2t17;
        ft18 = g1t18 + g2t18;

        // order parameter
        rho1 = g1t0 + g1t1 + g1t2 + g1t3 + g1t4 + g1t5 + g1t6 + g1t7 + g1t8 + g1t9 + g1t10 + g1t11 + g1t12 + g1t13 + g1t14 + g1t15 + g1t16 + g1t17 + g1t18;
        rho2 = g2t0 + g2t1 + g2t2 + g2t3 + g2t4 + g2t5 + g2t6 + g2t7 + g2t8 + g2t9 + g2t10 + g2t11 + g2t12 + g2t13 + g2t14 + g2t15 + g2t16 + g2t17 + g2t18;

        T_P phi_loc = (rho1 - rho2) / (rho1 + rho2);
        phi[i_s4(i, j, k)] = phi_loc;

        cnx = cn_x[i_s2(i, j, k)];
        cny = cn_y[i_s2(i, j, k)];
        cnz = cn_z[i_s2(i, j, k)];

        tmp = prc(0.5) * lbm_gamma_d * curv[i_s1(i, j, k)] * c_norm[i_s2(i, j, k)];
        fx = tmp * cnx;
        fy = tmp * cny;
        fz = tmp * cnz + force_z_d;   // body force force_z along flow direction

        // ++++++++++++ - MRT COLLISION++++++++++++ + -
        // select viscosity++++++++++++++++++++++ + -
        omega = prc(1.) / (prc(6.) / ((prc(1.0) + phi_loc) * la_nui1_d + (prc(1.0) - phi_loc) * la_nui2_d) + prc(0.5));
        // MRT PARAMETERS
        s_nu = omega;
#if (mrt==1)
        // ************bounceback opt * ***********
        s_e = omega;
        s_e2 = omega;
        s_pi = omega;
        s_q = prc(8.0) * (prc(2.0) - omega) / (prc(8.0) - omega);
        s_t = s_q;
#elif (mrt==2)
        //************original opt * ***********
        s_e = prc(1.19);
        s_e2 = prc(1.4);
        s_pi = prc(1.4);
        s_q = prc(1.2);
        s_t = prc(1.98);
#elif (mrt==3)
        // ************SRT * ***********
        s_e = omega;
        s_e2 = omega;
        s_pi = omega;
        s_q = omega;
        s_t = omega;
#elif (mrt==4)           
        // ************advection opt * ***********
        s_e = omega;
        s_e2 = omega;
        s_pi = omega;
        s_q = (prc(6.) - prc(3.) * omega) / (prc(3.) - omega);
        s_t = omega;
#endif

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MRT kernel, repeated part~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ++++++++ + -!calculate macroscopic variables++++++++++++
        den = rho1 + rho2;

        ux1 = ft1 - ft2 + ft7 - ft8 + ft9 - ft10 + ft11 - ft12 + ft13 - ft14 + prc(0.5) * fx;
        uy1 = ft3 - ft4 + ft7 + ft8 - ft9 - ft10 + ft15 - ft16 + ft17 - ft18 + prc(0.5) * fy;
        uz1 = ft5 - ft6 + ft11 + ft12 - ft13 - ft14 + ft15 + ft16 - ft17 - ft18 + prc(0.5) * fz;
        u2 = ux1 * ux1 + uy1 * uy1 + uz1 * uz1;

        // PDFs summations for computation efficiency purpose
        sum1 = ft1 + ft2 + ft3 + ft4 + ft5 + ft6;
        sum2 = ft7 + ft8 + ft9 + ft10 + ft11 + ft12 + ft13 + ft14 + ft15 + ft16 + ft17 + ft18;
        sum3 = ft7 - ft8 + ft9 - ft10 + ft11 - ft12 + ft13 - ft14;
        sum4 = ft7 + ft8 - ft9 - ft10 + ft15 - ft16 + ft17 - ft18;
        sum5 = ft11 + ft12 - ft13 - ft14 + ft15 + ft16 - ft17 - ft18;
        sum6 = prc(2.) * (ft1 + ft2) - ft3 - ft4 - ft5 - ft6;
        sum7 = ft7 + ft8 + ft9 + ft10 + ft11 + ft12 + ft13 + ft14 - prc(2.) * (ft15 + ft16 + ft17 + ft18);
        sum8 = ft3 + ft4 - ft5 - ft6;
        sum9 = ft7 + ft8 + ft9 + ft10 - ft11 - ft12 - ft13 - ft14;
        // PDF to moment
        m_rho = den;
        m_e = prc(-30.) * ft0 - prc(11.) * sum1 + prc(8.) * sum2;
        m_e2 = prc(12.) * ft0 - prc(4.) * sum1 + sum2;
        m_jx = ft1 - ft2 + sum3;
        m_qx = prc(-4.) * (ft1 - ft2) + sum3;
        m_jy = ft3 - ft4 + sum4;
        m_qy = prc(-4.) * (ft3 - ft4) + sum4;
        m_jz = ft5 - ft6 + sum5;
        m_qz = prc(-4.) * (ft5 - ft6) + sum5;
        m_3pxx = sum6 + sum7;
        m_3pixx = prc(-2.) * sum6 + sum7;
        m_pww = sum8 + sum9;
        m_piww = prc(-2.) * sum8 + sum9;
        m_pxy = ft7 - ft8 - ft9 + ft10;
        m_pyz = ft15 - ft16 - ft17 + ft18;
        m_pzx = ft11 - ft12 - ft13 + ft14;
        m_tx = ft7 - ft8 + ft9 - ft10 - ft11 + ft12 - ft13 + ft14;
        m_ty = -ft7 - ft8 + ft9 + ft10 + ft15 - ft16 + ft17 - ft18;
        m_tz = ft11 + ft12 - ft13 - ft14 - ft15 - ft16 + ft17 + ft18;

        // relaxtion in moment space
        m_e = m_e - s_e * (m_e - (prc(-11.0) * den + prc(19.0) * u2)) + (prc(38.) - prc(19.) * s_e) * (fx * ux1 + fy * uy1 + fz * uz1);                           //m1
        m_e2 = m_e2 - s_e2 * (m_e2 - (mrt_e2_coef1_d * den + mrt_e2_coef2_d * u2)) + (prc(-11.) + prc(5.5) * s_e2) * (fx * ux1 + fy * uy1 + fz * uz1);                     //m2
        m_jx = m_jx + fx;                                                                                                  //m3
        m_qx = m_qx - s_q * (m_qx - (prc(-0.666666666666666667) * ux1)) + (prc(-0.666666666666666667) + prc(0.333333333333333333) * s_q) * fx; //m4
        m_jy = m_jy + fy;                                                                                                   //m5
        m_qy = m_qy - s_q * (m_qy - (prc(-0.666666666666666667) * uy1)) + (prc(-0.666666666666666667) + prc(0.333333333333333333) * s_q) * fy; //m6
        m_jz = m_jz + fz;                                                                                                  //m7
        m_qz = m_qz - s_q * (m_qz - (prc(-0.666666666666666667) * uz1)) + (prc(-0.666666666666666667) + prc(0.333333333333333333) * s_q) * fz; //m8

        m_3pxx = m_3pxx - s_nu * (m_3pxx - (prc(3.) * ux1 * ux1 - u2)) + (prc(2.) - s_nu) * (prc(2.) * fx * ux1 - fy * uy1 - fz * uz1);                            //m9
        m_3pixx = m_3pixx - s_pi * (m_3pixx - mrt_omega_xx_d * (prc(3.) * ux1 * ux1 - u2)) + (prc(1.) - prc(0.5) * s_pi) * (prc(-2.) * fx * ux1 + fy * uy1 + fz * uz1);         //m10
        m_pww = m_pww - s_nu * (m_pww - (uy1 * uy1 - uz1 * uz1)) + (prc(2.) - s_nu) * (fy * uy1 - fz * uz1);                                      //m11
        m_piww = m_piww - s_pi * (m_piww - mrt_omega_xx_d * (uy1 * uy1 - uz1 * uz1)) + (prc(1.) - prc(0.5) * s_pi) * (-fy * uy1 + fz * uz1);                   //m12
        m_pxy = m_pxy - s_nu * (m_pxy - (ux1 * uy1)) + (prc(1.) - prc(0.5) * s_nu) * (fx * uy1 + fy * ux1);                                        //m13
        m_pyz = m_pyz - s_nu * (m_pyz - (uy1 * uz1)) + (prc(1.) - prc(0.5) * s_nu) * (fy * uz1 + fz * uy1);                                        //m14
        m_pzx = m_pzx - s_nu * (m_pzx - (ux1 * uz1)) + (prc(1.) - prc(0.5) * s_nu) * (fx * uz1 + fz * ux1);                                        //m15
        m_tx = m_tx - s_t * (m_tx); //m16
        m_ty = m_ty - s_t * (m_ty); //m17
        m_tz = m_tz - s_t * (m_tz); //m18


        // transform back to PDFs
        // coeffcients for performance
        m_rho = mrt_coef1_d * m_rho;            //1 / 19
        m_e = mrt_coef2_d * m_e;              //1 / 2394
        m_e2 = mrt_coef3_d * m_e2;              //1 / 252
        m_jx = prc(0.1) * m_jx;
        m_qx = prc(0.025) * m_qx;
        m_jy = prc(0.1) * m_jy;
        m_qy = prc(0.025) * m_qy;
        m_jz = prc(0.1) * m_jz;
        m_qz = prc(0.025) * m_qz;
        m_3pxx = prc(2.) * mrt_coef4_d * m_3pxx;      //1 / 36
        m_3pixx = mrt_coef4_d * m_3pixx;          // 1 / 72
        m_pww = prc(6.) * mrt_coef4_d * m_pww;      //!1 / 12
        m_piww = prc(3.) * mrt_coef4_d * m_piww;     //1 / 24
        m_pxy = prc(0.25) * m_pxy;
        m_pyz = prc(0.25) * m_pyz;
        m_pzx = prc(0.25) * m_pzx;
        m_tx = prc(0.125) * m_tx;
        m_ty = prc(0.125) * m_ty;
        m_tz = prc(0.125) * m_tz;
        sum1 = m_rho - prc(11.) * m_e - prc(4.) * m_e2;
        sum2 = prc(2.) * m_3pxx - prc(4.) * m_3pixx;
        sum3 = m_pww - prc(2.) * m_piww;
        sum4 = m_rho + prc(8.) * m_e + m_e2;
        sum5 = m_jx + m_qx;
        sum6 = m_jy + m_qy;
        sum7 = m_jz + m_qz;
        sum8 = m_3pxx + m_3pixx;
        sum9 = m_pww + m_piww;

        ft0 = m_rho - prc(30.) * m_e + prc(12.) * m_e2;
        ft1 = sum1 + m_jx - prc(4.) * m_qx + sum2;
        ft2 = sum1 - m_jx + prc(4.) * m_qx + sum2;
        ft3 = sum1 + m_jy - prc(4.) * m_qy - prc(0.5) * sum2 + sum3;
        ft4 = sum1 - m_jy + prc(4.) * m_qy - prc(0.5) * sum2 + sum3;
        ft5 = sum1 + m_jz - prc(4.) * m_qz - prc(0.5) * sum2 - sum3;
        ft6 = sum1 - m_jz + prc(4.) * m_qz - prc(0.5) * sum2 - sum3;
        ft7 = sum4 + sum5 + sum6 + sum8 + sum9 + m_pxy + m_tx - m_ty;
        ft8 = sum4 - sum5 + sum6 + sum8 + sum9 - m_pxy - m_tx - m_ty;
        ft9 = sum4 + sum5 - sum6 + sum8 + sum9 - m_pxy + m_tx + m_ty;
        ft10 = sum4 - sum5 - sum6 + sum8 + sum9 + m_pxy - m_tx + m_ty;
        ft11 = sum4 + sum5 + sum7 + sum8 - sum9 + m_pzx - m_tx + m_tz;
        ft12 = sum4 - sum5 + sum7 + sum8 - sum9 - m_pzx + m_tx + m_tz;
        ft13 = sum4 + sum5 - sum7 + sum8 - sum9 - m_pzx - m_tx - m_tz;
        ft14 = sum4 - sum5 - sum7 + sum8 - sum9 + m_pzx + m_tx - m_tz;
        ft15 = sum4 + sum6 + sum7 - sum8 * prc(2.) + m_pyz + m_ty - m_tz;
        ft16 = sum4 - sum6 + sum7 - sum8 * prc(2.) - m_pyz - m_ty - m_tz;
        ft17 = sum4 + sum6 - sum7 - sum8 * prc(2.) - m_pyz + m_ty + m_tz;
        ft18 = sum4 - sum6 - sum7 - sum8 * prc(2.) + m_pyz - m_ty + m_tz;
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MRT kernel, repeated part~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // ++++++++++++ - recoloring++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        tmp1 = rho1 / den;
        pdf[i_f1(i, j, k, 0, 0)] = tmp1 * ft0;
        pdf[i_f1(i, j, k, 0, 1)] = ft0 * (prc(1.) - tmp1);

        // R - K method
        tmp = rho1 * rho2 * lbm_beta_d / den;

        g1t1 = tmp1 * ft1 + w_equ_1_d * tmp * (cnx);
        g1t2 = tmp1 * ft2 + w_equ_1_d * tmp * (-cnx);
        g1t3 = tmp1 * ft3 + w_equ_1_d * tmp * (cny);
        g1t4 = tmp1 * ft4 + w_equ_1_d * tmp * (-cny);
        g1t5 = tmp1 * ft5 + w_equ_1_d * tmp * (cnz);
        g1t6 = tmp1 * ft6 + w_equ_1_d * tmp * (-cnz);
        g1t7 = tmp1 * ft7 + RK_weight2_d * tmp * (cnx + cny);
        g1t8 = tmp1 * ft8 + RK_weight2_d * tmp * (-cnx + cny);
        g1t9 = tmp1 * ft9 + RK_weight2_d * tmp * (cnx - cny);
        g1t10 = tmp1 * ft10 + RK_weight2_d * tmp * (-cnx - cny);
        g1t11 = tmp1 * ft11 + RK_weight2_d * tmp * (cnx + cnz);
        g1t12 = tmp1 * ft12 + RK_weight2_d * tmp * (-cnx + cnz);
        g1t13 = tmp1 * ft13 + RK_weight2_d * tmp * (cnx - cnz);
        g1t14 = tmp1 * ft14 + RK_weight2_d * tmp * (-cnx - cnz);
        g1t15 = tmp1 * ft15 + RK_weight2_d * tmp * (cny + cnz);
        g1t16 = tmp1 * ft16 + RK_weight2_d * tmp * (-cny + cnz);
        g1t17 = tmp1 * ft17 + RK_weight2_d * tmp * (cny - cnz);
        g1t18 = tmp1 * ft18 + RK_weight2_d * tmp * (-cny - cnz);

        g2t1 = ft1 - g1t1;
        g2t2 = ft2 - g1t2;
        g2t3 = ft3 - g1t3;
        g2t4 = ft4 - g1t4;
        g2t5 = ft5 - g1t5;
        g2t6 = ft6 - g1t6;
        g2t7 = ft7 - g1t7;
        g2t8 = ft8 - g1t8;
        g2t9 = ft9 - g1t9;
        g2t10 = ft10 - g1t10;
        g2t11 = ft11 - g1t11;
        g2t12 = ft12 - g1t12;
        g2t13 = ft13 - g1t13;
        g2t14 = ft14 - g1t14;
        g2t15 = ft15 - g1t15;
        g2t16 = ft16 - g1t16;
        g2t17 = ft17 - g1t17;
        g2t18 = ft18 - g1t18;

        // ++++++++ + -AA pattern++++++++++++
        pdf[i_f1(i, j, k, 1, 0)] = g1t1;
        pdf[i_f1(i, j, k, 2, 0)] = g1t2;
        pdf[i_f1(i, j, k, 3, 0)] = g1t3;
        pdf[i_f1(i, j, k, 4, 0)] = g1t4;
        pdf[i_f1(i, j, k, 5, 0)] = g1t5;
        pdf[i_f1(i, j, k, 6, 0)] = g1t6;
        pdf[i_f1(i, j, k, 7, 0)] = g1t7;
        pdf[i_f1(i, j, k, 8, 0)] = g1t8;
        pdf[i_f1(i, j, k, 9, 0)] = g1t9;
        pdf[i_f1(i, j, k, 10, 0)] = g1t10;
        pdf[i_f1(i, j, k, 11, 0)] = g1t11;
        pdf[i_f1(i, j, k, 12, 0)] = g1t12;
        pdf[i_f1(i, j, k, 13, 0)] = g1t13;
        pdf[i_f1(i, j, k, 14, 0)] = g1t14;
        pdf[i_f1(i, j, k, 15, 0)] = g1t15;
        pdf[i_f1(i, j, k, 16, 0)] = g1t16;
        pdf[i_f1(i, j, k, 17, 0)] = g1t17;
        pdf[i_f1(i, j, k, 18, 0)] = g1t18;

        pdf[i_f1(i, j, k, 1, 1)] = g2t1;
        pdf[i_f1(i, j, k, 2, 1)] = g2t2;
        pdf[i_f1(i, j, k, 3, 1)] = g2t3;
        pdf[i_f1(i, j, k, 4, 1)] = g2t4;
        pdf[i_f1(i, j, k, 5, 1)] = g2t5;
        pdf[i_f1(i, j, k, 6, 1)] = g2t6;
        pdf[i_f1(i, j, k, 7, 1)] = g2t7;
        pdf[i_f1(i, j, k, 8, 1)] = g2t8;
        pdf[i_f1(i, j, k, 9, 1)] = g2t9;
        pdf[i_f1(i, j, k, 10, 1)] = g2t10;
        pdf[i_f1(i, j, k, 11, 1)] = g2t11;
        pdf[i_f1(i, j, k, 12, 1)] = g2t12;
        pdf[i_f1(i, j, k, 13, 1)] = g2t13;
        pdf[i_f1(i, j, k, 14, 1)] = g2t14;
        pdf[i_f1(i, j, k, 15, 1)] = g2t15;
        pdf[i_f1(i, j, k, 16, 1)] = g2t16;
        pdf[i_f1(i, j, k, 17, 1)] = g2t17;
        pdf[i_f1(i, j, k, 18, 1)] = g2t18;
    }

}

#pragma endregion (kernel_multiphase)

#pragma region (color gradient)
/* ~~~~~~~~~~~~~~~~~~~~~~~ extrapolate phi values to solid boundary nodes ~~~~~~~~~~~~~~~~~~ */
__global__ void extrapolate_phi_toSolid(int* walls_type, T_P* phi) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1 - 3;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1 - 3;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1 - 3;

    if (i <= nxGlobal_d + 3 && j<= nyGlobal_d + 3 && k <= nzGlobal_d + 3) {
        int node_type_loc = walls_type[i_s4(i, j, k)];
        if (node_type_loc == 2) { // solid boundary node
            T_P phi_sum = prc(0.), weight_sum = prc(0.);
            for (int q = 1; q < 19; q++) {
                int iex = i + ex_d[q];
                int iey = j + ey_d[q];
                int iez = k + ez_d[q];
                int node_type_neb = walls_type[i_s4(iex, iey, iez)];
                if (node_type_neb <= 0) {
                    phi_sum += phi[i_s4(iex, iey, iez)] * w_equ_d[q];
                    weight_sum += w_equ_d[q];
                }
            }
            phi[i_s4(i, j, k)] = phi_sum / weight_sum;
        }
    }

}
/* ~~~~~~~~~~~~~~~~~~ calculate normal directions of interfaces from phi gradient ~~~~~~~~~~~~~~~~~~ */
__global__ void normalDirectionsOfInterfaces(int* walls, T_P* phi, T_P* cn_x, T_P* cn_y, T_P* cn_z, T_P* c_norm) {
    //int overlap_temp = 2;
    // Indexing (Thread)
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1 - 2;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1 - 2;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1 - 2;

    if (i < nxGlobal_d + 4 && j < nyGlobal_d + 4 && k < nzGlobal_d + 4) {
        cn_x[i_s2(i, j, k)] =
            ISO4_d[1 - 1] * (phi[i_s4(i + 1, j, k)] - phi[i_s4(i - 1, j, k)]) +

            ISO4_d[2 - 1] * (
                phi[i_s4(i + 1, j + 1, k)] - phi[i_s4(i - 1, j - 1, k)] +
                phi[i_s4(i + 1, j - 1, k)] - phi[i_s4(i - 1, j + 1, k)] +
                phi[i_s4(i + 1, j, k + 1)] - phi[i_s4(i - 1, j, k - 1)] +
                phi[i_s4(i + 1, j, k - 1)] - phi[i_s4(i - 1, j, k + 1)]);

        cn_y[i_s2(i, j, k)] =
            ISO4_d[1 - 1] * (phi[i_s4(i, j + 1, k)] - phi[i_s4(i, j - 1, k)]) +

            ISO4_d[2 - 1] * (
                phi[i_s4(i + 1, j + 1, k)] - phi[i_s4(i - 1, j - 1, k)] +
                phi[i_s4(i - 1, j + 1, k)] - phi[i_s4(i + 1, j - 1, k)] +
                phi[i_s4(i, j + 1, k + 1)] - phi[i_s4(i, j - 1, k - 1)] +
                phi[i_s4(i, j + 1, k - 1)] - phi[i_s4(i, j - 1, k + 1)]);


        cn_z[i_s2(i, j, k)] =
            ISO4_d[1 - 1] * (phi[i_s4(i, j, k + 1)] - phi[i_s4(i, j, k - 1)]) +

            ISO4_d[2 - 1] * (
                phi[i_s4(i + 1, j, k + 1)] - phi[i_s4(i - 1, j, k - 1)] +
                phi[i_s4(i - 1, j, k + 1)] - phi[i_s4(i + 1, j, k - 1)] +
                phi[i_s4(i, j + 1, k + 1)] - phi[i_s4(i, j - 1, k - 1)] +
                phi[i_s4(i, j - 1, k + 1)] - phi[i_s4(i, j + 1, k - 1)]);

        c_norm[i_s2(i, j, k)] = prc(sqrt)(cn_x[i_s2(i, j, k)] * cn_x[i_s2(i, j, k)] + cn_y[i_s2(i, j, k)] * cn_y[i_s2(i, j, k)] + cn_z[i_s2(i, j, k)] * cn_z[i_s2(i, j, k)]);

        if (c_norm[i_s2(i, j, k)] < prc(1e-6) || walls[i_s2(i, j, k)] == 1) {
            cn_x[i_s2(i, j, k)] = prc(0.);
            cn_y[i_s2(i, j, k)] = prc(0.);
            cn_z[i_s2(i, j, k)] = prc(0.);
            c_norm[i_s2(i, j, k)] = prc(0.);
        }
        else {
            cn_x[i_s2(i, j, k)] = cn_x[i_s2(i, j, k)] / c_norm[i_s2(i, j, k)];
            cn_y[i_s2(i, j, k)] = cn_y[i_s2(i, j, k)] / c_norm[i_s2(i, j, k)];
            cn_z[i_s2(i, j, k)] = cn_z[i_s2(i, j, k)] / c_norm[i_s2(i, j, k)];    //normalized color gradient - interface normal direction
        }
    }
}
/* ~~~~~~~~~~~~~~ extrapolate normal direction info to solid boundary nodes, to minimize unbalanced forces ~~~~~~~~~~~~~~ */
__global__ void alter_color_gradient_solid_surface_GPU(int* walls_type, T_P* cn_x, T_P* cn_y, T_P* cn_z, T_P* c_norm, T_P* s_nx, T_P* s_ny, T_P* s_nz) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1 - 2;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1 - 2;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1 - 2;

    if (i <= nxGlobal_d + 2 && j <= nyGlobal_d + 2 && k <= nzGlobal_d + 2) {
        int node_type_loc = walls_type[i_s4(i, j, k)];
        if (node_type_loc == -1) { // fluid boundary node
            int iteration, iteration_max;
            T_P nwx, nwy, nwz, lambda, local_eps;
            T_P vcx0, vcy0, vcz0, vcx1, vcy1, vcz1, vcx2, vcy2, vcz2, err0, err1, err2, tmp;

            lambda = prc(0.5);
            local_eps = prc(1e-6);
            iteration_max = 4;

            if (c_norm[i_s2(i, j, k)] > local_eps) {

                nwx = s_nx[i_s4(i, j, k)];
                nwy = s_ny[i_s4(i, j, k)];
                nwz = s_nz[i_s4(i, j, k)];
                vcx0 = cn_x[i_s2(i, j, k)];
                vcy0 = cn_y[i_s2(i, j, k)];
                vcz0 = cn_z[i_s2(i, j, k)];
                vcx1 = vcx0 - lambda * (vcx0 + nwx);
                vcy1 = vcy0 - lambda * (vcy0 + nwy);
                vcz1 = vcz0 - lambda * (vcz0 + nwz);

                err0 = (nwx * vcx0 + nwy * vcy0 + nwz * vcz0) - cos_theta_d;

                if ((pprc(abs)(vcx0 + nwx) + pprc(abs)(vcy0 + nwy) + pprc(abs)(vcz0 + nwz) > local_eps || pprc(abs)(vcx0 - nwx) + pprc(abs)(vcy0 - nwy) + pprc(abs)(vcz0 - nwz) > local_eps) && err0 > local_eps) {
                    // do not perform alteration when the normal direction of the solid surface aligned with the fluid interface direction,
                    // or the initial fluid direction is already the desired direction
                    err1 = (nwx * vcx1 + nwy * vcy1 + nwz * vcz1) - prc(sqrt)(vcx1 * vcx1 + vcy1 * vcy1 + vcz1 * vcz1) * cos_theta_d;
                    tmp = prc(1.) / (err1 - err0);
                    vcx2 = tmp * (vcx0 * err1 - vcx1 * err0);
                    vcy2 = tmp * (vcy0 * err1 - vcy1 * err0);
                    vcz2 = tmp * (vcz0 * err1 - vcz1 * err0);

                    err2 = (nwx * vcx2 + nwy * vcy2 + nwz * vcz2) - prc(sqrt)(vcx2 * vcx2 + vcy2 * vcy2 + vcz2 * vcz2) * cos_theta_d;

                    if (err2 > local_eps) {
                        for (iteration = 2; iteration <= iteration_max; iteration++) {
                            vcx0 = vcx1;
                            vcy0 = vcy1;
                            vcz0 = vcz1;
                            vcx1 = vcx2;
                            vcy1 = vcy2;
                            vcz1 = vcz2;
                            err0 = (nwx * vcx0 + nwy * vcy0 + nwz * vcz0) - prc(sqrt)(vcx0 * vcx0 + vcy0 * vcy0 + vcz0 * vcz0) * cos_theta_d;
                            err1 = (nwx * vcx1 + nwy * vcy1 + nwz * vcz1) - prc(sqrt)(vcx1 * vcx1 + vcy1 * vcy1 + vcz1 * vcz1) * cos_theta_d;
                            tmp = prc(1.) / (err1 - err0);
                            if (isinf(tmp)) break;
                            vcx2 = tmp * (vcx0 * err1 - vcx1 * err0);
                            vcy2 = tmp * (vcy0 * err1 - vcy1 * err0);
                            vcz2 = tmp * (vcz0 * err1 - vcz1 * err0);
                            err2 = (nwx * vcx2 + nwy * vcy2 + nwz * vcz2) - prc(sqrt)(vcx2 * vcx2 + vcy2 * vcy2 + vcz2 * vcz2) * cos_theta_d;
                        }
                        //if (iteration >= iteration_max)print*, 'after', iteration_max, ' iterations, theta=', dacos((nwx * vcx2 + nwy * vcy2 + nwz * vcz2) / dsqrt(vcx2 * vcx2 + vcy2 * vcy2 + vcz2 * vcz2)) / pi * 180
                    }
                    tmp = prc(1.) / ((prc(1e-30)) + prc(sqrt)(vcx2 * vcx2 + vcy2 * vcy2 + vcz2 * vcz2));
                    cn_x[i_s2(i, j, k)] = vcx2 * tmp;
                    cn_y[i_s2(i, j, k)] = vcy2 * tmp;
                    cn_z[i_s2(i, j, k)] = vcz2 * tmp;
                }
            }
        }
    }

}
/* ~~~~~~~~~~~~~~ extrapolate normal direction info to solid boundary nodes, to minimize unbalanced forces ~~~~~~~~~~~~~~ */
__global__ void extrapolateNormalToSolid(int* walls_type, T_P* cn_x, T_P* cn_y, T_P* cn_z, T_P* phi) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1 - 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1 - 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1 - 1;

    if (i <= nxGlobal_d + 1 && j <= nyGlobal_d + 1 && k <= nzGlobal_d + 1) {
        int node_type_loc = walls_type[i_s4(i, j, k)];
        if (node_type_loc == 2) { // solid boundary node
            T_P cn_x_sum = prc(0.), cn_y_sum = prc(0.), cn_z_sum = prc(0.), weight_sum = prc(0.);
            for (int q = 1; q < 19; q++) {
                int iex = i + ex_d[q];
                int iey = j + ey_d[q];
                int iez = k + ez_d[q];
                int node_type_neb = walls_type[i_s4(iex, iey, iez)];
                if (node_type_neb <= 0) {
                    cn_x_sum += cn_x[i_s2(iex, iey, iez)] * w_equ_d[q];
                    cn_y_sum += cn_y[i_s2(iex, iey, iez)] * w_equ_d[q];
                    cn_z_sum += cn_z[i_s2(iex, iey, iez)] * w_equ_d[q];
                    weight_sum += w_equ_d[q];
                }
            }
            cn_x[i_s2(i, j, k)] = cn_x_sum / weight_sum;
            cn_y[i_s2(i, j, k)] = cn_y_sum / weight_sum;
            cn_z[i_s2(i, j, k)] = cn_z_sum / weight_sum;
        }
    }
}
/* ~~~~~~~~~~~~~~~~~~ calculate CSF forces based on interace curvature  ~~~~~~~~~~~~~~~~~~ */
__global__ void CSF_Forces(T_P* cn_x, T_P* cn_y, T_P* cn_z, T_P* curv) {
    // Indexing (Thread)
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    T_P kxx, kxy, kxz, kyx, kyy, kyz, kzx, kzy, kzz;
    if (i <= nxGlobal_d && j <= nyGlobal_d && k <= nzGlobal_d) {
        kxx =
            ISO4_d[1 - 1] * (cn_x[i_s2(i + 1, j, k)] - cn_x[i_s2(i - 1, j, k)]) +

            ISO4_d[2 - 1] * (
                cn_x[i_s2(i + 1, j + 1, k)] - cn_x[i_s2(i - 1, j - 1, k)] +
                cn_x[i_s2(i + 1, j - 1, k)] - cn_x[i_s2(i - 1, j + 1, k)] +
                cn_x[i_s2(i + 1, j, k + 1)] - cn_x[i_s2(i - 1, j, k - 1)] +
                cn_x[i_s2(i + 1, j, k - 1)] - cn_x[i_s2(i - 1, j, k + 1)]);

        kyy =
            ISO4_d[1 - 1] * (cn_y[i_s2(i, j + 1, k)] - cn_y[i_s2(i, j - 1, k)]) +

            ISO4_d[2 - 1] * (
                cn_y[i_s2(i + 1, j + 1, k)] - cn_y[i_s2(i - 1, j - 1, k)] +
                cn_y[i_s2(i - 1, j + 1, k)] - cn_y[i_s2(i + 1, j - 1, k)] +
                cn_y[i_s2(i, j + 1, k + 1)] - cn_y[i_s2(i, j - 1, k - 1)] +
                cn_y[i_s2(i, j + 1, k - 1)] - cn_y[i_s2(i, j - 1, k + 1)]);

        kzz =
            ISO4_d[1 - 1] * (cn_z[i_s2(i, j, k + 1)] - cn_z[i_s2(i, j, k - 1)]) +

            ISO4_d[2 - 1] * (
                cn_z[i_s2(i + 1, j, k + 1)] - cn_z[i_s2(i - 1, j, k - 1)] +
                cn_z[i_s2(i - 1, j, k + 1)] - cn_z[i_s2(i + 1, j, k - 1)] +
                cn_z[i_s2(i, j + 1, k + 1)] - cn_z[i_s2(i, j - 1, k - 1)] +
                cn_z[i_s2(i, j - 1, k + 1)] - cn_z[i_s2(i, j + 1, k - 1)]);


        kxy =
            ISO4_d[1 - 1] * (cn_x[i_s2(i, j + 1, k)] - cn_x[i_s2(i, j - 1, k)]) +

            ISO4_d[2 - 1] * (
                cn_x[i_s2(i + 1, j + 1, k)] - cn_x[i_s2(i - 1, j - 1, k)] +
                cn_x[i_s2(i - 1, j + 1, k)] - cn_x[i_s2(i + 1, j - 1, k)] +
                cn_x[i_s2(i, j + 1, k + 1)] - cn_x[i_s2(i, j - 1, k - 1)] +
                cn_x[i_s2(i, j + 1, k - 1)] - cn_x[i_s2(i, j - 1, k + 1)]);

        kxz =
            ISO4_d[1 - 1] * (cn_x[i_s2(i, j, k + 1)] - cn_x[i_s2(i, j, k - 1)]) +

            ISO4_d[2 - 1] * (
                cn_x[i_s2(i + 1, j, k + 1)] - cn_x[i_s2(i - 1, j, k - 1)] +
                cn_x[i_s2(i - 1, j, k + 1)] - cn_x[i_s2(i + 1, j, k - 1)] +
                cn_x[i_s2(i, j + 1, k + 1)] - cn_x[i_s2(i, j - 1, k - 1)] +
                cn_x[i_s2(i, j - 1, k + 1)] - cn_x[i_s2(i, j + 1, k - 1)]);

        kyx =
            ISO4_d[1 - 1] * (cn_y[i_s2(i + 1, j, k)] - cn_y[i_s2(i - 1, j, k)]) +

            ISO4_d[2 - 1] * (
                cn_y[i_s2(i + 1, j + 1, k)] - cn_y[i_s2(i - 1, j - 1, k)] +
                cn_y[i_s2(i + 1, j - 1, k)] - cn_y[i_s2(i - 1, j + 1, k)] +
                cn_y[i_s2(i + 1, j, k + 1)] - cn_y[i_s2(i - 1, j, k - 1)] +
                cn_y[i_s2(i + 1, j, k - 1)] - cn_y[i_s2(i - 1, j, k + 1)]);

        kyz =
            ISO4_d[1 - 1] * (cn_y[i_s2(i, j, k + 1)] - cn_y[i_s2(i, j, k - 1)]) +

            ISO4_d[2 - 1] * (
                cn_y[i_s2(i + 1, j, k + 1)] - cn_y[i_s2(i - 1, j, k - 1)] +
                cn_y[i_s2(i - 1, j, k + 1)] - cn_y[i_s2(i + 1, j, k - 1)] +
                cn_y[i_s2(i, j + 1, k + 1)] - cn_y[i_s2(i, j - 1, k - 1)] +
                cn_y[i_s2(i, j - 1, k + 1)] - cn_y[i_s2(i, j + 1, k - 1)]);

        kzx =
            ISO4_d[1 - 1] * (cn_z[i_s2(i + 1, j, k)] - cn_z[i_s2(i - 1, j, k)]) +

            ISO4_d[2 - 1] * (
                cn_z[i_s2(i + 1, j + 1, k)] - cn_z[i_s2(i - 1, j - 1, k)] +
                cn_z[i_s2(i + 1, j - 1, k)] - cn_z[i_s2(i - 1, j + 1, k)] +
                cn_z[i_s2(i + 1, j, k + 1)] - cn_z[i_s2(i - 1, j, k - 1)] +
                cn_z[i_s2(i + 1, j, k - 1)] - cn_z[i_s2(i - 1, j, k + 1)]);

        kzy =
            ISO4_d[1 - 1] * (cn_z[i_s2(i, j + 1, k)] - cn_z[i_s2(i, j - 1, k)]) +

            ISO4_d[2 - 1] * (
                cn_z[i_s2(i + 1, j + 1, k)] - cn_z[i_s2(i - 1, j - 1, k)] +
                cn_z[i_s2(i - 1, j + 1, k)] - cn_z[i_s2(i + 1, j - 1, k)] +
                cn_z[i_s2(i, j + 1, k + 1)] - cn_z[i_s2(i, j - 1, k - 1)] +
                cn_z[i_s2(i, j + 1, k - 1)] - cn_z[i_s2(i, j - 1, k + 1)]);

        curv[i_s1(i, j, k)] = (prc(pow)(cn_x[i_s2(i, j, k)], 2) - prc(1.)) * kxx + (prc(pow)(cn_y[i_s2(i, j, k)], 2) - prc(1.)) * kyy
            + (prc(pow)(cn_z[i_s2(i, j, k)], 2) - prc(1.)) * kzz +
            cn_x[i_s2(i, j, k)] * cn_y[i_s2(i, j, k)] * (kxy + kyx) + cn_x[i_s2(i, j, k)] * cn_z[i_s2(i, j, k)] * (kxz + kzx)
            + cn_y[i_s2(i, j, k)] * cn_z[i_s2(i, j, k)] * (kzy + kyz);
    }
}
#pragma endregion (color gradient)

#pragma region (Boundary Conditions)

#pragma region (Inlet Boundary Conditions)
__global__ void inlet_bounce_back_velocity_BC_before_odd_GPU(int* walls, T_P* phi, T_P* pdf, T_P* W_in) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

    if (i <= nxGlobal_d && j <= nyGlobal_d) {

        int wall_indicator;
        T_P tmp1, tmp2;

        wall_indicator = walls[i_s2(i, j, 1)];

        phi[i_s4(i, j, 0)] = phi_inlet_d * (1 - wall_indicator) + phi[i_s4(i, j, 0)] * wall_indicator;
        phi[i_s4(i, j, -1)] = phi[i_s4(i, j, 0)];
        phi[i_s4(i, j, -2)] = phi[i_s4(i, j, 0)];
        phi[i_s4(i, j, -3)] = phi[i_s4(i, j, 0)];   //overlap_phi = 4

        //inlet velocity BC    k = 1  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tmp2 = W_in[i_s1(i, j, 0)] * relaxation_d;
        tmp1 = tmp2 * sa_inject_d;              //fluid 1 injection
        tmp2 = tmp2 - tmp1;                   //fluid 2 injection

        pdf[i_f1(i, j, 0, 5, 0)] = (pdf[i_f1(i, j, 1, 6, 0)] + prc(6.0) * w_equ_1_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j, 0, 5, 0)] * wall_indicator;
        pdf[i_f1(i - 1, j, 0, 11, 0)] = (pdf[i_f1(i, j, 1, 14, 0)] + prc(6.0) * w_equ_2_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i - 1, j, 0, 11, 0)] * wall_indicator;
        pdf[i_f1(i + 1, j, 0, 12, 0)] = (pdf[i_f1(i, j, 1, 13, 0)] + prc(6.0) * w_equ_2_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i + 1, j, 0, 12, 0)] * wall_indicator;
        pdf[i_f1(i, j - 1, 0, 15, 0)] = (pdf[i_f1(i, j, 1, 18, 0)] + prc(6.0) * w_equ_2_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j - 1, 0, 15, 0)] * wall_indicator;
        pdf[i_f1(i, j + 1, 0, 16, 0)] = (pdf[i_f1(i, j, 1, 17, 0)] + prc(6.0) * w_equ_2_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j + 1, 0, 16, 0)] * wall_indicator;

        pdf[i_f1(i, j, 0, 5, 1)] = (pdf[i_f1(i, j, 1, 6, 1)] + prc(6.0) * w_equ_1_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j, 0, 5, 1)] * wall_indicator;
        pdf[i_f1(i - 1, j, 0, 11, 1)] = (pdf[i_f1(i, j, 1, 14, 1)] + prc(6.0) * w_equ_2_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i - 1, j, 0, 11, 1)] * wall_indicator;
        pdf[i_f1(i + 1, j, 0, 12, 1)] = (pdf[i_f1(i, j, 1, 13, 1)] + prc(6.0) * w_equ_2_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i + 1, j, 0, 12, 1)] * wall_indicator;
        pdf[i_f1(i, j - 1, 0, 15, 1)] = (pdf[i_f1(i, j, 1, 18, 1)] + prc(6.0) * w_equ_2_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j - 1, 0, 15, 1)] * wall_indicator;
        pdf[i_f1(i, j + 1, 0, 16, 1)] = (pdf[i_f1(i, j, 1, 17, 1)] + prc(6.0) * w_equ_2_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j + 1, 0, 16, 1)] * wall_indicator;
    }
}

__global__ void inlet_bounce_back_velocity_BC_after_odd_GPU(int* walls, T_P* phi, T_P* pdf, T_P* W_in) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        int wall_indicator;
        T_P tmp1, tmp2;

        wall_indicator = walls[i_s2(i, j, 1)];

        phi[i_s4(i, j, 0)] = phi_inlet_d * (1 - wall_indicator) + phi[i_s4(i, j, 0)] * wall_indicator;
        phi[i_s4(i, j, -1)] = phi[i_s4(i, j, 0)];
        phi[i_s4(i, j, -2)] = phi[i_s4(i, j, 0)];
        phi[i_s4(i, j, -3)] = phi[i_s4(i, j, 0)];   //overlap_phi = 4

        // inlet velocity BC    k = 1  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tmp2 = W_in[i_s1(i, j, 0)] * relaxation_d;
        tmp1 = tmp2 * sa_inject_d;              //fluid 1 injection
        tmp2 = tmp2 - tmp1;                   //fluid 2 injection

        pdf[i_f1(i, j, 1, 6, 0)] = (pdf[i_f1(i, j, 0, 5, 0)] + prc(6.0) * w_equ_1_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 6, 0)] * wall_indicator;
        pdf[i_f1(i, j, 1, 13, 0)] = (pdf[i_f1(i + 1, j, 0, 12, 0)] + prc(6.0) * w_equ_2_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 13, 0)] * wall_indicator;
        pdf[i_f1(i, j, 1, 14, 0)] = (pdf[i_f1(i - 1, j, 0, 11, 0)] + prc(6.0) * w_equ_2_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 14, 0)] * wall_indicator;
        pdf[i_f1(i, j, 1, 17, 0)] = (pdf[i_f1(i, j + 1, 0, 16, 0)] + prc(6.0) * w_equ_2_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 17, 0)] * wall_indicator;
        pdf[i_f1(i, j, 1, 18, 0)] = (pdf[i_f1(i, j - 1, 0, 15, 0)] + prc(6.0) * w_equ_2_d * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 18, 0)] * wall_indicator;

        pdf[i_f1(i, j, 1, 6, 1)] = (pdf[i_f1(i, j, 0, 5, 1)] + prc(6.0) * w_equ_1_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 6, 1)] * wall_indicator;
        pdf[i_f1(i, j, 1, 13, 1)] = (pdf[i_f1(i + 1, j, 0, 12, 1)] + prc(6.0) * w_equ_2_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 13, 1)] * wall_indicator;
        pdf[i_f1(i, j, 1, 14, 1)] = (pdf[i_f1(i - 1, j, 0, 11, 1)] + prc(6.0) * w_equ_2_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 14, 1)] * wall_indicator;
        pdf[i_f1(i, j, 1, 17, 1)] = (pdf[i_f1(i, j + 1, 0, 16, 1)] + prc(6.0) * w_equ_2_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 17, 1)] * wall_indicator;
        pdf[i_f1(i, j, 1, 18, 1)] = (pdf[i_f1(i, j - 1, 0, 15, 1)] + prc(6.0) * w_equ_2_d * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 18, 1)] * wall_indicator;
    }
}

//================================================================================================================================================================= =
//----------------------Zou - He type pressure / velocity open inlet boundary conditions----------------------
//currently, there should be only one dominant phase at the inlet boundary nodes,
//otherwise recolor scheme conflicts with zou - he BC for individual fluid component(momentumn for individual fluid is not conserved)
//================================================================================================================================================================= =
//**************************before odd step kernel * ****************************************
__global__ void inlet_Zou_He_pressure_BC_before_odd_GPU(T_P rho_in, int* walls, T_P* phi, T_P* pdf) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        int wall_indicator;

        T_P tmp1, tmp2, tnx, tny, tmpRho1, tmpRho2;

        wall_indicator = walls[i_s2(i, j, 1)];

        phi[i_s4(i, j, 0)] = phi_inlet_d * (1 - wall_indicator) + phi[i_s4(i, j, 0)] * wall_indicator;
        phi[i_s4(i, j, -1)] = phi[i_s4(i, j, 0)];
        phi[i_s4(i, j, -2)] = phi[i_s4(i, j, 0)];
        phi[i_s4(i, j, -3)] = phi[i_s4(i, j, 0)];   //overlap_phi = 4

        // inlet pressure BC    k = 1  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Zou - He pressure BC applied to the bulk PDF
        // inlet velocity BC    k = 1  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tmpRho2 = rho_in;
        tmpRho1 = rho_in * sa_inject_d;                  //fluid 1 injection
        tmpRho2 = tmpRho2 - tmpRho1;                   // fluid 2 injection

        tmp1 = (tmpRho1 -
            (pdf[i_f1(i, j, 1, 0, 0)] +
                pdf[i_f1(i - 1, j, 1, 1, 0)] +
                pdf[i_f1(i + 1, j, 1, 2, 0)] +
                pdf[i_f1(i, j - 1, 1, 3, 0)] +
                pdf[i_f1(i, j + 1, 1, 4, 0)] +
                pdf[i_f1(i - 1, j - 1, 1, 7, 0)] +
                pdf[i_f1(i + 1, j - 1, 1, 8, 0)] +
                pdf[i_f1(i - 1, j + 1, 1, 9, 0)] +
                pdf[i_f1(i + 1, j + 1, 1, 10, 0)] + prc(2.) * (
                    pdf[i_f1(i, j, 2, 6, 0)] +
                    pdf[i_f1(i + 1, j, 2, 14, 0)] +
                    pdf[i_f1(i - 1, j, 2, 13, 0)] +
                    pdf[i_f1(i, j + 1, 2, 18, 0)] +
                    pdf[i_f1(i, j - 1, 2, 17, 0)]))) * relaxation_d;

        tnx = prc(0.5) * (
            pdf[i_f1(i - 1, j, 1, 1, 0)] + pdf[i_f1(i - 1, j - 1, 1, 7, 0)] + pdf[i_f1(i - 1, j + 1, 1, 9, 0)] - (
                pdf[i_f1(i + 1, j, 1, 2, 0)] + pdf[i_f1(i + 1, j - 1, 1, 8, 0)] + pdf[i_f1(i + 1, j + 1, 1, 10, 0)]));
        tny = prc(0.5) * (
            pdf[i_f1(i, j - 1, 1, 3, 0)] + pdf[i_f1(i - 1, j - 1, 1, 7, 0)] + pdf[i_f1(i + 1, j - 1, 1, 8, 0)] - (
                pdf[i_f1(i, j + 1, 1, 4, 0)] + pdf[i_f1(i + 1, j + 1, 1, 10, 0)] + pdf[i_f1(i - 1, j + 1, 1, 9, 0)]));

        pdf[i_f1(i, j, 0, 5, 0)] = (pdf[i_f1(i, j, 2, 6, 0)] + prc(0.333333333333333333) * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j, 0, 5, 0)] * wall_indicator;
        pdf[i_f1(i - 1, j, 0, 11, 0)] = (pdf[i_f1(i + 1, j, 2, 14, 0)] + prc(0.166666666666666667) * tmp1 - tnx) * (1 - wall_indicator) + pdf[i_f1(i - 1, j, 0, 11, 0)] * wall_indicator;
        pdf[i_f1(i + 1, j, 0, 12, 0)] = (pdf[i_f1(i - 1, j, 2, 13, 0)] + prc(0.166666666666666667) * tmp1 + tnx) * (1 - wall_indicator) + pdf[i_f1(i + 1, j, 0, 12, 0)] * wall_indicator;
        pdf[i_f1(i, j - 1, 0, 15, 0)] = (pdf[i_f1(i, j + 1, 2, 18, 0)] + prc(0.166666666666666667) * tmp1 - tny) * (1 - wall_indicator) + pdf[i_f1(i, j - 1, 0, 15, 0)] * wall_indicator;
        pdf[i_f1(i, j + 1, 0, 16, 0)] = (pdf[i_f1(i, j - 1, 2, 17, 0)] + prc(0.166666666666666667) * tmp1 + tny) * (1 - wall_indicator) + pdf[i_f1(i, j + 1, 0, 16, 0)] * wall_indicator;

        tmp2 = (tmpRho2 -
            (pdf[i_f1(i, j, 1, 0, 1)] +
                pdf[i_f1(i - 1, j, 1, 1, 1)] +
                pdf[i_f1(i + 1, j, 1, 2, 1)] +
                pdf[i_f1(i, j - 1, 1, 3, 1)] +
                pdf[i_f1(i, j + 1, 1, 4, 1)] +
                pdf[i_f1(i - 1, j - 1, 1, 7, 1)] +
                pdf[i_f1(i + 1, j - 1, 1, 8, 1)] +
                pdf[i_f1(i - 1, j + 1, 1, 9, 1)] +
                pdf[i_f1(i + 1, j + 1, 1, 10, 1)] + prc(2.) * (
                    pdf[i_f1(i, j, 2, 6, 1)] +
                    pdf[i_f1(i + 1, j, 2, 14, 1)] +
                    pdf[i_f1(i - 1, j, 2, 13, 1)] +
                    pdf[i_f1(i, j + 1, 2, 18, 1)] +
                    pdf[i_f1(i, j - 1, 2, 17, 1)]))) * relaxation_d;

        tnx = prc(0.5) * (
            pdf[i_f1(i - 1, j, 1, 1, 1)] + pdf[i_f1(i - 1, j - 1, 1, 7, 1)] + pdf[i_f1(i - 1, j + 1, 1, 9, 1)] - (
                pdf[i_f1(i + 1, j, 1, 2, 1)] + pdf[i_f1(i + 1, j - 1, 1, 8, 1)] + pdf[i_f1(i + 1, j + 1, 1, 10, 1)]));
        tny = prc(0.5) * (
            pdf[i_f1(i, j - 1, 1, 3, 1)] + pdf[i_f1(i - 1, j - 1, 1, 7, 1)] + pdf[i_f1(i + 1, j - 1, 1, 8, 1)] - (
                pdf[i_f1(i, j + 1, 1, 4, 1)] + pdf[i_f1(i + 1, j + 1, 1, 10, 1)] + pdf[i_f1(i - 1, j + 1, 1, 9, 1)]));

        pdf[i_f1(i, j, 0, 5, 1)] = (pdf[i_f1(i, j, 2, 6, 1)] + prc(0.333333333333333333) * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j, 0, 5, 1)] * wall_indicator;
        pdf[i_f1(i - 1, j, 0, 11, 1)] = (pdf[i_f1(i + 1, j, 2, 14, 1)] + prc(0.166666666666666667) * tmp2 - tnx) * (1 - wall_indicator) + pdf[i_f1(i - 1, j, 0, 11, 1)] * wall_indicator;
        pdf[i_f1(i + 1, j, 0, 12, 1)] = (pdf[i_f1(i - 1, j, 2, 13, 1)] + prc(0.166666666666666667) * tmp2 + tnx) * (1 - wall_indicator) + pdf[i_f1(i + 1, j, 0, 12, 1)] * wall_indicator;
        pdf[i_f1(i, j - 1, 0, 15, 1)] = (pdf[i_f1(i, j + 1, 2, 18, 1)] + prc(0.166666666666666667) * tmp2 - tny) * (1 - wall_indicator) + pdf[i_f1(i, j - 1, 0, 15, 1)] * wall_indicator;
        pdf[i_f1(i, j + 1, 0, 16, 1)] = (pdf[i_f1(i, j - 1, 2, 17, 1)] + prc(0.166666666666666667) * tmp2 + tny) * (1 - wall_indicator) + pdf[i_f1(i, j + 1, 0, 16, 1)] * wall_indicator;
    }
}

//************************** after odd step kernel *****************************************
__global__ void inlet_Zou_He_pressure_BC_after_odd_GPU(T_P rho_in, int* walls, T_P* phi, T_P* pdf) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        int wall_indicator;

        T_P tmp1, tmp2, tnx, tny, tmpRho1, tmpRho2;

        wall_indicator = walls[i_s2(i, j, 1)];

        phi[i_s4(i, j, 0)] = phi_inlet_d * (1 - wall_indicator) + phi[i_s4(i, j, 0)] * wall_indicator;
        phi[i_s4(i, j, -1)] = phi[i_s4(i, j, 0)];
        phi[i_s4(i, j, -2)] = phi[i_s4(i, j, 0)];
        phi[i_s4(i, j, -3)] = phi[i_s4(i, j, 0)];   //overlap_phi = 4

        // inlet pressure BC    k = 1  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tmpRho2 = rho_in;
        tmpRho1 = rho_in * sa_inject_d;              // fluid 1 injection
        tmpRho2 = tmpRho2 - tmpRho1;                   // fluid 2 injection

        tmp1 = (tmpRho1 -
            (pdf[i_f1(i, j, 1, 0, 0)] +
                pdf[i_f1(i, j, 1, 2, 0)] +
                pdf[i_f1(i, j, 1, 1, 0)] +
                pdf[i_f1(i, j, 1, 4, 0)] +
                pdf[i_f1(i, j, 1, 3, 0)] +
                pdf[i_f1(i, j, 1, 8, 0)] +
                pdf[i_f1(i, j, 1, 7, 0)] +
                pdf[i_f1(i, j, 1, 10, 0)] +
                pdf[i_f1(i, j, 1, 9, 0)] + prc(2.) * (
                    pdf[i_f1(i, j, 1, 5, 0)] +
                    pdf[i_f1(i, j, 1, 11, 0)] +
                    pdf[i_f1(i, j, 1, 12, 0)] +
                    pdf[i_f1(i, j, 1, 15, 0)] +
                    pdf[i_f1(i, j, 1, 16, 0)]))) * relaxation_d;


        tnx = prc(0.5) * (pdf[i_f1(i, j, 1, 2, 0)] + pdf[i_f1(i, j, 1, 8, 0)] + pdf[i_f1(i, j, 1, 10, 0)] - (pdf[i_f1(i, j, 1, 1, 0)] + pdf[i_f1(i, j, 1, 7, 0)] + pdf[i_f1(i, j, 1, 9, 0)]));
        tny = prc(0.5) * (pdf[i_f1(i, j, 1, 4, 0)] + pdf[i_f1(i, j, 1, 9, 0)] + pdf[i_f1(i, j, 1, 10, 0)] - (pdf[i_f1(i, j, 1, 3, 0)] + pdf[i_f1(i, j, 1, 8, 0)] + pdf[i_f1(i, j, 1, 7, 0)]));

        pdf[i_f1(i, j, 1, 6, 0)] = (pdf[i_f1(i, j, 1, 5, 0)] + prc(0.333333333333333333) * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 6, 0)] * wall_indicator;
        pdf[i_f1(i, j, 1, 13, 0)] = (pdf[i_f1(i, j, 1, 12, 0)] + prc(0.166666666666666667) * tmp1 + tnx) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 13, 0)] * wall_indicator;
        pdf[i_f1(i, j, 1, 14, 0)] = (pdf[i_f1(i, j, 1, 11, 0)] + prc(0.166666666666666667) * tmp1 - tnx) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 14, 0)] * wall_indicator;
        pdf[i_f1(i, j, 1, 17, 0)] = (pdf[i_f1(i, j, 1, 16, 0)] + prc(0.166666666666666667) * tmp1 + tny) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 17, 0)] * wall_indicator;
        pdf[i_f1(i, j, 1, 18, 0)] = (pdf[i_f1(i, j, 1, 15, 0)] + prc(0.166666666666666667) * tmp1 - tny) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 18, 0)] * wall_indicator;

        tmp2 = (tmpRho2 - (
            pdf[i_f1(i, j, 1, 0, 1)] +
            pdf[i_f1(i, j, 1, 2, 1)] +
            pdf[i_f1(i, j, 1, 1, 1)] +
            pdf[i_f1(i, j, 1, 4, 1)] +
            pdf[i_f1(i, j, 1, 3, 1)] +
            pdf[i_f1(i, j, 1, 8, 1)] +
            pdf[i_f1(i, j, 1, 7, 1)] +
            pdf[i_f1(i, j, 1, 10, 1)] +
            pdf[i_f1(i, j, 1, 9, 1)] + prc(2.) * (
                pdf[i_f1(i, j, 1, 5, 1)] +
                pdf[i_f1(i, j, 1, 11, 1)] +
                pdf[i_f1(i, j, 1, 12, 1)] +
                pdf[i_f1(i, j, 1, 15, 1)] +
                pdf[i_f1(i, j, 1, 16, 1)]))) * relaxation_d;

        tnx = prc(0.5) * (pdf[i_f1(i, j, 1, 2, 1)] + pdf[i_f1(i, j, 1, 8, 1)] + pdf[i_f1(i, j, 1, 10, 1)] - (pdf[i_f1(i, j, 1, 1, 1)] + pdf[i_f1(i, j, 1, 7, 1)] + pdf[i_f1(i, j, 1, 9, 1)]));
        tny = prc(0.5) * (pdf[i_f1(i, j, 1, 4, 1)] + pdf[i_f1(i, j, 1, 9, 1)] + pdf[i_f1(i, j, 1, 10, 1)] - (pdf[i_f1(i, j, 1, 3, 1)] + pdf[i_f1(i, j, 1, 8, 1)] + pdf[i_f1(i, j, 1, 7, 1)]));

        pdf[i_f1(i, j, 1, 6, 1)] = (pdf[i_f1(i, j, 1, 5, 1)] + prc(0.333333333333333333) * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 6, 1)] * wall_indicator;
        pdf[i_f1(i, j, 1, 13, 1)] = (pdf[i_f1(i, j, 1, 12, 1)] + prc(0.166666666666666667) * tmp2 + tnx) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 13, 1)] * wall_indicator;
        pdf[i_f1(i, j, 1, 14, 1)] = (pdf[i_f1(i, j, 1, 11, 1)] + prc(0.166666666666666667) * tmp2 - tnx) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 14, 1)] * wall_indicator;
        pdf[i_f1(i, j, 1, 17, 1)] = (pdf[i_f1(i, j, 1, 16, 1)] + prc(0.166666666666666667) * tmp2 + tny) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 17, 1)] * wall_indicator;
        pdf[i_f1(i, j, 1, 18, 1)] = (pdf[i_f1(i, j, 1, 15, 1)] + prc(0.166666666666666667) * tmp2 - tny) * (1 - wall_indicator) + pdf[i_f1(i, j, 1, 18, 1)] * wall_indicator;
    }
}

#pragma endregion (Inlet Boundary Conditions)

#pragma region (Outlet Boundary Conditions)
__global__ void outlet_convective_BC_before_odd_GPU(int* walls, T_P* phi, T_P* phi_convec_bc, T_P* pdf, T_P* f_convec_bc, T_P* g_convec_bc) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        T_P temp, u_convec;
        int wall_indicator;

        u_convec = uin_avg_d;
        temp = prc(1.) / (prc(1.) + u_convec);

        wall_indicator = walls[i_s2(i, j, nzGlobal_d)];

        phi[i_s4(i, j, nzGlobal_d + 1)] = ((phi_convec_bc[i_s1(i, j, 0)] + u_convec * phi[i_s4(i, j, nzGlobal_d)]) * temp) * (1 - wall_indicator)
            + phi[i_s4(i, j, nzGlobal_d + 1)] * wall_indicator;
        phi_convec_bc[i_s1(i, j, 0)] = phi[i_s4(i, j, nzGlobal_d + 1)];   //store PDF for next step
        phi[i_s4(i, j, nzGlobal_d + 2)] = phi[i_s4(i, j, nzGlobal_d + 1)];
        phi[i_s4(i, j, nzGlobal_d + 3)] = phi[i_s4(i, j, nzGlobal_d + 1)];
        phi[i_s4(i, j, nzGlobal_d + 4)] = phi[i_s4(i, j, nzGlobal_d + 1)];   //overlap_phi = 4

        //if outlet convective BC
        pdf[i_f1(i, j, nzGlobal_d + 1, 6, 0)] = ((f_convec_bc[icnv_f1(i, j, 6)] + u_convec * pdf[i_f1(i, j, nzGlobal_d + 1 - 1, 6, 0)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d + 1, 6, 0)] * wall_indicator;
        pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 0)] = ((f_convec_bc[icnv_f1(i, j, 13)] + u_convec * pdf[i_f1(i - 1, j, nzGlobal_d + 1 - 1, 13, 0)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 0)] * wall_indicator;
        pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 0)] = ((f_convec_bc[icnv_f1(i, j, 14)] + u_convec * pdf[i_f1(i + 1, j, nzGlobal_d + 1 - 1, 14, 0)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 0)] * wall_indicator;
        pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 0)] = ((f_convec_bc[icnv_f1(i, j, 17)] + u_convec * pdf[i_f1(i, j - 1, nzGlobal_d + 1 - 1, 17, 0)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 0)] * wall_indicator;
        pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 0)] = ((f_convec_bc[icnv_f1(i, j, 18)] + u_convec * pdf[i_f1(i, j + 1, nzGlobal_d + 1 - 1, 18, 0)]) * temp)
            * (1 - wall_indicator) + pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 0)] * wall_indicator;

        pdf[i_f1(i, j, nzGlobal_d + 1, 6, 1)] = ((f_convec_bc[icnv_f1(i, j, 6)] + u_convec * pdf[i_f1(i, j, nzGlobal_d + 1 - 1, 6, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d + 1, 6, 1)] * wall_indicator;
        pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 1)] = ((f_convec_bc[icnv_f1(i, j, 13)] + u_convec * pdf[i_f1(i - 1, j, nzGlobal_d + 1 - 1, 13, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 1)] * wall_indicator;
        pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 1)] = ((f_convec_bc[icnv_f1(i, j, 14)] + u_convec * pdf[i_f1(i + 1, j, nzGlobal_d + 1 - 1, 14, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 1)] * wall_indicator;
        pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 1)] = ((f_convec_bc[icnv_f1(i, j, 17)] + u_convec * pdf[i_f1(i, j - 1, nzGlobal_d + 1 - 1, 17, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 1)] * wall_indicator;
        pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 1)] = ((f_convec_bc[icnv_f1(i, j, 18)] + u_convec * pdf[i_f1(i, j + 1, nzGlobal_d + 1 - 1, 18, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 1)] * wall_indicator;

        f_convec_bc[icnv_f1(i, j, 6)] = pdf[i_f1(i, j, nzGlobal_d + 1, 6, 0)];
        f_convec_bc[icnv_f1(i, j, 13)] = pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 0)];
        f_convec_bc[icnv_f1(i, j, 14)] = pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 0)];
        f_convec_bc[icnv_f1(i, j, 17)] = pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 0)];
        f_convec_bc[icnv_f1(i, j, 18)] = pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 0)];

        g_convec_bc[icnv_f1(i, j, 6)] = pdf[i_f1(i, j, nzGlobal_d + 1, 6, 1)];
        g_convec_bc[icnv_f1(i, j, 13)] = pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 1)];
        g_convec_bc[icnv_f1(i, j, 14)] = pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 1)];
        g_convec_bc[icnv_f1(i, j, 17)] = pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 1)];
        g_convec_bc[icnv_f1(i, j, 18)] = pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 1)];
    }
}

__global__ void outlet_convective_BC_after_odd_GPU(int* walls, T_P* phi, T_P* phi_convec_bc, T_P* pdf, T_P* f_convec_bc, T_P* g_convec_bc) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        T_P temp, u_convec;
        int wall_indicator;

        u_convec = uin_avg_d;
        temp = prc(1.) / (prc(1.) + u_convec);

        wall_indicator = walls[i_s2(i, j, nzGlobal_d)];

        phi[i_s4(i, j, nzGlobal_d + 1)] = ((phi_convec_bc[i_s1(i, j, 0)] + u_convec * phi[i_s4(i, j, nzGlobal_d)]) * temp) * (1 - wall_indicator)
            + phi[i_s4(i, j, nzGlobal_d + 1)] * wall_indicator;
        phi_convec_bc[i_s1(i, j, 0)] = phi[i_s4(i, j, nzGlobal_d + 1)];   //store PDF for next step
        phi[i_s4(i, j, nzGlobal_d + 2)] = phi[i_s4(i, j, nzGlobal_d + 1)];
        phi[i_s4(i, j, nzGlobal_d + 3)] = phi[i_s4(i, j, nzGlobal_d + 1)];
        phi[i_s4(i, j, nzGlobal_d + 4)] = phi[i_s4(i, j, nzGlobal_d + 1)];   //overlap_phi = 4
        // outlet convective bc

        pdf[i_f1(i, j, nzGlobal_d, 5, 0)] = ((f_convec_bc[icnv_f1(i, j, 6)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 5, 0)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 5, 0)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 11, 0)] = ((f_convec_bc[icnv_f1(i, j, 14)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 11, 0)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 11, 0)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 12, 0)] = ((f_convec_bc[icnv_f1(i, j, 13)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 12, 0)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 12, 0)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 15, 0)] = ((f_convec_bc[icnv_f1(i, j, 18)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 15, 0)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 15, 0)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 16, 0)] = ((f_convec_bc[icnv_f1(i, j, 17)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 16, 0)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 16, 0)] * wall_indicator;

        pdf[i_f1(i, j, nzGlobal_d, 5, 1)] = ((f_convec_bc[icnv_f1(i, j, 6)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 5, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 5, 1)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 11, 1)] = ((f_convec_bc[icnv_f1(i, j, 14)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 11, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 11, 1)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 12, 1)] = ((f_convec_bc[icnv_f1(i, j, 13)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 12, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 12, 1)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 15, 1)] = ((f_convec_bc[icnv_f1(i, j, 18)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 15, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 15, 1)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 16, 1)] = ((f_convec_bc[icnv_f1(i, j, 17)] + u_convec * pdf[i_f1(i, j, nzGlobal_d - 1, 16, 1)]) * temp) * (1 - wall_indicator)
            + pdf[i_f1(i, j, nzGlobal_d, 16, 1)] * wall_indicator;

        f_convec_bc[icnv_f1(i, j, 6)] = pdf[i_f1(i, j, nzGlobal_d, 5, 0)];
        f_convec_bc[icnv_f1(i, j, 14)] = pdf[i_f1(i, j, nzGlobal_d, 11, 0)];
        f_convec_bc[icnv_f1(i, j, 13)] = pdf[i_f1(i, j, nzGlobal_d, 12, 0)];
        f_convec_bc[icnv_f1(i, j, 18)] = pdf[i_f1(i, j, nzGlobal_d, 15, 0)];
        f_convec_bc[icnv_f1(i, j, 17)] = pdf[i_f1(i, j, nzGlobal_d, 16, 0)];

        g_convec_bc[icnv_f1(i, j, 6)] = pdf[i_f1(i, j, nzGlobal_d, 5, 1)];
        g_convec_bc[icnv_f1(i, j, 14)] = pdf[i_f1(i, j, nzGlobal_d, 11, 1)];
        g_convec_bc[icnv_f1(i, j, 13)] = pdf[i_f1(i, j, nzGlobal_d, 12, 1)];
        g_convec_bc[icnv_f1(i, j, 18)] = pdf[i_f1(i, j, nzGlobal_d, 15, 1)];
        g_convec_bc[icnv_f1(i, j, 17)] = pdf[i_f1(i, j, nzGlobal_d, 16, 1)];
    }
}

//=============================================================================================
//----------------------Zou - He type pressure open outlet boundary conditions----------------
//=============================================================================================
//**************************before odd step kernel * ****************************************
__global__ void outlet_Zou_He_pressure_BC_before_odd_GPU(T_P rho_out, int* walls,T_P* phi, T_P* pdf)
    {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        int wall_indicator;

        T_P tmp1, tmp2, tnx, tny;

        wall_indicator = walls[i_s2(i, j, nzGlobal_d)];
        phi[i_s4(i, j, nzGlobal_d + 1)] = phi[i_s4(i, j, nzGlobal_d)];
        phi[i_s4(i, j, nzGlobal_d + 2)] = phi[i_s4(i, j, nzGlobal_d)];
        phi[i_s4(i, j, nzGlobal_d + 3)] = phi[i_s4(i, j, nzGlobal_d)];
        phi[i_s4(i, j, nzGlobal_d + 4)] = phi[i_s4(i, j, nzGlobal_d)];   // overlap_phi = 4

        // outlet pressure BC    k=1a
        tmp1 = (
            pdf[i_f1(i, j, nzGlobal_d, 0, 0)] +
            pdf[i_f1(i - 1, j, nzGlobal_d, 1, 0)] +
            pdf[i_f1(i + 1, j, nzGlobal_d, 2, 0)] +
            pdf[i_f1(i, j - 1, nzGlobal_d, 3, 0)] +
            pdf[i_f1(i, j + 1, nzGlobal_d, 4, 0)] +
            pdf[i_f1(i - 1, j - 1, nzGlobal_d, 7, 0)] +
            pdf[i_f1(i + 1, j - 1, nzGlobal_d, 8, 0)] +
            pdf[i_f1(i - 1, j + 1, nzGlobal_d, 9, 0)] +
            pdf[i_f1(i + 1, j + 1, nzGlobal_d, 10, 0)] + prc(2.) * (
                pdf[i_f1(i, j, nzGlobal_d - 1, 5, 0)] +
                pdf[i_f1(i - 1, j, nzGlobal_d - 1, 11, 0)] +
                pdf[i_f1(i + 1, j, nzGlobal_d - 1, 12, 0)] +
                pdf[i_f1(i, j - 1, nzGlobal_d - 1, 15, 0)] +
                pdf[i_f1(i, j + 1, nzGlobal_d - 1, 16, 0)]) +
            pdf[i_f1(i, j, nzGlobal_d, 0, 1)] +
            pdf[i_f1(i - 1, j, nzGlobal_d, 1, 1)] +
            pdf[i_f1(i + 1, j, nzGlobal_d, 2, 1)] +
            pdf[i_f1(i, j - 1, nzGlobal_d, 3, 1)] +
            pdf[i_f1(i, j + 1, nzGlobal_d, 4, 1)] +
            pdf[i_f1(i - 1, j - 1, nzGlobal_d, 7, 1)] +
            pdf[i_f1(i + 1, j - 1, nzGlobal_d, 8, 1)] +
            pdf[i_f1(i - 1, j + 1, nzGlobal_d, 9, 1)] +
            pdf[i_f1(i + 1, j + 1, nzGlobal_d, 10, 1)] + prc(2.) * (
                pdf[i_f1(i, j, nzGlobal_d - 1, 5, 1)] +
                pdf[i_f1(i - 1, j, nzGlobal_d - 1, 11, 1)] +
                pdf[i_f1(i + 1, j, nzGlobal_d - 1, 12, 1)] +
                pdf[i_f1(i, j - 1, nzGlobal_d - 1, 15, 1)] +
                pdf[i_f1(i, j + 1, nzGlobal_d - 1, 16, 1)])) - rho_out;


        tmp2 = tmp1 * prc(0.5) * (prc(1.) - phi[i_s4(i, j, nzGlobal_d)]);    // fluid 2 net flux
        tmp1 = tmp1 - tmp2;                   // fluid 1 net flux

        tnx = prc(0.5) * (
            pdf[i_f1(i - 1, j, nzGlobal_d, 1, 0)] + pdf[i_f1(i - 1, j - 1, nzGlobal_d, 7, 0)] + pdf[i_f1(i - 1, j + 1, nzGlobal_d, 9, 0)] - (
                pdf[i_f1(i + 1, j, nzGlobal_d, 2, 0)] + pdf[i_f1(i + 1, j - 1, nzGlobal_d, 8, 0)] + pdf[i_f1(i + 1, j + 1, nzGlobal_d, 10, 0)]));
        tny = prc(0.5) * (
            pdf[i_f1(i, j - 1, nzGlobal_d, 3, 0)] + pdf[i_f1(i - 1, j - 1, nzGlobal_d, 7, 0)] + pdf[i_f1(i + 1, j - 1, nzGlobal_d, 8, 0)] - (
                pdf[i_f1(i, j + 1, nzGlobal_d, 4, 0)] + pdf[i_f1(i + 1, j + 1, nzGlobal_d, 10, 0)] + pdf[i_f1(i - 1, j + 1, nzGlobal_d, 9, 0)]));

        pdf[i_f1(i, j, nzGlobal_d + 1, 6, 0)] = (pdf[i_f1(i, j, nzGlobal_d - 1, 5, 0)] - prc(0.333333333333333333) * tmp1) * (1 - wall_indicator) +
            pdf[i_f1(i, j, nzGlobal_d + 1, 6, 0)] * wall_indicator;
        pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 0)] = (pdf[i_f1(i + 1, j, nzGlobal_d - 1, 12, 0)] - prc(0.166666666666666667) * tmp1 - tnx) * (1 - wall_indicator)
            + pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 0)] * wall_indicator;
        pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 0)] = (pdf[i_f1(i - 1, j, nzGlobal_d - 1, 11, 0)] - prc(0.166666666666666667) * tmp1 + tnx) * (1 - wall_indicator)
            + pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 0)] * wall_indicator;
        pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 0)] = (pdf[i_f1(i, j + 1, nzGlobal_d - 1, 16, 0)] - prc(0.166666666666666667) * tmp1 - tny) * (1 - wall_indicator)
            + pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 0)] * wall_indicator;
        pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 0)] = (pdf[i_f1(i, j - 1, nzGlobal_d - 1, 15, 0)] - prc(0.166666666666666667) * tmp1 + tny) * (1 - wall_indicator)
            + pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 0)] * wall_indicator;

        tnx = prc(0.5) * (
            pdf[i_f1(i - 1, j, nzGlobal_d, 1, 1)] + pdf[i_f1(i - 1, j - 1, nzGlobal_d, 7, 1)] + pdf[i_f1(i - 1, j + 1, nzGlobal_d, 9, 1)] - (
                pdf[i_f1(i + 1, j, nzGlobal_d, 2, 1)] + pdf[i_f1(i + 1, j - 1, nzGlobal_d, 8, 1)] + pdf[i_f1(i + 1, j + 1, nzGlobal_d, 10, 1)]));
        tny = prc(0.5) * (
            pdf[i_f1(i, j - 1, nzGlobal_d, 3, 1)] + pdf[i_f1(i - 1, j - 1, nzGlobal_d, 7, 1)] + pdf[i_f1(i + 1, j - 1, nzGlobal_d, 8, 1)] - (
                pdf[i_f1(i, j + 1, nzGlobal_d, 4, 1)] + pdf[i_f1(i + 1, j + 1, nzGlobal_d, 10, 1)] + pdf[i_f1(i - 1, j + 1, nzGlobal_d, 9, 1)]));

        pdf[i_f1(i, j, nzGlobal_d + 1, 6, 1)] = (pdf[i_f1(i, j, nzGlobal_d - 1, 5, 1)] - prc(0.333333333333333333) * tmp2) * (1 - wall_indicator) +
            pdf[i_f1(i, j, nzGlobal_d + 1, 6, 1)] * wall_indicator;
        pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 1)] = (pdf[i_f1(i + 1, j, nzGlobal_d - 1, 12, 1)] - prc(0.166666666666666667) * tmp2 - tnx) * (1 - wall_indicator)
            + pdf[i_f1(i - 1, j, nzGlobal_d + 1, 13, 1)] * wall_indicator;
        pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 1)] = (pdf[i_f1(i - 1, j, nzGlobal_d - 1, 11, 1)] - prc(0.166666666666666667) * tmp2 + tnx) * (1 - wall_indicator)
            + pdf[i_f1(i + 1, j, nzGlobal_d + 1, 14, 1)] * wall_indicator;
        pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 1)] = (pdf[i_f1(i, j + 1, nzGlobal_d - 1, 16, 1)] - prc(0.166666666666666667) * tmp2 - tny) * (1 - wall_indicator)
            + pdf[i_f1(i, j - 1, nzGlobal_d + 1, 17, 1)] * wall_indicator;
        pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 1)] = (pdf[i_f1(i, j - 1, nzGlobal_d - 1, 15, 1)] - prc(0.166666666666666667) * tmp2 + tny) * (1 - wall_indicator)
            + pdf[i_f1(i, j + 1, nzGlobal_d + 1, 18, 1)] * wall_indicator;
    }
}

// ************************** after odd step kernel *****************************************
__global__ void outlet_Zou_He_pressure_BC_after_odd_GPU(T_P rho_out, int* walls, T_P* phi, T_P* pdf) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        int wall_indicator;

        T_P tmp1, tmp2, tnx, tny;

        wall_indicator = walls[i_s2(i, j, nzGlobal_d)];
        phi[i_s4(i, j, nzGlobal_d + 1)] = phi[i_s4(i, j, nzGlobal_d)];
        phi[i_s4(i, j, nzGlobal_d + 2)] = phi[i_s4(i, j, nzGlobal_d)];
        phi[i_s4(i, j, nzGlobal_d + 3)] = phi[i_s4(i, j, nzGlobal_d)];
        phi[i_s4(i, j, nzGlobal_d + 4)] = phi[i_s4(i, j, nzGlobal_d)];   // overlap_phi = 4

        // inlet pressure BC    k=1
        tmp1 = (
            pdf[i_f1(i, j, nzGlobal_d, 0, 0)] +
            pdf[i_f1(i, j, nzGlobal_d, 2, 0)] +
            pdf[i_f1(i, j, nzGlobal_d, 1, 0)] +
            pdf[i_f1(i, j, nzGlobal_d, 4, 0)] +
            pdf[i_f1(i, j, nzGlobal_d, 3, 0)] +
            pdf[i_f1(i, j, nzGlobal_d, 8, 0)] +
            pdf[i_f1(i, j, nzGlobal_d, 7, 0)] +
            pdf[i_f1(i, j, nzGlobal_d, 10, 0)] +
            pdf[i_f1(i, j, nzGlobal_d, 9, 0)] + prc(2.) * (
                pdf[i_f1(i, j, nzGlobal_d, 6, 0)] +
                pdf[i_f1(i, j, nzGlobal_d, 14, 0)] +
                pdf[i_f1(i, j, nzGlobal_d, 13, 0)] +
                pdf[i_f1(i, j, nzGlobal_d, 18, 0)] +
                pdf[i_f1(i, j, nzGlobal_d, 17, 0)]) +
            pdf[i_f1(i, j, nzGlobal_d, 0, 1)] +
            pdf[i_f1(i, j, nzGlobal_d, 2, 1)] +
            pdf[i_f1(i, j, nzGlobal_d, 1, 1)] +
            pdf[i_f1(i, j, nzGlobal_d, 4, 1)] +
            pdf[i_f1(i, j, nzGlobal_d, 3, 1)] +
            pdf[i_f1(i, j, nzGlobal_d, 8, 1)] +
            pdf[i_f1(i, j, nzGlobal_d, 7, 1)] +
            pdf[i_f1(i, j, nzGlobal_d, 10, 1)] +
            pdf[i_f1(i, j, nzGlobal_d, 9, 1)] + prc(2.) * (
                pdf[i_f1(i, j, nzGlobal_d, 6, 1)] +
                pdf[i_f1(i, j, nzGlobal_d, 14, 1)] +
                pdf[i_f1(i, j, nzGlobal_d, 13, 1)] +
                pdf[i_f1(i, j, nzGlobal_d, 18, 1)] +
                pdf[i_f1(i, j, nzGlobal_d, 17, 1)])) - rho_out;

        tmp2 = tmp1 * prc(0.5) * (prc(1.) - phi[i_s4(i, j, nzGlobal_d)]);    //fluid 2 net flux
        tmp1 = tmp1 - tmp2;                   // fluid 1 net flux

        tnx = prc(0.5) * (pdf[i_f1(i, j, nzGlobal_d, 2, 0)] + pdf[i_f1(i, j, nzGlobal_d, 8, 0)] + pdf[i_f1(i, j, nzGlobal_d, 10, 0)]
            - (pdf[i_f1(i, j, nzGlobal_d, 1, 0)] + pdf[i_f1(i, j, nzGlobal_d, 7, 0)] + pdf[i_f1(i, j, nzGlobal_d, 9, 0)]));
        tny = prc(0.5) * (pdf[i_f1(i, j, nzGlobal_d, 4, 0)] + pdf[i_f1(i, j, nzGlobal_d, 10, 0)] + pdf[i_f1(i, j, nzGlobal_d, 9, 0)]
            - (pdf[i_f1(i, j, nzGlobal_d, 3, 0)] + pdf[i_f1(i, j, nzGlobal_d, 7, 0)] + pdf[i_f1(i, j, nzGlobal_d, 8, 0)]));

        pdf[i_f1(i, j, nzGlobal_d, 5, 0)] = (pdf[i_f1(i, j, nzGlobal_d, 6, 0)] - prc(0.333333333333333333) * tmp1) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 5, 0)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 11, 0)] = (pdf[i_f1(i, j, nzGlobal_d, 14, 0)] - prc(0.166666666666666667) * tmp1 + tnx) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 11, 0)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 12, 0)] = (pdf[i_f1(i, j, nzGlobal_d, 13, 0)] - prc(0.166666666666666667) * tmp1 - tnx) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 12, 0)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 15, 0)] = (pdf[i_f1(i, j, nzGlobal_d, 18, 0)] - prc(0.166666666666666667) * tmp1 + tny) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 15, 0)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 16, 0)] = (pdf[i_f1(i, j, nzGlobal_d, 17, 0)] - prc(0.166666666666666667) * tmp1 - tny) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 16, 0)] * wall_indicator;

        tnx = prc(0.5) * (pdf[i_f1(i, j, nzGlobal_d, 2, 1)] + pdf[i_f1(i, j, nzGlobal_d, 8, 1)] + pdf[i_f1(i, j, nzGlobal_d, 10, 1)]
            - (pdf[i_f1(i, j, nzGlobal_d, 1, 1)] + pdf[i_f1(i, j, nzGlobal_d, 7, 1)] + pdf[i_f1(i, j, nzGlobal_d, 9, 1)]));
        tny = prc(0.5) * (pdf[i_f1(i, j, nzGlobal_d, 4, 1)] + pdf[i_f1(i, j, nzGlobal_d, 10, 1)] + pdf[i_f1(i, j, nzGlobal_d, 9, 1)]
            - (pdf[i_f1(i, j, nzGlobal_d, 3, 1)] + pdf[i_f1(i, j, nzGlobal_d, 7, 1)] + pdf[i_f1(i, j, nzGlobal_d, 8, 1)]));

        pdf[i_f1(i, j, nzGlobal_d, 5, 1)] = (pdf[i_f1(i, j, nzGlobal_d, 6, 1)] - prc(0.333333333333333333) * tmp2) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 5, 1)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 11, 1)] = (pdf[i_f1(i, j, nzGlobal_d, 14, 1)] - prc(0.166666666666666667) * tmp2 + tnx) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 11, 1)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 12, 1)] = (pdf[i_f1(i, j, nzGlobal_d, 13, 1)] - prc(0.166666666666666667) * tmp2 - tnx) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 12, 1)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 15, 1)] = (pdf[i_f1(i, j, nzGlobal_d, 18, 1)] - prc(0.166666666666666667) * tmp2 + tny) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 15, 1)] * wall_indicator;
        pdf[i_f1(i, j, nzGlobal_d, 16, 1)] = (pdf[i_f1(i, j, nzGlobal_d, 17, 1)] - prc(0.166666666666666667) * tmp2 - tny) * (1 - wall_indicator) + pdf[i_f1(i, j, nzGlobal_d, 16, 1)] * wall_indicator;
    }
}

#pragma endregion (Outlet Boundary Conditions)

#pragma region (Periodic Boundary Conditions)
__global__ void perioidc_BC_Z_even(T_P* pdf) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        //****************************z direction******************************************
        pdf[i_f1(i, j, 1 + nzGlobal_d, 6, 0)] = pdf[i_f1(i, j, 1, 6, 0)];
        pdf[i_f1(i, j, 1 + nzGlobal_d, 14, 0)] = pdf[i_f1(i, j, 1, 14, 0)];
        pdf[i_f1(i, j, 1 + nzGlobal_d, 13, 0)] = pdf[i_f1(i, j, 1, 13, 0)];
        pdf[i_f1(i, j, 1 + nzGlobal_d, 18, 0)] = pdf[i_f1(i, j, 1, 18, 0)];
        pdf[i_f1(i, j, 1 + nzGlobal_d, 17, 0)] = pdf[i_f1(i, j, 1, 17, 0)];

        pdf[i_f1(i, j, 1 - 1, 5, 0)] = pdf[i_f1(i, j, nzGlobal_d, 5, 0)];
        pdf[i_f1(i, j, 1 - 1, 11, 0)] = pdf[i_f1(i, j, nzGlobal_d, 11, 0)];
        pdf[i_f1(i, j, 1 - 1, 12, 0)] = pdf[i_f1(i, j, nzGlobal_d, 12, 0)];
        pdf[i_f1(i, j, 1 - 1, 15, 0)] = pdf[i_f1(i, j, nzGlobal_d, 15, 0)];
        pdf[i_f1(i, j, 1 - 1, 16, 0)] = pdf[i_f1(i, j, nzGlobal_d, 16, 0)];

        pdf[i_f1(i, j, 1 + nzGlobal_d, 6, 1)] = pdf[i_f1(i, j, 1, 6, 1)];
        pdf[i_f1(i, j, 1 + nzGlobal_d, 14, 1)] = pdf[i_f1(i, j, 1, 14, 1)];
        pdf[i_f1(i, j, 1 + nzGlobal_d, 13, 1)] = pdf[i_f1(i, j, 1, 13, 1)];
        pdf[i_f1(i, j, 1 + nzGlobal_d, 18, 1)] = pdf[i_f1(i, j, 1, 18, 1)];
        pdf[i_f1(i, j, 1 + nzGlobal_d, 17, 1)] = pdf[i_f1(i, j, 1, 17, 1)];

        pdf[i_f1(i, j, 1 - 1, 5, 1)] = pdf[i_f1(i, j, nzGlobal_d, 5, 1)];
        pdf[i_f1(i, j, 1 - 1, 11, 1)] = pdf[i_f1(i, j, nzGlobal_d, 11, 1)];
        pdf[i_f1(i, j, 1 - 1, 12, 1)] = pdf[i_f1(i, j, nzGlobal_d, 12, 1)];
        pdf[i_f1(i, j, 1 - 1, 15, 1)] = pdf[i_f1(i, j, nzGlobal_d, 15, 1)];
        pdf[i_f1(i, j, 1 - 1, 16, 1)] = pdf[i_f1(i, j, nzGlobal_d, 16, 1)];
    }
}

__global__ void perioidc_BC_Y_even(T_P* pdf) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if (i <= nxGlobal_d && k <= nzGlobal_d) {
        //****************************y direction******************************************
        pdf[i_f1(i, 1 + nyGlobal_d, k, 4, 0)] = pdf[i_f1(i, 1, k, 4, 0)];
        pdf[i_f1(i, 1 + nyGlobal_d, k, 10, 0)] = pdf[i_f1(i, 1, k, 10, 0)];
        pdf[i_f1(i, 1 + nyGlobal_d, k, 9, 0)] = pdf[i_f1(i, 1, k, 9, 0)];
        pdf[i_f1(i, 1 + nyGlobal_d, k, 16, 0)] = pdf[i_f1(i, 1, k, 16, 0)];
        pdf[i_f1(i, 1 + nyGlobal_d, k, 18, 0)] = pdf[i_f1(i, 1, k, 18, 0)];

        pdf[i_f1(i, 1 - 1, k, 3, 0)] = pdf[i_f1(i, nyGlobal_d, k, 3, 0)];
        pdf[i_f1(i, 1 - 1, k, 7, 0)] = pdf[i_f1(i, nyGlobal_d, k, 7, 0)];
        pdf[i_f1(i, 1 - 1, k, 8, 0)] = pdf[i_f1(i, nyGlobal_d, k, 8, 0)];
        pdf[i_f1(i, 1 - 1, k, 15, 0)] = pdf[i_f1(i, nyGlobal_d, k, 15, 0)];
        pdf[i_f1(i, 1 - 1, k, 17, 0)] = pdf[i_f1(i, nyGlobal_d, k, 17, 0)];

        pdf[i_f1(i, 1 + nyGlobal_d, k, 4, 1)] = pdf[i_f1(i, 1, k, 4, 1)];
        pdf[i_f1(i, 1 + nyGlobal_d, k, 10, 1)] = pdf[i_f1(i, 1, k, 10, 1)];
        pdf[i_f1(i, 1 + nyGlobal_d, k, 9, 1)] = pdf[i_f1(i, 1, k, 9, 1)];
        pdf[i_f1(i, 1 + nyGlobal_d, k, 16, 1)] = pdf[i_f1(i, 1, k, 16, 1)];
        pdf[i_f1(i, 1 + nyGlobal_d, k, 18, 1)] = pdf[i_f1(i, 1, k, 18, 1)];

        pdf[i_f1(i, 1 - 1, k, 3, 1)] = pdf[i_f1(i, nyGlobal_d, k, 3, 1)];
        pdf[i_f1(i, 1 - 1, k, 7, 1)] = pdf[i_f1(i, nyGlobal_d, k, 7, 1)];
        pdf[i_f1(i, 1 - 1, k, 8, 1)] = pdf[i_f1(i, nyGlobal_d, k, 8, 1)];
        pdf[i_f1(i, 1 - 1, k, 15, 1)] = pdf[i_f1(i, nyGlobal_d, k, 15, 1)];
        pdf[i_f1(i, 1 - 1, k, 17, 1)] = pdf[i_f1(i, nyGlobal_d, k, 17, 1)];
    }
}

__global__ void perioidc_BC_ZY_even(T_P* pdf) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if (i <= nxGlobal_d) {
        // ****************************edges******************************************
        pdf[i_f1(i, nyGlobal_d + 1, nzGlobal_d + 1, 18, 0)] = pdf[i_f1(i, 1, 1, 18, 0)];
        pdf[i_f1(i, nyGlobal_d + 1, 1 - 1, 16, 0)] = pdf[i_f1(i, 1, nzGlobal_d, 16, 0)];
        pdf[i_f1(i, 1 - 1, nzGlobal_d + 1, 17, 0)] = pdf[i_f1(i, nyGlobal_d, 1, 17, 0)];
        pdf[i_f1(i, 1 - 1, 1 - 1, 15, 0)] = pdf[i_f1(i, nyGlobal_d, nzGlobal_d, 15, 0)];

        pdf[i_f1(i, nyGlobal_d + 1, nzGlobal_d + 1, 18, 1)] = pdf[i_f1(i, 1, 1, 18, 1)];
        pdf[i_f1(i, nyGlobal_d + 1, 1 - 1, 16, 1)] = pdf[i_f1(i, 1, nzGlobal_d, 16, 1)];
        pdf[i_f1(i, 1 - 1, nzGlobal_d + 1, 17, 1)] = pdf[i_f1(i, nyGlobal_d, 1, 17, 1)];
        pdf[i_f1(i, 1 - 1, 1 - 1, 15, 1)] = pdf[i_f1(i, nyGlobal_d, nzGlobal_d, 15, 1)];
    }
}

__global__ void perioidc_BC_Z_odd(T_P* pdf) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        //****************************z direction******************************************
        pdf[i_f1(i, j, 1, 6, 0)] = pdf[i_f1(i, j, nzGlobal_d + 1, 6, 0)];
        pdf[i_f1(i, j, 1, 14, 0)] = pdf[i_f1(i, j, nzGlobal_d + 1, 14, 0)];
        pdf[i_f1(i, j, 1, 13, 0)] = pdf[i_f1(i, j, nzGlobal_d + 1, 13, 0)];
        pdf[i_f1(i, j, 1, 18, 0)] = pdf[i_f1(i, j, nzGlobal_d + 1, 18, 0)];
        pdf[i_f1(i, j, 1, 17, 0)] = pdf[i_f1(i, j, nzGlobal_d + 1, 17, 0)];

        pdf[i_f1(i, j, nzGlobal_d, 5, 0)] = pdf[i_f1(i, j, 1 - 1, 5, 0)];
        pdf[i_f1(i, j, nzGlobal_d, 11, 0)] = pdf[i_f1(i, j, 1 - 1, 11, 0)];
        pdf[i_f1(i, j, nzGlobal_d, 12, 0)] = pdf[i_f1(i, j, 1 - 1, 12, 0)];
        pdf[i_f1(i, j, nzGlobal_d, 15, 0)] = pdf[i_f1(i, j, 1 - 1, 15, 0)];
        pdf[i_f1(i, j, nzGlobal_d, 16, 0)] = pdf[i_f1(i, j, 1 - 1, 16, 0)];

        pdf[i_f1(i, j, 1, 6, 1)] = pdf[i_f1(i, j, nzGlobal_d + 1, 6, 1)];
        pdf[i_f1(i, j, 1, 14, 1)] = pdf[i_f1(i, j, nzGlobal_d + 1, 14, 1)];
        pdf[i_f1(i, j, 1, 13, 1)] = pdf[i_f1(i, j, nzGlobal_d + 1, 13, 1)];
        pdf[i_f1(i, j, 1, 18, 1)] = pdf[i_f1(i, j, nzGlobal_d + 1, 18, 1)];
        pdf[i_f1(i, j, 1, 17, 1)] = pdf[i_f1(i, j, nzGlobal_d + 1, 17, 1)];

        pdf[i_f1(i, j, nzGlobal_d, 5, 1)] = pdf[i_f1(i, j, 1 - 1, 5, 1)];
        pdf[i_f1(i, j, nzGlobal_d, 11, 1)] = pdf[i_f1(i, j, 1 - 1, 11, 1)];
        pdf[i_f1(i, j, nzGlobal_d, 12, 1)] = pdf[i_f1(i, j, 1 - 1, 12, 1)];
        pdf[i_f1(i, j, nzGlobal_d, 15, 1)] = pdf[i_f1(i, j, 1 - 1, 15, 1)];
        pdf[i_f1(i, j, nzGlobal_d, 16, 1)] = pdf[i_f1(i, j, 1 - 1, 16, 1)];
    }
}

__global__ void perioidc_BC_Y_odd(T_P* pdf) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if (i <= nxGlobal_d && k <= nzGlobal_d) {
        //****************************y direction******************************************
        pdf[i_f1(i, 1, k, 4, 0)] = pdf[i_f1(i, nyGlobal_d + 1, k, 4, 0)];
        pdf[i_f1(i, 1, k, 10, 0)] = pdf[i_f1(i, nyGlobal_d + 1, k, 10, 0)];
        pdf[i_f1(i, 1, k, 9, 0)] = pdf[i_f1(i, nyGlobal_d + 1, k, 9, 0)];
        pdf[i_f1(i, 1, k, 16, 0)] = pdf[i_f1(i, nyGlobal_d + 1, k, 16, 0)];
        pdf[i_f1(i, 1, k, 18, 0)] = pdf[i_f1(i, nyGlobal_d + 1, k, 18, 0)];

        pdf[i_f1(i, nyGlobal_d, k, 3, 0)] = pdf[i_f1(i, 1 - 1, k, 3, 0)];
        pdf[i_f1(i, nyGlobal_d, k, 7, 0)] = pdf[i_f1(i, 1 - 1, k, 7, 0)];
        pdf[i_f1(i, nyGlobal_d, k, 8, 0)] = pdf[i_f1(i, 1 - 1, k, 8, 0)];
        pdf[i_f1(i, nyGlobal_d, k, 15, 0)] = pdf[i_f1(i, 1 - 1, k, 15, 0)];
        pdf[i_f1(i, nyGlobal_d, k, 17, 0)] = pdf[i_f1(i, 1 - 1, k, 17, 0)];

        pdf[i_f1(i, 1, k, 4, 1)] = pdf[i_f1(i, nyGlobal_d + 1, k, 4, 1)];
        pdf[i_f1(i, 1, k, 10, 1)] = pdf[i_f1(i, nyGlobal_d + 1, k, 10, 1)];
        pdf[i_f1(i, 1, k, 9, 1)] = pdf[i_f1(i, nyGlobal_d + 1, k, 9, 1)];
        pdf[i_f1(i, 1, k, 16, 1)] = pdf[i_f1(i, nyGlobal_d + 1, k, 16, 1)];
        pdf[i_f1(i, 1, k, 18, 1)] = pdf[i_f1(i, nyGlobal_d + 1, k, 18, 1)];

        pdf[i_f1(i, nyGlobal_d, k, 3, 1)] = pdf[i_f1(i, 1 - 1, k, 3, 1)];
        pdf[i_f1(i, nyGlobal_d, k, 7, 1)] = pdf[i_f1(i, 1 - 1, k, 7, 1)];
        pdf[i_f1(i, nyGlobal_d, k, 8, 1)] = pdf[i_f1(i, 1 - 1, k, 8, 1)];
        pdf[i_f1(i, nyGlobal_d, k, 15, 1)] = pdf[i_f1(i, 1 - 1, k, 15, 1)];
        pdf[i_f1(i, nyGlobal_d, k, 17, 1)] = pdf[i_f1(i, 1 - 1, k, 17, 1)];
    }
}

__global__ void perioidc_BC_ZY_odd(T_P* pdf) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if (i <= nxGlobal_d) {
        // ****************************edges******************************************
        pdf[i_f1(i, 1, 1, 18, 0)] = pdf[i_f1(i, nyGlobal_d + 1, nzGlobal_d + 1, 18, 0)];
        pdf[i_f1(i, 1, nzGlobal_d, 16, 0)] = pdf[i_f1(i, nyGlobal_d + 1, 1 - 1, 16, 0)];
        pdf[i_f1(i, nyGlobal_d, 1, 17, 0)] = pdf[i_f1(i, 1 - 1, nzGlobal_d + 1, 17, 0)];
        pdf[i_f1(i, nyGlobal_d, nzGlobal_d, 15, 0)] = pdf[i_f1(i, 1 - 1, 1 - 1, 15, 0)];

        pdf[i_f1(i, 1, 1, 18, 1)] = pdf[i_f1(i, nyGlobal_d + 1, nzGlobal_d + 1, 18, 1)];
        pdf[i_f1(i, 1, nzGlobal_d, 16, 1)] = pdf[i_f1(i, nyGlobal_d + 1, 1 - 1, 16, 1)];
        pdf[i_f1(i, nyGlobal_d, 1, 17, 1)] = pdf[i_f1(i, 1 - 1, nzGlobal_d + 1, 17, 1)];
        pdf[i_f1(i, nyGlobal_d, nzGlobal_d, 15, 1)] = pdf[i_f1(i, 1 - 1, 1 - 1, 15, 1)];
    }
}

__global__ void periodic_phi_Z(int overlap_phi, T_P* phi) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        // **************************** faces ******************************************
        int k;
        for (k = 1; k <= overlap_phi; k++) {
            phi[i_s4(i, j, k + nzGlobal_d)] = phi[i_s4(i, j, k)];
            phi[i_s4(i, j, k - overlap_phi)] = phi[i_s4(i, j, nzGlobal_d + k - overlap_phi)];
        }
    }
}
__global__ void periodic_phi_Y(int overlap_phi, T_P* phi) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    if (i <= nxGlobal_d && k <= nzGlobal_d) {
        // **************************** faces ******************************************
        int j;
        for (j = 1; j <= overlap_phi; j++) {
            phi[i_s4(i, j + nyGlobal_d, k)] = phi[i_s4(i, j, k)];
            phi[i_s4(i, j - overlap_phi, k)] = phi[i_s4(i, nyGlobal_d + j - overlap_phi, k)];
        }
    }
}

__global__ void periodic_phi_ZY(int overlap_phi, T_P* phi) {

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if (i <= nxGlobal_d) {
        // **************************** edges ******************************************
        int k, j;
        for (k = 1; k <= overlap_phi; k++) {
            for (j = 1; j <= overlap_phi; j++) {
                phi[i_s4(i, j - overlap_phi, k - overlap_phi)] = phi[i_s4(i, nyGlobal_d + j - overlap_phi, nzGlobal_d + k - overlap_phi)];
                phi[i_s4(i, j + nyGlobal_d, k - overlap_phi)] = phi[i_s4(i, j, nzGlobal_d + k - overlap_phi)];
                phi[i_s4(i, j + nyGlobal_d, k + nzGlobal_d)] = phi[i_s4(i, j, k)];
                phi[i_s4(i, j - overlap_phi, k + nzGlobal_d)] = phi[i_s4(i, nyGlobal_d + j - overlap_phi, k)];
            }
        }
    }
}

#pragma endregion (Periodic Boundary Conditions)

#pragma region (Other Boundary Conditions)
//===================================================================================================================================================================================
//----------------------place a porous plate in the domain : 0 - no; 1 - block fluid 1; 2 - block fluid 2  ----------------------
//example1: inject fluid 2 (wetting)during imbibition cycle and block fluid 1 from exiting the inlet
//example2 : inject fluid 1 (nonwetting)during drainage cycle and block fluid 1 from exiting the outlet
//===================================================================================================================================================================================
//**************************before odd step kernel * ****************************************
__global__ void porous_plate_BC_before_odd(T_P* pdf) { // before streaming type BC
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        int zmin, zmax, z_porous_plate_local;

        zmin = 1;
        zmax = nzGlobal_d;

        if (Z_porous_plate_d >= zmin && Z_porous_plate_d <= zmax) {
            z_porous_plate_local = Z_porous_plate_d;
            if (porous_plate_cmd_d == 1) { // block fluid 1, default (assuming fluid 1 is the nonwetting phase)

                pdf[i_f1(i, j, z_porous_plate_local, 6, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 5, 0)];
                pdf[i_f1(i - 1, j, z_porous_plate_local, 13, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 12, 0)];
                pdf[i_f1(i + 1, j, z_porous_plate_local, 14, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 11, 0)];
                pdf[i_f1(i, j - 1, z_porous_plate_local, 17, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 16, 0)];
                pdf[i_f1(i, j + 1, z_porous_plate_local, 18, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 15, 0)];

                pdf[i_f1(i, j, z_porous_plate_local, 5, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 6, 0)];
                pdf[i_f1(i + 1, j, z_porous_plate_local, 12, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 13, 0)];
                pdf[i_f1(i - 1, j, z_porous_plate_local, 11, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 14, 0)];
                pdf[i_f1(i, j + 1, z_porous_plate_local, 16, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 17, 0)];
                pdf[i_f1(i, j - 1, z_porous_plate_local, 15, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 18, 0)];

                pdf[i_f1(i, j, z_porous_plate_local, 6, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 6, 1)];
                pdf[i_f1(i, j, z_porous_plate_local, 13, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 13, 1)];
                pdf[i_f1(i, j, z_porous_plate_local, 14, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 14, 1)];
                pdf[i_f1(i, j, z_porous_plate_local, 17, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 17, 1)];
                pdf[i_f1(i, j, z_porous_plate_local, 18, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 18, 1)];

                pdf[i_f1(i, j, z_porous_plate_local, 5, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 5, 1)];
                pdf[i_f1(i, j, z_porous_plate_local, 12, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 12, 1)];
                pdf[i_f1(i, j, z_porous_plate_local, 11, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 11, 1)];
                pdf[i_f1(i, j, z_porous_plate_local, 16, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 16, 1)];
                pdf[i_f1(i, j, z_porous_plate_local, 15, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 15, 1)];

            }
            else if (porous_plate_cmd_d == 2) { // block fluid 2

                pdf[i_f1(i, j, z_porous_plate_local, 6, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 5, 1)];
                pdf[i_f1(i - 1, j, z_porous_plate_local, 13, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 12, 1)];
                pdf[i_f1(i + 1, j, z_porous_plate_local, 14, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 11, 1)];
                pdf[i_f1(i, j - 1, z_porous_plate_local, 17, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 16, 1)];
                pdf[i_f1(i, j + 1, z_porous_plate_local, 18, 1)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 15, 1)];

                pdf[i_f1(i, j, z_porous_plate_local, 5, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 6, 1)];
                pdf[i_f1(i + 1, j, z_porous_plate_local, 12, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 13, 1)];
                pdf[i_f1(i - 1, j, z_porous_plate_local, 11, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 14, 1)];
                pdf[i_f1(i, j + 1, z_porous_plate_local, 16, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 17, 1)];
                pdf[i_f1(i, j - 1, z_porous_plate_local, 15, 1)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 18, 1)];

                pdf[i_f1(i, j, z_porous_plate_local, 6, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 6, 0)];
                pdf[i_f1(i, j, z_porous_plate_local, 13, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 13, 0)];
                pdf[i_f1(i, j, z_porous_plate_local, 14, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 14, 0)];
                pdf[i_f1(i, j, z_porous_plate_local, 17, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 17, 0)];
                pdf[i_f1(i, j, z_porous_plate_local, 18, 0)] = pdf[i_f1(i, j, z_porous_plate_local + 1, 18, 0)];

                pdf[i_f1(i, j, z_porous_plate_local, 5, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 5, 0)];
                pdf[i_f1(i, j, z_porous_plate_local, 12, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 12, 0)];
                pdf[i_f1(i, j, z_porous_plate_local, 11, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 11, 0)];
                pdf[i_f1(i, j, z_porous_plate_local, 16, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 16, 0)];
                pdf[i_f1(i, j, z_porous_plate_local, 15, 0)] = pdf[i_f1(i, j, z_porous_plate_local - 1, 15, 0)];

            } 

        }
    }
}

// ************************** after odd step kernel *****************************************
__global__ void porous_plate_BC_after_odd(T_P* pdf) { // after streaming type BC
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i <= nxGlobal_d && j <= nyGlobal_d) {
        int zmin, zmax, z_porous_plate_local;

        zmin = 1;
        zmax = nzGlobal_d;

        if (Z_porous_plate_d >= zmin && Z_porous_plate_d <= zmax) {
            z_porous_plate_local = Z_porous_plate_d;
            if (porous_plate_cmd_d == 1) { // block fluid 1, default (assuming fluid 1 is the nonwetting phase)
                pdf[i_f1(i, j, z_porous_plate_local - 1, 5, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 6, 0)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 12, 0)] = pdf[i_f1(i - 1, j, z_porous_plate_local, 13, 0)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 11, 0)] = pdf[i_f1(i + 1, j, z_porous_plate_local, 14, 0)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 16, 0)] = pdf[i_f1(i, j - 1, z_porous_plate_local, 17, 0)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 15, 0)] = pdf[i_f1(i, j + 1, z_porous_plate_local, 18, 0)];

                pdf[i_f1(i, j, z_porous_plate_local + 1, 6, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 5, 0)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 13, 0)] = pdf[i_f1(i + 1, j, z_porous_plate_local, 12, 0)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 14, 0)] = pdf[i_f1(i - 1, j, z_porous_plate_local, 11, 0)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 17, 0)] = pdf[i_f1(i, j + 1, z_porous_plate_local, 16, 0)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 18, 0)] = pdf[i_f1(i, j - 1, z_porous_plate_local, 15, 0)];

                pdf[i_f1(i, j, z_porous_plate_local - 1, 5, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 5, 1)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 12, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 12, 1)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 11, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 11, 1)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 16, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 16, 1)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 15, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 15, 1)];

                pdf[i_f1(i, j, z_porous_plate_local + 1, 6, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 6, 1)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 13, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 13, 1)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 14, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 14, 1)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 17, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 17, 1)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 18, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 18, 1)];
            }
            else if (porous_plate_cmd_d == 2) { // block fluid 2

                pdf[i_f1(i, j, z_porous_plate_local - 1, 5, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 6, 1)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 12, 1)] = pdf[i_f1(i - 1, j, z_porous_plate_local, 13, 1)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 11, 1)] = pdf[i_f1(i + 1, j, z_porous_plate_local, 14, 1)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 16, 1)] = pdf[i_f1(i, j - 1, z_porous_plate_local, 17, 1)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 15, 1)] = pdf[i_f1(i, j + 1, z_porous_plate_local, 18, 1)];

                pdf[i_f1(i, j, z_porous_plate_local + 1, 6, 1)] = pdf[i_f1(i, j, z_porous_plate_local, 5, 1)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 13, 1)] = pdf[i_f1(i + 1, j, z_porous_plate_local, 12, 1)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 14, 1)] = pdf[i_f1(i - 1, j, z_porous_plate_local, 11, 1)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 17, 1)] = pdf[i_f1(i, j + 1, z_porous_plate_local, 16, 1)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 18, 1)] = pdf[i_f1(i, j - 1, z_porous_plate_local, 15, 1)];

                pdf[i_f1(i, j, z_porous_plate_local + 1, 6, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 6, 0)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 13, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 13, 0)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 14, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 14, 0)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 17, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 17, 0)];
                pdf[i_f1(i, j, z_porous_plate_local + 1, 18, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 18, 0)];

                pdf[i_f1(i, j, z_porous_plate_local - 1, 5, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 5, 0)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 12, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 12, 0)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 11, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 11, 0)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 16, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 16, 0)];
                pdf[i_f1(i, j, z_porous_plate_local - 1, 15, 0)] = pdf[i_f1(i, j, z_porous_plate_local, 15, 0)];

            }
        }

    }
}


#pragma endregion (Other Boundary Conditions)

#pragma endregion (Boundary Conditions)


void main_iteration_kernel_GPU() {
    /* define grid structure */
    dim3 block;
    dim3 grid;

    if (ntime % 2 == 0) {
        /* ************************** even step ***************************************** */
        block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = block_Threads_Z;
        grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = (nzGlobal + block.z - 1) / block.z;
        kernel_even_color_GPU << <grid, block >> > (1, nxGlobal, 1, nyGlobal, 1, nzGlobal, walls_d, pdf_d, phi_d, cn_x_d, cn_y_d, cn_z_d, curv_d, c_norm_d);
        cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        /* ************************** even step ***************************************** */
        /* ************************** periodic boundary conditions  ***************************************** */
        if (kper) {
            block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
            grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
            perioidc_BC_Z_even << <grid, block >> > (pdf_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            periodic_phi_Z << <grid, block >> > (4, phi_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        }
        if (jper) {
            block.x = block_Threads_X;    block.y = 1;    block.z = block_Threads_Z;
            grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = 1; grid.z = (nzGlobal + block.z - 1) / block.z;   
            perioidc_BC_Y_even << <grid, block >> > (pdf_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            periodic_phi_Y << <grid, block >> > (4, phi_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());

        }
        if (jper && kper) {
            block.x = block_Threads_X; block.y = 1; block.z = 1;
            grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = 1; grid.z = 1;
            perioidc_BC_ZY_even << <grid, block >> > (pdf_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            periodic_phi_ZY << <grid, block >> > (4, phi_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        }
        /* ************************** periodic boundary conditions  ***************************************** */

        if (kper == 0 && domain_wall_status_z_min == 0 && domain_wall_status_z_max == 0) { // non - periodic BC along flow direction(z)
            if (inlet_BC == 1) {
                block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
                grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
                inlet_bounce_back_velocity_BC_before_odd_GPU << <grid, block >> > (walls_d, phi_d, pdf_d, W_in_d);
            }
            else if (inlet_BC == 2) {
                block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
                grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
                inlet_Zou_He_pressure_BC_before_odd_GPU << <grid, block >> > (rho_in, walls_d, phi_d, pdf_d); // pressure inlet bc
            }
            if (outlet_BC == 1) {
                block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
                grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
                outlet_convective_BC_before_odd_GPU << <grid, block >> > (walls_d, phi_d, phi_convec_bc_d, pdf_d, f_convec_bc_d, g_convec_bc_d);
            }
            else if (outlet_BC == 2) {
                block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
                grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
                //grid.x = ceil(double(nxGlobal) / double(block.x));   grid.y = ceil(double(nyGlobal) / double(block.y)); grid.z = 1;
                outlet_Zou_He_pressure_BC_before_odd_GPU << <grid, block >> > (rho_out, walls_d, phi_d, pdf_d); // pressure outlet bc
            }
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        }
        if (porous_plate_cmd != 0) { // place a porous plate
            block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
            porous_plate_BC_before_odd << <grid, block >> > (pdf_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        }

    }
    else {
        /* ************************** odd step ***************************************** */
        block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = block_Threads_Z;
        grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = (nzGlobal + block.z - 1) / block.z;
        kernel_odd_color_GPU << <grid, block >> > (1, nxGlobal, 1, nyGlobal, 1, nzGlobal, walls_d, pdf_d, phi_d, cn_x_d, cn_y_d, cn_z_d, curv_d, c_norm_d);
        cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        /* ************************** odd step ***************************************** */
        /* ************************** periodic boundary conditions  ***************************************** */
        if (kper) {
            block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
            grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
            perioidc_BC_Z_odd << <grid, block >> > (pdf_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            periodic_phi_Z << <grid, block >> > (4, phi_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        }
        if (jper) {
            block.x = block_Threads_X;    block.y = 1;    block.z = block_Threads_Z;
            grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = 1; grid.z = (nzGlobal + block.z - 1) / block.z;
            perioidc_BC_Y_odd << <grid, block >> > (pdf_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            periodic_phi_Y << <grid, block >> > (4, phi_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        }
        if (jper && kper) {
            block.x = block_Threads_X; block.y = 1; block.z = 1;
            grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = 1; grid.z = 1;
            perioidc_BC_ZY_odd << <grid, block >> > (pdf_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
            periodic_phi_ZY << <grid, block >> > (4, phi_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        }
        /* ************************** periodic boundary conditions  ***************************************** */

        if (kper == 0 && domain_wall_status_z_min == 0 && domain_wall_status_z_max == 0) { // non - periodic BC along flow direction(z)
            if (inlet_BC == 1) { // velocity inlet bc
                block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
                grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
                inlet_bounce_back_velocity_BC_after_odd_GPU << <grid, block >> > (walls_d, phi_d, pdf_d, W_in_d);
            }
            else if (inlet_BC == 2) { // pressure inlet bc
                block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
                grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
                inlet_Zou_He_pressure_BC_after_odd_GPU << <grid, block >> > (rho_in, walls_d, phi_d, pdf_d); // pressure inlet bc
            }
            if (outlet_BC == 1) { // convective outlet bc
                block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
                grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
                outlet_convective_BC_after_odd_GPU << <grid, block >> >
                    (walls_d, phi_d, phi_convec_bc_d, pdf_d, f_convec_bc_d, g_convec_bc_d);
            }
            else if (outlet_BC == 2) { // pressure outlet bc
                block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
                grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = 1;
                outlet_Zou_He_pressure_BC_after_odd_GPU << <grid, block >> > (rho_out, walls_d, phi_d, pdf_d);   // pressure outlet bc
            }
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        }
        if (porous_plate_cmd != 0) { // place a porous plate
            block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = 1;
            porous_plate_BC_after_odd << <grid, block >> > (pdf_d);
            cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        }

    }

    /* Color Gradient */
#pragma region (color gradient)
     /* ~~~~~~~~~~~~~~~~~~~~~~~ extrapolate phi values to solid boundary nodes ~~~~~~~~~~~~~~~~~~ */
    block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = block_Threads_Z;
    grid.x = (nxGlobal + 6 + block.x - 1) / block.x; grid.y = (nyGlobal + 6 + block.y - 1) / block.y; grid.z = (nzGlobal + 6 + block.z - 1) / block.z;
    extrapolate_phi_toSolid << <grid, block >> > (walls_type_d, phi_d);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    /* ~~~~~~~~~~~~~~~~~~ calculate normal directions of interfaces from phi gradient ~~~~~~~~~~~~~~~~~~ */
    block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = block_Threads_Z;
    grid.x = (nxGlobal + 4 + block.x - 1) / block.x; grid.y = (nyGlobal + 4 + block.y - 1) / block.y; grid.z = (nzGlobal + 4 + block.z - 1) / block.z;
    normalDirectionsOfInterfaces << < grid, block >> > (walls_d, phi_d, cn_x_d, cn_y_d, cn_z_d, c_norm_d);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    /* ~~~~~~~~~~~~~~~ numerically alter the normal directions of interfaces to desired contact angle ~~~~~~~~~~~~~~~~ */
    block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = block_Threads_Z;
    grid.x = (nxGlobal + 4 + block.x - 1) / block.x; grid.y = (nyGlobal + 4 + block.y - 1) / block.y; grid.z = (nzGlobal + 4 + block.z - 1) / block.z;
    alter_color_gradient_solid_surface_GPU << < grid, block >> > (walls_type_d, cn_x_d, cn_y_d, cn_z_d, c_norm_d, s_nx_d, s_ny_d, s_nz_d);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    /* ~~~~~~~~~~~~~~ extrapolate normal direction info to solid boundary nodes, to minimize unbalanced forces ~~~~~~~~~~~~~~ */
    block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = block_Threads_Z;
    grid.x = (nxGlobal + 2 + block.x - 1) / block.x; grid.y = (nyGlobal + 2 + block.y - 1) / block.y; grid.z = (nzGlobal + 2 + block.z - 1) / block.z;
    extrapolateNormalToSolid << <grid, block >> > (walls_type_d, cn_x_d, cn_y_d, cn_z_d, phi_d);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
    /* ~~~~~~~~~~~~~~~~~~ calculate CSF forces based on interace curvature  ~~~~~~~~~~~~~~~~~~ */
    //block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = block_Threads_Z;
    block.x = block_Threads_X;    block.y = block_Threads_Y;    block.z = block_Threads_Z;
    grid.x = (nxGlobal + block.x - 1) / block.x; grid.y = (nyGlobal + block.y - 1) / block.y; grid.z = (nzGlobal + block.z - 1) / block.z;
    CSF_Forces << <grid, block >> > (cn_x_d, cn_y_d, cn_z_d, curv_d);
    cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
#pragma endregion (color gradient)


    /* send results back to the host side */
    if (ntime % ntime_monitor == 0 || ntime % ntime_animation == 0 || ntime % ntime_visual == 0 || ntime % ntime_clock_sum == 0) {
        cout << "Copying results back to the host side .... ";
        cudaErrorCheck(cudaMemcpy(u, u_d, mem_size_s1_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(v, v_d, mem_size_s1_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(w, w_d, mem_size_s1_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(rho, rho_d, mem_size_s1_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(phi, phi_d, mem_size_s4_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(curv, curv_d, mem_size_s1_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(c_norm, c_norm_d, mem_size_s2_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(cn_x, cn_x_d, mem_size_s2_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(cn_y, cn_y_d, mem_size_s2_TP, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(cn_z, cn_z_d, mem_size_s2_TP, cudaMemcpyDeviceToHost));
        
        cudaErrorCheck(cudaMemcpy(pdf, pdf_d, mem_size_f1_TP, cudaMemcpyDeviceToHost));

        cudaErrorCheck(cudaDeviceSynchronize()); cudaErrorCheck(cudaPeekAtLastError());
        cout << " Complete " << endl;
    }

}





