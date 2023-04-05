#include "externLib.h"
#include "solver_precision.h"
#include "preprocessor.h"
#include "Module_extern.h"
#include "Fluid_singlephase_extern.h"
#include "Fluid_multiphase_extern.h"
#include "Idx_cpu.h"

#include "Phase_gradient.h"


//=====================================================================================================================================
//----------------------calculate color gradient----------------------
//=====================================================================================================================================
void color_gradient() {
    long long i, j, k, overlap_temp;
    T_P kxx, kxy, kxz, kyx, kyy, kyz, kzx, kzy, kzz;

    /* ~~~~~~~~~~~~~~~~~~~~~~~ extrapolate phi values to solid boundary nodes ~~~~~~~~~~~~~~~~~~ */
    for (k = 1 - 3; k <= nzGlobal + 3; k++) {
        for (j = 1 - 3; j <= nyGlobal + 3; j++) {
            for (i = 1 - 3; i <= nxGlobal + 3; i++) {
                int node_type_loc = walls_type[i_s4(i, j, k)];
                if (node_type_loc == 2) { // solid boundary node
                    T_P phi_sum = prc(0.), weight_sum = prc(0.);
                    for (int q = 1; q < 19; q++) {
                        long long iex = i + ex[q];
                        long long iey = j + ey[q];
                        long long iez = k + ez[q];
                        int node_type_neb = walls_type[i_s4(iex, iey, iez)];
                        if (node_type_neb <= 0) {
                            phi_sum += phi[i_s4(iex, iey, iez)] * w_equ[q];
                            weight_sum += w_equ[q];
                        }
                    }
                    phi[i_s4(i, j, k)] = phi_sum / weight_sum;

                }
            }
        }
    }

    /* ~~~~~~~~~~~~~~~~~~ calculate normal directions of interfaces from phi gradient ~~~~~~~~~~~~~~~~~~ */
    overlap_temp = 2;
    for (k = 1 - overlap_temp; k <= nzGlobal + overlap_temp; k++) {
        for (j = 1 - overlap_temp; j <= nyGlobal + overlap_temp; j++) {
            for (i = 1 - overlap_temp; i <= nxGlobal + overlap_temp; i++) {
                cn_x[i_s2(i, j, k)] =
                    ISO4[1 - 1] * (phi[i_s4(i + 1, j, k)] - phi[i_s4(i - 1, j, k)]) +

                    ISO4[2 - 1] * (
                        phi[i_s4(i + 1, j + 1, k)] - phi[i_s4(i - 1, j - 1, k)] +
                        phi[i_s4(i + 1, j - 1, k)] - phi[i_s4(i - 1, j + 1, k)] +
                        phi[i_s4(i + 1, j, k + 1)] - phi[i_s4(i - 1, j, k - 1)] +
                        phi[i_s4(i + 1, j, k - 1)] - phi[i_s4(i - 1, j, k + 1)]);

                cn_y[i_s2(i, j, k)] =
                    ISO4[1 - 1] * (phi[i_s4(i, j + 1, k)] - phi[i_s4(i, j - 1, k)]) +

                    ISO4[2 - 1] * (
                        phi[i_s4(i + 1, j + 1, k)] - phi[i_s4(i - 1, j - 1, k)] +
                        phi[i_s4(i - 1, j + 1, k)] - phi[i_s4(i + 1, j - 1, k)] +
                        phi[i_s4(i, j + 1, k + 1)] - phi[i_s4(i, j - 1, k - 1)] +
                        phi[i_s4(i, j + 1, k - 1)] - phi[i_s4(i, j - 1, k + 1)]);


                cn_z[i_s2(i, j, k)] =
                    ISO4[1 - 1] * (phi[i_s4(i, j, k + 1)] - phi[i_s4(i, j, k - 1)]) +

                    ISO4[2 - 1] * (
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
    }

    /* ~~~~~~~~~~~~~~~ numerically alter the normal directions of interfaces to desired contact angle ~~~~~~~~~~~~~~~~ */
    alter_color_gradient_solid_surface();

    /* ~~~~~~~~~~~~~~ extrapolate normal direction info to solid boundary nodes, to minimize unbalanced forces ~~~~~~~~~~~~~~ */
    for (k = 1 - 1; k <= nzGlobal + 1; k++) {
        for (j = 1 - 1; j <= nyGlobal + 1; j++) {
            for (i = 1 - 1; i <= nxGlobal + 1; i++) {
                int node_type_loc = walls_type[i_s4(i, j, k)];
                if (node_type_loc == 2) { // solid boundary node
                    T_P cn_x_sum = prc(0.), cn_y_sum = prc(0.), cn_z_sum = prc(0.), weight_sum = prc(0.);
                    for (int q = 1; q < 19; q++) {
                        long long iex = i + ex[q];
                        long long iey = j + ey[q];
                        long long iez = k + ez[q];
                        int node_type_neb = walls_type[i_s4(iex, iey, iez)];
                        if (node_type_neb <= 0) {
                            cn_x_sum += cn_x[i_s2(iex, iey, iez)] * w_equ[q];
                            cn_y_sum += cn_y[i_s2(iex, iey, iez)] * w_equ[q];
                            cn_z_sum += cn_z[i_s2(iex, iey, iez)] * w_equ[q];
                            weight_sum += w_equ[q];
                        }
                    }
                    cn_x[i_s2(i, j, k)] = cn_x_sum / weight_sum;
                    cn_y[i_s2(i, j, k)] = cn_y_sum / weight_sum;
                    cn_z[i_s2(i, j, k)] = cn_z_sum / weight_sum;
                }
            }
        }
    }

    /* ~~~~~~~~~~~~~~~~~~ calculate CSF forces based on interace curvature  ~~~~~~~~~~~~~~~~~~ */
    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                kxx =
                    ISO4[1 - 1] * (cn_x[i_s2(i + 1, j, k)] - cn_x[i_s2(i - 1, j, k)]) +

                    ISO4[2 - 1] * (
                        cn_x[i_s2(i + 1, j + 1, k)] - cn_x[i_s2(i - 1, j - 1, k)] +
                        cn_x[i_s2(i + 1, j - 1, k)] - cn_x[i_s2(i - 1, j + 1, k)] +
                        cn_x[i_s2(i + 1, j, k + 1)] - cn_x[i_s2(i - 1, j, k - 1)] +
                        cn_x[i_s2(i + 1, j, k - 1)] - cn_x[i_s2(i - 1, j, k + 1)]);

                kyy =
                    ISO4[1 - 1] * (cn_y[i_s2(i, j + 1, k)] - cn_y[i_s2(i, j - 1, k)]) +

                    ISO4[2 - 1] * (
                        cn_y[i_s2(i + 1, j + 1, k)] - cn_y[i_s2(i - 1, j - 1, k)] +
                        cn_y[i_s2(i - 1, j + 1, k)] - cn_y[i_s2(i + 1, j - 1, k)] +
                        cn_y[i_s2(i, j + 1, k + 1)] - cn_y[i_s2(i, j - 1, k - 1)] +
                        cn_y[i_s2(i, j + 1, k - 1)] - cn_y[i_s2(i, j - 1, k + 1)]);

                kzz =
                    ISO4[1 - 1] * (cn_z[i_s2(i, j, k + 1)] - cn_z[i_s2(i, j, k - 1)]) +

                    ISO4[2 - 1] * (
                        cn_z[i_s2(i + 1, j, k + 1)] - cn_z[i_s2(i - 1, j, k - 1)] +
                        cn_z[i_s2(i - 1, j, k + 1)] - cn_z[i_s2(i + 1, j, k - 1)] +
                        cn_z[i_s2(i, j + 1, k + 1)] - cn_z[i_s2(i, j - 1, k - 1)] +
                        cn_z[i_s2(i, j - 1, k + 1)] - cn_z[i_s2(i, j + 1, k - 1)]);


                kxy =
                    ISO4[1 - 1] * (cn_x[i_s2(i, j + 1, k)] - cn_x[i_s2(i, j - 1, k)]) +

                    ISO4[2 - 1] * (
                        cn_x[i_s2(i + 1, j + 1, k)] - cn_x[i_s2(i - 1, j - 1, k)] +
                        cn_x[i_s2(i - 1, j + 1, k)] - cn_x[i_s2(i + 1, j - 1, k)] +
                        cn_x[i_s2(i, j + 1, k + 1)] - cn_x[i_s2(i, j - 1, k - 1)] +
                        cn_x[i_s2(i, j + 1, k - 1)] - cn_x[i_s2(i, j - 1, k + 1)]);

                kxz =
                    ISO4[1 - 1] * (cn_x[i_s2(i, j, k + 1)] - cn_x[i_s2(i, j, k - 1)]) +

                    ISO4[2 - 1] * (
                        cn_x[i_s2(i + 1, j, k + 1)] - cn_x[i_s2(i - 1, j, k - 1)] +
                        cn_x[i_s2(i - 1, j, k + 1)] - cn_x[i_s2(i + 1, j, k - 1)] +
                        cn_x[i_s2(i, j + 1, k + 1)] - cn_x[i_s2(i, j - 1, k - 1)] +
                        cn_x[i_s2(i, j - 1, k + 1)] - cn_x[i_s2(i, j + 1, k - 1)]);

                kyx =
                    ISO4[1 - 1] * (cn_y[i_s2(i + 1, j, k)] - cn_y[i_s2(i - 1, j, k)]) +

                    ISO4[2 - 1] * (
                        cn_y[i_s2(i + 1, j + 1, k)] - cn_y[i_s2(i - 1, j - 1, k)] +
                        cn_y[i_s2(i + 1, j - 1, k)] - cn_y[i_s2(i - 1, j + 1, k)] +
                        cn_y[i_s2(i + 1, j, k + 1)] - cn_y[i_s2(i - 1, j, k - 1)] +
                        cn_y[i_s2(i + 1, j, k - 1)] - cn_y[i_s2(i - 1, j, k + 1)]);

                kyz =
                    ISO4[1 - 1] * (cn_y[i_s2(i, j, k + 1)] - cn_y[i_s2(i, j, k - 1)]) +

                    ISO4[2 - 1] * (
                        cn_y[i_s2(i + 1, j, k + 1)] - cn_y[i_s2(i - 1, j, k - 1)] +
                        cn_y[i_s2(i - 1, j, k + 1)] - cn_y[i_s2(i + 1, j, k - 1)] +
                        cn_y[i_s2(i, j + 1, k + 1)] - cn_y[i_s2(i, j - 1, k - 1)] +
                        cn_y[i_s2(i, j - 1, k + 1)] - cn_y[i_s2(i, j + 1, k - 1)]);

                kzx =
                    ISO4[1 - 1] * (cn_z[i_s2(i + 1, j, k)] - cn_z[i_s2(i - 1, j, k)]) +

                    ISO4[2 - 1] * (
                        cn_z[i_s2(i + 1, j + 1, k)] - cn_z[i_s2(i - 1, j - 1, k)] +
                        cn_z[i_s2(i + 1, j - 1, k)] - cn_z[i_s2(i - 1, j + 1, k)] +
                        cn_z[i_s2(i + 1, j, k + 1)] - cn_z[i_s2(i - 1, j, k - 1)] +
                        cn_z[i_s2(i + 1, j, k - 1)] - cn_z[i_s2(i - 1, j, k + 1)]);

                kzy =
                    ISO4[1 - 1] * (cn_z[i_s2(i, j + 1, k)] - cn_z[i_s2(i, j - 1, k)]) +

                    ISO4[2 - 1] * (
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
    }



}

// alter solid surface normal directions to control wettability
void alter_color_gradient_solid_surface() {
    long long i, j, k, iteration, iteration_max;
    T_P nwx, nwy, nwz, lambda, local_eps;
    T_P vcx0, vcy0, vcz0, vcx1, vcy1, vcz1, vcx2, vcy2, vcz2, err0, err1, err2, tmp;

    lambda = prc(0.5);
    local_eps = prc(1e-6);
    iteration_max = 4;

    for (k = 1 - 2; k <= nzGlobal + 2; k++) {
        for (j = 1 - 2; j <= nyGlobal + 2; j++) {
            for (i = 1 - 2; i <= nxGlobal + 2; i++) {
                int node_type_loc = walls_type[i_s4(i, j, k)];
                if (node_type_loc == -1) { // fluid boundary node
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

                        err0 = (nwx * vcx0 + nwy * vcy0 + nwz * vcz0) - cos_theta;

                        if ((pprc(abs)(vcx0 + nwx) + pprc(abs)(vcy0 + nwy) + pprc(abs)(vcz0 + nwz) > local_eps || pprc(abs)(vcx0 - nwx) + pprc(abs)(vcy0 - nwy) + pprc(abs)(vcz0 - nwz) > local_eps) && err0 > local_eps) {
                            // do not perform alteration when the normal direction of the solid surface aligned with the fluid interface direction,
                            // or the initial fluid direction is already the desired direction
                            err1 = (nwx * vcx1 + nwy * vcy1 + nwz * vcz1) - prc(sqrt)(vcx1 * vcx1 + vcy1 * vcy1 + vcz1 * vcz1) * cos_theta;
                            tmp = prc(1.) / (err1 - err0);
                            vcx2 = tmp * (vcx0 * err1 - vcx1 * err0);
                            vcy2 = tmp * (vcy0 * err1 - vcy1 * err0);
                            vcz2 = tmp * (vcz0 * err1 - vcz1 * err0);

                            err2 = (nwx * vcx2 + nwy * vcy2 + nwz * vcz2) - prc(sqrt)(vcx2 * vcx2 + vcy2 * vcy2 + vcz2 * vcz2) * cos_theta;

                            if (err2 > local_eps) {
                                for (iteration = 2; iteration <= iteration_max; iteration++) {
                                    vcx0 = vcx1;
                                    vcy0 = vcy1;
                                    vcz0 = vcz1;
                                    vcx1 = vcx2;
                                    vcy1 = vcy2;
                                    vcz1 = vcz2;
                                    err0 = (nwx * vcx0 + nwy * vcy0 + nwz * vcz0) - prc(sqrt)(vcx0 * vcx0 + vcy0 * vcy0 + vcz0 * vcz0) * cos_theta;
                                    err1 = (nwx * vcx1 + nwy * vcy1 + nwz * vcz1) - prc(sqrt)(vcx1 * vcx1 + vcy1 * vcy1 + vcz1 * vcz1) * cos_theta;
                                    tmp = prc(1.) / (err1 - err0);
                                    if (isinf(tmp)) break;
                                    vcx2 = tmp * (vcx0 * err1 - vcx1 * err0);
                                    vcy2 = tmp * (vcy0 * err1 - vcy1 * err0);
                                    vcz2 = tmp * (vcz0 * err1 - vcz1 * err0);
                                    err2 = (nwx * vcx2 + nwy * vcy2 + nwz * vcz2) - prc(sqrt)(vcx2 * vcx2 + vcy2 * vcy2 + vcz2 * vcz2) * cos_theta;
                                }
                                
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
    }
}