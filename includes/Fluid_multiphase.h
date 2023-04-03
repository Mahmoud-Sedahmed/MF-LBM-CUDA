#ifndef Fluid_multiphase_H
#define Fluid_multiphase_H

#include "Module_extern.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~command input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// breakthrough check(only works when use open BCsand Ca > 0) : whether exiting the simulations when the invading phase reaches the outlet
// 0 - not check; 1 - check
int breakthrough_check;
int phase_dist_cmd;   // initialization of the phase field : 0 - interface location at z axis; 1 - random distribution according to target saturation; 2 - other distribution
int fluid_injection_indicator;   //1 - inject fluid 1; 2 - inject fluid 2; to be used with buffer zone injection for rel - perm measurement

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~fluid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// order parameter, phase gradient, curvature, norm of color gradient, using the phase field difference to indicate steady state
T_P* phi, * cn_x, * cn_y, * cn_z, * c_norm, * curv, * phi_old;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~flow condition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// fluid distribution of previous step for convective outlet BC
T_P* f_convec_bc, * g_convec_bc, * phi_convec_bc;

T_P w_darcy[3];

T_P kinetic_energy[2]; // kinetic enegy of each phase

int Z_porous_plate;  // porous plate or membrane used to prevent one phase from passing through

T_P rt2, rti2, la_nu2, la_nui2, phys_nu2;
T_P phi_inlet;   // injecting fluid selection
int outlet_phase1_sum;   // total fluid 1 nodes at the outlet(indicating breakthrough)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~multiphase model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
T_P RK_weight2 = prc(1.) / prc(sqrt)(prc(2.)) / prc(36.); // R - K recolor scheme weight for diagonal directions
// phase gradient calculation weights
T_P ISO4[2] = { prc(1.) / prc(6.), prc(1.) / prc(12.) };
T_P ISO8[7] = { prc(4.) / prc(45.), prc(1.) / prc(21.), prc(2.) / prc(105.), prc(5.) / prc(504.), prc(1.) / prc(315.), prc(1.) / prc(630.), prc(1.) / prc(5040.) };
// surface tension, contact angle
T_P lbm_gamma, phys_gamma, theta, lbm_beta, ca_0, u_filter, ca;
// initial interface position, saturation
T_P interface_z0, Sa0, saturation, saturation_full_domain, sa_inject, sa_target, mass1_sum, mass2_sum, saturation_old, vol1_sum, vol2_sum;
int lap_mr;   // multirange color gradient model overlaping area
// temporary arrays
T_P* fl1, * fl2, * sa1, * mass1, * mass2, * vol1, * vol2;

T_P cos_theta;

/* GPU pointers */
T_P* phi_d, * cn_x_d, * cn_y_d, * cn_z_d, * c_norm_d, * curv_d, * phi_old_d;
T_P* f_convec_bc_d, * g_convec_bc_d, * phi_convec_bc_d;
T_P* fl1_d, * fl2_d, * sa1_d, * mass1_d, * mass2_d, * vol1_d, * vol2_d;
#endif