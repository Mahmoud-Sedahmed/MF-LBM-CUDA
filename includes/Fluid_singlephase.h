#ifndef FLUID_SINGLEPHASE_H
#define FLUID_SINGLEPHASE_H

#include "Module_extern.h"

T_P* W_in; // inlet velocity profile
T_P* u, * v, * w, * rho;
T_P* pdf;

T_P rt1, rti1, la_nu1, la_nui1;		 // viscosity for singlephase fluid
T_P force_x, force_y, force_z, D_force_z, force_z0;   // body force, default flow direction : z
T_P uin_max, uin_avg, uin_avg_0, flowrate, p_gradient, rho_out, rho_in, rho_in_avg, rho_avg_inlet, rho_avg_outlet, rho_drop;
T_P relaxation, uin_avg_convec, p_max, rho_in_max, umax_global;
// temporary arrays 
T_P* fl, * pre; // flowrate, pressure, saturation vs z

/* GPU pointers */
T_P* W_in_d, * u_d, * v_d, * w_d, * rho_d;
T_P* pdf_d;
T_P* fl_d, * pre_d;
#endif