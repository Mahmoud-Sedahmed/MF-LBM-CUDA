#ifndef FLUID_SINGLEPHASE_EXTERN_H
#define FLUID_SINGLEPHASE_EXTERN_H

#include "Module_extern.h"

extern T_P* W_in; // inlet velocity profile

extern T_P* u, *v, *w, *rho;
extern T_P* pdf;

extern T_P rt1, rti1, la_nu1, la_nui1;		 // viscosity for singlephase fluid
extern T_P force_x, force_y, force_z, D_force_z, force_z0;   // body force, default flow direction : z
extern T_P uin_max, uin_avg, uin_avg_0, flowrate, p_gradient, rho_out, rho_in, rho_in_avg, rho_avg_inlet, rho_avg_outlet, rho_drop;
extern T_P relaxation, uin_avg_convec, p_max, rho_in_max, umax_global;
// temporary arrays
extern T_P* fl, *pre; // flowrate, pressure, saturation vs z

/* GPU pointers */
extern T_P* W_in_d, * u_d, * v_d, * w_d, * rho_d;
extern T_P* pdf_d;
extern T_P* fl_d, * pre_d;
#endif