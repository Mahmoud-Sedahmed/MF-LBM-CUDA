#ifndef Misc_H
#define Misc_H

/* geometry related */
/* set walls */
void set_walls();

/* modify geometry */
void modify_geometry();

// *****************************read walls************************************
void read_walls();

/* pore profile */
void pore_profile();

/* compute macroscopic varaibles from PDFs */
void compute_macro_vars();

/* inlet velocity - analytical solution */
void inlet_vel_profile_rectangular(T_P vel_avg, int num_terms);

/* change inlet fluid phase */
void change_inlet_fluid_phase();

#endif