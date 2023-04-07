#ifndef MODULE_EXTERN_H
#define MODULE_EXTERN_H

#include "preprocessor.h"
#include "externLib.h"
#include "solver_precision.h"

extern T_P Pi;
extern T_P eps;
extern T_P convergence_criteria;

// job status : new_simulation; continue_simulation; simulation_done; simulation_failed
extern string job_status;
extern string geo_file_path, geo_boundary_file_path;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~input commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 1 - simulation ends; 0 - initial value(exceed max time step); 2 - simulation not finished, but save dataand exit program; 3 - simulation failed
extern int simulation_end_indicator;

// Modify geometry command(not modify 1; modify 2)
extern int modify_geometry_cmd;

extern int initial_fluid_distribution_option;

// steady state option : 0 - unsteady state simulation; 1 - steady state simulation based on capillary pressure;
// 2 - steady state simulation based on phase field; 3 - steady state simulation based on fractional flowrate
extern int steady_state_option;

// read external geometry cmd : 0 - no; 1 - yes, default name: .. / .. / walldata / walls.dat
extern int external_geometry_read_cmd;

// geometry preprocess cmd : 0 - process the geometry during the simulation; 1 - load external preprocessed geometry data
extern int geometry_preprocess_cmd;

// choose type size of dimension variables saved in the geometry binary file : 0 - 4 bytes(int, uint, ..etc.); 1 - 8 bytes(long long, unint64, ...etc.)
extern int geometry_dims_type_size;

// benchmarking simulation = 1; regular simulation = 0
extern int benchmark_cmd;

// double back up checkpoint data : 0 - no; 1 - yes(increase storage space)
extern int double_bak_checkpoint_pdf_cmd;

// place porous plate in the domain(usually near inlet or outlet) : 0 - nothing; 1 - block fluid 1; 2 - block fluid 2
extern int porous_plate_cmd;

// Change inlet fluid phase before initial_interface_position : 0 - nothing; 1 - fluid1; 2 - fluid 2
extern int change_inlet_fluid_phase_cmd;

// necesssary modifications for extreme large simulations including number limitand I / O related issues : 0 - no; 1 - yes
extern int extreme_large_sim_cmd;

// output field data precision(simulation is always double precision) : 0 - single precision; 1 - double precision
extern int output_fieldData_precision_cmd;
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~input commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~domain and geometry ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
extern T_P la_x, la_y, la_z;                                             // effective simulation domain dimension, in lattice units
extern long long nxGlobal, nyGlobal, nzGlobal;                                   // full grid : 1 to n ? Global

extern int iper, jper, kper;                                                 //periodic BC indicator

// domain_wall_status of the domain boundaries : 1 - solid wall; 0 - fluid
extern int domain_wall_status_x_min, domain_wall_status_x_max;
extern int domain_wall_status_y_min, domain_wall_status_y_max;
extern int domain_wall_status_z_min, domain_wall_status_z_max;

extern long long nx_sample, ny_sample, nz_sample;     // rock sample size

extern int n_exclude_inlet, n_exclude_outlet;

extern T_P A_xy, A_xy_effective, volume_sample;   // Axy: cross section area

extern int inlet_BC, outlet_BC;
extern T_P target_inject_pore_volume;

extern long long n_fluid_node_local;  // total fluid nodes of local domain

extern long long num_solid_boundary_global, num_solid_boundary;     // number of solid boundary nodes
extern long long num_fluid_boundary_global, num_fluid_boundary;      //number of fluid boundary nodes

extern int* pore_profile_z;                                    // used in IO for easier data process.
extern long long pore_sum, pore_sum_effective;                        // effeictive pore sum excludes inletand outlet portion
extern T_P porosity_full, porosity_effective;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~lattice ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// streaming direction - D3Q19
extern int ex[19];
extern int ey[19];
extern int ez[19];
extern int opc[19]; // oppsite directions

// MRT
extern T_P mrt_coef1;
extern T_P mrt_coef2;
extern T_P mrt_coef3;
extern T_P mrt_coef4;
extern T_P mrt_e2_coef1;
extern T_P mrt_e2_coef2;
extern T_P mrt_omega_xx;

// D3Q19 MODEL
extern T_P w_equ[19];
extern T_P w_equ_0;
extern T_P w_equ_1;
extern T_P w_equ_2;
extern T_P la_vel_norm_i; // account for the diagonal length

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~unsorted ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// timers
extern int ntime0, ntime, ntime_max, ntime_max_benchmark, ntime_macro, ntime_visual, ntime_pdf, ntime_relaxation, ntime_animation;
extern int ntime_clock_sum, ntime_display_steps, ntime_monitor, ntime_monitor_profile, ntime_monitor_profile_ratio;
extern int num_slice;   // how many monitoring slices(evenly divided through y axis)
extern T_P d_sa_vol, sa_vol, d_vol_animation, d_vol_detail, d_vol_monitor, d_vol_monitor_prof;
extern T_P checkpoint_save_timer, checkpoint_2rd_save_timer, simulation_duration_timer;
extern int wallclock_timer_status_sum;   
extern double wallclock_pdfsave_timer;  // save PDF data for restart simulation based on wall clock timer

// monitors
extern T_P monitor_previous_value, monitor_current_value;
const int time_pt_max = 10;      // maximum 10 points in the history
class monitor_time_pt;

// extern monitor_time_pt monitor_pt[time_pt_max];
extern int n_current_monitor;  // current index in the monitor_time_pt array

/* M.Sedahmed - memory increment */
extern int xlen, ylen, zlen; // used with walls
extern int fm_x, fm_y, fm_z; // used with flow related arrays
extern int cm_x, cm_y, cm_z; // used with cn,c_norm
extern int phim_x, phim_y, phim_z; // used with phi

extern double memory_gpu; // total amount of memory needed on GPU

extern int rho_in_new;
extern int rho_out_BC;

extern char empty_char[6];

extern int* walls, * walls_global, * walls_type;
//extern int* walls_ghost;

/* domain and memory sizes */
extern long long num_cells_s0, num_cells_s1, num_cells_s2, num_cells_s4;
extern long long num_cells_v1, num_cells_v2;
extern long long num_cells_f1;

extern long long mem_size_s0_TP, mem_size_s1_TP, mem_size_s2_TP, mem_size_s4_TP, mem_size_s0_int, mem_size_s2_int, mem_size_s4_int;
extern long long mem_size_v1_TP, mem_size_v2_TP;
extern long long mem_size_f1_TP;

// solid normal vector
extern T_P* s_nx, *s_ny, *s_nz;
// monitor parameter
extern T_P* tk;


/* GPU pointers */
extern int* walls_d, * walls_type_d;
extern T_P* tk_d;
extern T_P* s_nx_d, * s_ny_d, * s_nz_d;

/* CUDA threads in a block */
extern int block_Threads_X, block_Threads_Y, block_Threads_Z;
#endif


