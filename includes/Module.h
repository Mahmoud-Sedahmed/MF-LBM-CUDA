#ifndef MODULE_H
#define MODULE_H
#include "preprocessor.h"
#include "externLib.h"
#include "solver_precision.h"
#include "Module_extern.h"
#include <limits>

T_P Pi = prc(3.14159265358979323846);
#if(PRECISION == SINGLE_PRECISION)
T_P eps = numeric_limits<float>::epsilon();
//T_P eps = prc(1.110223025e-8);
#elif (PRECISION == DOUBLE_PRECISION)
//T_P eps = prc(1.110223025e-16);
T_P eps = numeric_limits<float>::epsilon();
#endif

T_P convergence_criteria;

string job_status;
string geo_file_path, geo_boundary_file_path;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~input commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 1 - simulation ends; 0 - initial value(exceed max time step); 2 - simulation not finished, but save dataand exit program; 3 - simulation failed
int simulation_end_indicator;

// Modify geometry command(not modify 1; modify 2)
int modify_geometry_cmd;

int initial_fluid_distribution_option;

// steady state option : 0 - unsteady state simulation; 1 - steady state simulation based on capillary pressure;
// 2 - steady state simulation based on phase field; 3 - steady state simulation based on fractional flowrate
int steady_state_option;

// read external geometry cmd : 0 - no; 1 - yes, default name: .. / .. / walldata / walls.dat
int external_geometry_read_cmd;

// geometry preprocess cmd : 0 - process the geometry during the simulation; 1 - load external preprocessed geometry data
int geometry_preprocess_cmd;

// benchmarking simulation = 1; regular simulation = 0
int benchmark_cmd;

// double back up checkpoint data : 0 - no; 1 - yes(increase storage space)
int double_bak_checkpoint_pdf_cmd;

// place porous plate in the domain(usually near inlet or outlet) : 0 - nothing; 1 - block fluid 1; 2 - block fluid 2
int porous_plate_cmd;

// Change inlet fluid phase before initial_interface_position : 0 - nothing; 1 - fluid1; 2 - fluid 2
int change_inlet_fluid_phase_cmd;

// necesssary modifications for extreme large simulations including number limitand I / O related issues : 0 - no; 1 - yes
int extreme_large_sim_cmd;

// output field data precision(simulation is always double precision) : 0 - single precision; 1 - double precision
int output_fieldData_precision_cmd;
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~input commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~domain and geometry ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
T_P la_x, la_y, la_z;                                             // effective simulation domain dimension, in lattice units
long long nxGlobal, nyGlobal, nzGlobal;                                   // full grid : 1 to n ? Global

int iper, jper, kper;                                                 //periodic BC indicator

// domain_wall_status of the domain boundaries : 1 - solid wall; 0 - fluid
int domain_wall_status_x_min, domain_wall_status_x_max;
int domain_wall_status_y_min, domain_wall_status_y_max;
int domain_wall_status_z_min, domain_wall_status_z_max;

long long nx_sample, ny_sample, nz_sample;     // rock sample size

int n_exclude_inlet, n_exclude_outlet;


T_P A_xy, A_xy_effective, volume_sample;   // Axy: cross section area

int inlet_BC, outlet_BC;
T_P target_inject_pore_volume;

long long n_fluid_node_local;  // total fluid nodes of local domain

long long num_solid_boundary_global, num_solid_boundary;     // number of solid boundary nodes
long long num_fluid_boundary_global, num_fluid_boundary;      //number of fluid boundary nodes

int* pore_profile_z;                                    // used in IO for easier data process.

long long pore_sum, pore_sum_effective;                        // effeictive pore sum excludes inletand outlet portion
T_P porosity_full, porosity_effective;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~lattice ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// streaming direction - D3Q19
int ex[19] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0 };
int ey[19] = { 0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1 };
int ez[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1 };
int opc[19] = { 0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15 }; // oppsite directions

// MRT
T_P mrt_coef1 = prc(1.) / prc(19.);
T_P mrt_coef2 = prc(1.) / prc(2394.);
T_P mrt_coef3 = prc(1.) / prc(252.);
T_P mrt_coef4 = prc(1.) / prc(72.);
T_P mrt_e2_coef1 = prc(0.);
T_P mrt_e2_coef2 = prc(-475.) / prc(63.);
T_P mrt_omega_xx = prc(0.);


// D3Q19 MODEL
T_P w_equ[19] = {
    prc(1.) / prc(3.),
    prc(1.) / prc(18.), prc(1.) / prc(18.), prc(1.) / prc(18.), prc(1.) / prc(18.), prc(1.) / prc(18.), prc(1.) / prc(18.),
    prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.),
    prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.), prc(1.) / prc(36.)
};
T_P w_equ_0 = prc(1.) / prc(3.);
T_P w_equ_1 = prc(1.) / prc(18.);
T_P w_equ_2 = prc(1.) / prc(36.);
T_P la_vel_norm_i = prc(1.) / prc(sqrt)(prc(2.)); // account for the diagonal length

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~unsorted ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// timers
int ntime0, ntime, ntime_max, ntime_max_benchmark, ntime_macro, ntime_visual, ntime_pdf, ntime_relaxation, ntime_animation;
int ntime_clock_sum, ntime_display_steps, ntime_monitor, ntime_monitor_profile, ntime_monitor_profile_ratio;
int num_slice;   // how many monitoring slices(evenly divided through y axis)
T_P d_sa_vol, sa_vol, d_vol_animation, d_vol_detail, d_vol_monitor, d_vol_monitor_prof;
T_P checkpoint_save_timer, checkpoint_2rd_save_timer, simulation_duration_timer;
int wallclock_timer_status_sum;   // wall clock time reach desired time for each MPI process
double wallclock_pdfsave_timer;  // save PDF data for restart simulation based on wall clock timer

// monitors
T_P monitor_previous_value, monitor_current_value;
//const int time_pt_max = 10;      // maximum 10 points in the history
class monitor_time_pt {
public:
    T_P v1, v2, v3, it;  // ntime / ntime_monitor
};

monitor_time_pt monitor_pt[time_pt_max];
int n_current_monitor;  // current index in the monitor_time_pt array


/* M.Sedahmed - memory increment */
int xlen, ylen, zlen; // used with walls
int fm_x, fm_y, fm_z; // used with flow related arrays
int cm_x, cm_y, cm_z; // used with cn,c_norm
int phim_x, phim_y, phim_z; // used with phi

double memory_gpu; // total amount of memory needed on GPU

int rho_in_new = 0;
int rho_out_BC = 0;

char empty_char[6] = "empty";

int* walls, * walls_global, * walls_type;
//extern int* walls_ghost;

/* domain and memory sizes */
long long num_cells_s0, num_cells_s1, num_cells_s2, num_cells_s4;
long long num_cells_v1, num_cells_v2;
long long num_cells_f1;

long long mem_size_s0_TP, mem_size_s1_TP, mem_size_s2_TP, mem_size_s4_TP, mem_size_s0_int, mem_size_s2_int, mem_size_s4_int;
long long mem_size_v1_TP, mem_size_v2_TP;
long long mem_size_f1_TP;

// solid normal vector
T_P* s_nx, *s_ny, *s_nz;
// monitor parameter
T_P* tk;


/* GPU pointers */
int* walls_d, * walls_type_d;
T_P* tk_d;
T_P* s_nx_d, * s_ny_d, * s_nz_d;

/* CUDA threads in a block */
int block_Threads_X, block_Threads_Y, block_Threads_Z;
#endif