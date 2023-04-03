#include "externLib.h"
#include "solver_precision.h"
#include "preprocessor.h"
#include "Module_extern.h"
#include "Fluid_singlephase_extern.h"
#include "Fluid_multiphase_extern.h"
#include "utils.h"
#include "Idx_cpu.h"
#include "Misc.h"
#include "IO_multiphase.h"


//=========================================================================================================================== =
//----------------------Read input parameters----------------------
//=========================================================================================================================== =
void read_parameter_multi() {
    cout << "Checking simulation status ... " << endl;
    string FILE_job_status = "input/job_status.txt";
    ifstream file_job_status(FILE_job_status.c_str());
    bool ALIVE = file_job_status.good();
    if (!ALIVE) {
        ERROR("Missing job status file! Exiting program!");
    }
    /* open the simulation status file */
    string job_status_s((istreambuf_iterator<char>(file_job_status)), istreambuf_iterator<char>());
    job_status = job_status_s;
    file_job_status.close();
    if (job_status == "new_simulation") {
        cout << "Simulation status is: New Simulation!" << endl;
    }
    else if (job_status == "continue_simulation") {
        cout << "Simulation status is: Continue existing simulation!" << endl;
    }
    else {
        ERROR("Wrong simlation status! Exiting program!");
    }
    /* open the simulation control file */
    string FILE_simulation_control = "input/simulation_control.txt";
    ifstream file_simulation_control(FILE_simulation_control.c_str());
    ALIVE = file_simulation_control.good();
    if (!ALIVE) {
        ERROR("Missing simulation control file! Exiting program!");
    }
    cout << " " << endl;
    cout << "****************** Reading in parameters from control file *******************" << endl;
    cout << "........." << endl;
    READ_INT(FILE_simulation_control.c_str(), initial_fluid_distribution_option);
    cout << "initial_fluid_distribution_option: " << initial_fluid_distribution_option << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), breakthrough_check); // breakthrough check: whether exiting the simulations when the invading phase reaches the outlet
    cout << "breakthrough_check:    " << breakthrough_check << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), steady_state_option); // 
    cout << "steady_state_option:   " << steady_state_option << endl;
    cout << "---------------------------" << endl;
    READ_T_P(FILE_simulation_control.c_str(), convergence_criteria); // 
    cout << "convergence_criteria:  " << convergence_criteria << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), benchmark_cmd); // 
    cout << "benchmark_cmd: " << benchmark_cmd << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), output_fieldData_precision_cmd); // 
    cout << "output_fieldData_precision_cmd:    " << output_fieldData_precision_cmd << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), extreme_large_sim_cmd); // 
    cout << "extreme_large_sim_cmd: " << extreme_large_sim_cmd << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), modify_geometry_cmd); // 
    cout << "modify_geometry_cmd:   " << modify_geometry_cmd << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), external_geometry_read_cmd); // 
    cout << "external_geometry_read_cmd:   " << external_geometry_read_cmd << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), geometry_preprocess_cmd); // 
    cout << "geometry_preprocess_cmd:   " << geometry_preprocess_cmd << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), porous_plate_cmd); // 
    cout << "porous_plate_cmd:   " << porous_plate_cmd << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), change_inlet_fluid_phase_cmd); // 
    cout << "change_inlet_fluid_phase_cmd:   " << change_inlet_fluid_phase_cmd << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), nxGlobal); //  total nodes for the whole domain
    cout << "nxGlobal:   " << nxGlobal << endl;
    READ_INT(FILE_simulation_control.c_str(), nyGlobal);
    cout << "nyGlobal:   " << nyGlobal << endl;
    READ_INT(FILE_simulation_control.c_str(), nzGlobal);
    cout << "nzGlobal:   " << nzGlobal << endl;
    READ_INT(FILE_simulation_control.c_str(), n_exclude_inlet); // excluded_layers
    cout << "Inlet_excluded_layers:   " << n_exclude_inlet << endl;
    READ_INT(FILE_simulation_control.c_str(), n_exclude_outlet);
    cout << "Outlet_excluded_layers:   " << n_exclude_outlet << endl;
    READ_INT(FILE_simulation_control.c_str(), domain_wall_status_x_min); // domain_wall_status
    cout << "Apply nonslip boundary condition at x = 1:   " << domain_wall_status_x_min << endl;
    READ_INT(FILE_simulation_control.c_str(), domain_wall_status_x_max);
    cout << "Apply nonslip boundary condition at x = nxGlobal:   " << domain_wall_status_x_max << endl;
    READ_INT(FILE_simulation_control.c_str(), domain_wall_status_y_min); // domain_wall_status
    cout << "Apply nonslip boundary condition at y = 1:   " << domain_wall_status_y_min << endl;
    READ_INT(FILE_simulation_control.c_str(), domain_wall_status_y_max);
    cout << "Apply nonslip boundary condition at y = nyGlobal:   " << domain_wall_status_y_max << endl;
    READ_INT(FILE_simulation_control.c_str(), domain_wall_status_z_min); // domain_wall_status
    cout << "Apply nonslip boundary condition at z = 1:   " << domain_wall_status_z_min << endl;
    READ_INT(FILE_simulation_control.c_str(), domain_wall_status_z_max);
    cout << "Apply nonslip boundary condition at z = nzGlobal:   " << domain_wall_status_z_max << endl;
    READ_INT(FILE_simulation_control.c_str(), iper); // periodic BC indicator, 1 periodic, 0 non-periodic
    cout << "X_periodic_option =   " << iper << endl;
    READ_INT(FILE_simulation_control.c_str(), jper);
    cout << "Y_periodic_option =   " << jper << endl;
    READ_INT(FILE_simulation_control.c_str(), kper);
    cout << "Z_periodic_option =   " << kper << endl;
    cout << "---------------------------" << endl;
    T_P fluid1_viscosity; READ_T_P(FILE_simulation_control.c_str(), fluid1_viscosity); la_nu1 = fluid1_viscosity; // fluid1_viscosity
    cout << "fluid1_viscosity:   " << la_nu1 << endl;
    T_P fluid2_viscosity; READ_T_P(FILE_simulation_control.c_str(), fluid2_viscosity); la_nu2 = fluid2_viscosity; // fluid2_viscosity
    cout << "fluid2_viscosity:   " << la_nu2 << endl;
    T_P surface_tension; READ_T_P(FILE_simulation_control.c_str(), surface_tension); lbm_gamma = surface_tension; // surface_tension
    cout << "surface_tension:   " << lbm_gamma << endl;
    READ_T_P(FILE_simulation_control.c_str(), theta); // contact angle
    cout << "Contact_angle:   " << theta << endl;
    T_P RK_beta; READ_T_P(FILE_simulation_control.c_str(), RK_beta); lbm_beta = RK_beta; // beta in RK recolor scheme
    cout << "RK_beta:   " << lbm_beta << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), inlet_BC); // inlet_BC
    cout << "inlet_BC =   " << inlet_BC << endl;
    READ_INT(FILE_simulation_control.c_str(), outlet_BC); // outlet_BC
    cout << "outlet_BC =   " << outlet_BC << endl;
    READ_T_P(FILE_simulation_control.c_str(), target_inject_pore_volume); // target_inject_pore_volume
    cout << "target_inject_pore_volume =   " << target_inject_pore_volume << endl;
    T_P capillary_number; READ_T_P(FILE_simulation_control.c_str(), capillary_number); ca_0 = capillary_number; // capillary_number
    cout << "capillary_number =   " << ca_0 << endl;
    T_P saturation_injection; READ_T_P(FILE_simulation_control.c_str(), saturation_injection); sa_inject = saturation_injection;
    cout << "Injecting_fluid_saturation =   " << sa_inject << endl;  // injecting fluid saturation (1: pure fluid1 injection; 0: pure fluid2 injection
    T_P initial_interface_position; READ_T_P(FILE_simulation_control.c_str(), initial_interface_position); interface_z0 = initial_interface_position;
    cout << "initial_interface_position =   " << interface_z0 << endl; // initial interface position alng z axis(multiphase)
    T_P body_force_0; READ_T_P(FILE_simulation_control.c_str(), body_force_0); force_z0 = body_force_0; // initial value of body force Z or pressure gradient
    cout << "Body_force_Z0 =   " << force_z0 << endl;
    T_P target_fluid1_saturation; READ_T_P(FILE_simulation_control.c_str(), target_fluid1_saturation); sa_target = target_fluid1_saturation;
    cout << "target_fluid1_saturation =   " << sa_target << endl;  // target fluid1 saturation (steady state relative permeability)
    READ_INT(FILE_simulation_control.c_str(), Z_porous_plate); // porous plate location 
    cout << "Z_porous_plate =   " << Z_porous_plate << endl;
    cout << "---------------------------" << endl;
    int max_time_step; READ_INT(FILE_simulation_control.c_str(), max_time_step); ntime_max = max_time_step; // timer: max iterations
    cout << "Max_iterations =   " << ntime_max << endl;
    int max_time_step_benchmark; READ_INT(FILE_simulation_control.c_str(), max_time_step_benchmark); ntime_max_benchmark = max_time_step_benchmark;
    cout << "Max_iterations_benchmark =   " << ntime_max_benchmark << endl; // timer: max iterations for the benchmark case
    READ_INT(FILE_simulation_control.c_str(), ntime_visual); // timer: when to output detailed visualization data
    cout << "Full_VTK_timer =   " << ntime_visual << endl;
    READ_INT(FILE_simulation_control.c_str(), ntime_animation); // timer: when to output animation data, only phi and wall
    cout << "Animation_VTK_timer =   " << ntime_animation << endl;
    int monitor_timer; READ_INT(FILE_simulation_control.c_str(), monitor_timer); ntime_monitor = monitor_timer;
    cout << "Monitor_timer =   " << ntime_monitor << endl; // timer: when to output monitor data, profile along flow direction
    int monitor_profile_timer_ratio; READ_INT(FILE_simulation_control.c_str(), monitor_profile_timer_ratio); ntime_monitor_profile_ratio = monitor_profile_timer_ratio;
    cout << "Monitor_profile_timer_ratio =   " << ntime_monitor_profile_ratio << endl;
    int computation_time_timer; READ_INT(FILE_simulation_control.c_str(), computation_time_timer); ntime_clock_sum = computation_time_timer;
    cout << "Computation_time_timer =   " << ntime_clock_sum << endl; // timer: gaps used to record computation time
    int display_steps_timer; READ_INT(FILE_simulation_control.c_str(), display_steps_timer); ntime_display_steps = display_steps_timer;
    cout << "Display_time_steps_timer =   " << ntime_display_steps << endl; // timer: when to display time steps
    cout << "---------------------------" << endl;
    READ_T_P(FILE_simulation_control.c_str(), checkpoint_save_timer); // save PDF data for restart simulation based on wall clock timer  (unit hours)
    cout << "Checkpoint_save_timer (wall clock time, hours) =   " << checkpoint_save_timer << endl;
    READ_T_P(FILE_simulation_control.c_str(), checkpoint_2rd_save_timer); //  save secondary PDF data for restart simulation based on wall clock timer  (unit hours)
    cout << "Checkpoint_2rd_save_timer (wall clock time, hours) =   " << checkpoint_2rd_save_timer << endl;
    READ_T_P(FILE_simulation_control.c_str(), simulation_duration_timer); // simulation duration in hours, exit and save simulation afterwards
    cout << "Simulation_duration (wall clock time, hours) =   " << simulation_duration_timer << endl;
    cout << "---------------------------" << endl;
    READ_T_P(FILE_simulation_control.c_str(), d_vol_animation); // when to output animation data - based on injected volume
    cout << "Animation_VTK_by_injected_vol =   " << d_vol_animation << endl;
    READ_T_P(FILE_simulation_control.c_str(), d_vol_detail); // when to output detailed visulization data - based on injected volume
    cout << "Full_VTK_by_injected_vol =   " << d_vol_detail << endl;
    READ_T_P(FILE_simulation_control.c_str(), d_vol_monitor); // when to output bulk property data - based on injected volume
    cout << "Monitor_by_injected_vol =   " << d_vol_monitor << endl;
    cout << "---------------------------" << endl;
    READ_INT(FILE_simulation_control.c_str(), rho_in_new);
    cout << "new_inlet_density: " << rho_in_new << endl;
    READ_INT(FILE_simulation_control.c_str(), rho_out_BC);
    if (rho_out_BC) { cout << "Assign outlet density value!" << endl; }
    cout << "---------------------------" << endl;

    READ_INT(FILE_simulation_control.c_str(), block_Threads_X);
    cout << "CUDA block_Threads_X: " << block_Threads_X << endl;
    READ_INT(FILE_simulation_control.c_str(), block_Threads_Y);
    cout << "CUDA block_Threads_Y: " << block_Threads_Y << endl;
    READ_INT(FILE_simulation_control.c_str(), block_Threads_Z);
    cout << "CUDA block_Threads_Z: " << block_Threads_Z << endl;

    
    file_simulation_control.close();
    cout << "---------------------------" << endl;
    ntime_monitor_profile = ntime_monitor_profile_ratio * ntime_monitor;
    cout << "monitor_profile_timer =     " << ntime_monitor_profile << endl;
    cout << "************** End reading in parameters from control file *******************" << endl;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ check correctness of input parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cout << "************ Start checking correctness of input parameters ******************" << endl;
    int error_signal = 0;

    if (theta > prc(90.)) {
        cout << "Error: contact angle is larger than 90 degrees! Please refer to the notes on 'simlation_control' file! Exiting program!" << endl;
        error_signal = 1;
    }
    theta = prc(180.) - theta;       // measured through defending phase
    theta = theta * Pi / prc(180.);  //contact angle
    cos_theta = prc(cos)(theta); // assign contact angle

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Periodic parameters
    if (iper == 1 || domain_wall_status_x_max == 0 || domain_wall_status_x_min == 0) {
        cout << "Error: X direction periodic BC enabled or non-slip BC not applied at x = xmin or x = xmax! Exiting program!" << endl;
        error_signal = 1;
    }
    if (jper == 0 && (domain_wall_status_y_max == 0 || domain_wall_status_y_min == 0)) {
        cout << "Error: non-slip BC not applied at y = ymin or y = ymax while y direction periodic BC not enabled! Exiting program!" << endl;
        error_signal = 1;
    }
    if (jper == 1 && (domain_wall_status_y_max == 1 || domain_wall_status_y_min == 1)) {
        cout << "Error: non-slip BC applied at y = ymin or y = ymax while y direction periodic BC enabled! Exiting program!" << endl;
        error_signal = 1;
    }
    if (kper == 1 && (domain_wall_status_z_max == 1 || domain_wall_status_z_min == 1)) {
        cout << "Error: non-slip BC applied at z = zmin or z = zmax while z direction periodic BC enabled! Exiting program!" << endl;
        error_signal = 1;
    }


    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Boundary conditions
    if (outlet_BC == 1 && inlet_BC == 2) {
        cout << "Inlet/outlet boundary condition error: Inlet pressure + outlet convective BC is not supported! Exiting program!" << endl;
        error_signal = 1;
    }

    if (error_signal == 1) {
        ERROR("Exit Program!");
    }
    else {
        cout << "Everything looks good!" << endl;
        cout << "************** End checking correctness of input parameters ******************" << endl;
        cout << endl;
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ check correctness of input parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


}

//=========================================================================================================================== =
//----------------------save data ----------------------
//=========================================================================================================================== = 
// ******************************* save checkpoint data *************************************
void save_checkpoint(int save_option) { // ! option 0 - default location; option 1 - secondary location
     /* File name */
    ostringstream flnm;
    flnm << "id" << setfill('0') << setw(4) << 0; // This line makes the width (4) and fills the values other than the required with (0)
    ostringstream filepath;
    if (save_option == 0) {
        filepath << "results/out2.checkpoint/" << flnm.str();
    }
    else {
        filepath << "results/out2.checkpoint/2rd_backup/" << flnm.str();
    }

    string fns = filepath.str();
    const char* fnc = fns.c_str();
    FILE* check_point_file = fopen(fnc, "wb+"); // open the file for the first time (create the file) 
    if (check_point_file == NULL) { ERROR("Could not create save_checkpoint file!"); }

    int nt_temp = ntime + 1;
    fwrite(&nt_temp, sizeof(int), 1, check_point_file);

    fwrite(&force_z, sizeof(T_P), 1, check_point_file);
    fwrite(&rho_in, sizeof(T_P), 1, check_point_file);

    fwrite(pdf, mem_size_f1_TP, 1, check_point_file);

    fwrite(phi, mem_size_s4_TP, 1, check_point_file);

    
    if (outlet_BC == 1) { // convective BC
        long long mem_size_tmp = NXG1 * NYG1 * 19 * sizeof(T_P); 
        fwrite(f_convec_bc, mem_size_tmp, 1, check_point_file);
        fwrite(g_convec_bc, mem_size_tmp, 1, check_point_file);
        mem_size_tmp = NXG1 * NYG1 * sizeof(T_P);
        fwrite(phi_convec_bc, mem_size_tmp, 1, check_point_file);
    }
    fclose(check_point_file);

    if (save_option == 0) {
        cout << "Saving checkpoint data completed!" << endl;
        string filepath = "./job_status.txt";
        ofstream job_stat(filepath.c_str(), ios_base::out);
        if (job_stat.good()) {
            job_stat << "continue_simulation" << endl;
            job_stat.close();
        }
        else {
            ERROR("Could not open ./job_status.txt");
        }
    }
    if (save_option == 1) {
        cout << "Saving secondary checkpoint data completed!" << endl;
    }

}

// ******************************* save data - phi *************************************
void save_phi(int nt) {

    int i, j, k;
    /* File name */
    ostringstream flnm;
    flnm << "phi_nt" << setfill('0') << setw(9) << nt << "_id" << setfill('0') << setw(5) << 0;

    cout << "Start to save phase field data." << endl;

    ostringstream filepath;
    filepath << "results/out3.field_data/phase_distribution/" << flnm.str();
    string fns = filepath.str();
    const char* fnc = fns.c_str();
    FILE* check_point_file = fopen(fnc, "wb+"); // open the file for the first time (create the file)
    if (check_point_file == NULL) { ERROR("Could not create save_checkpoint file!"); }

    fwrite(&nxGlobal, sizeof(int), 1, check_point_file);
    fwrite(&nyGlobal, sizeof(int), 1, check_point_file);
    fwrite(&nzGlobal, sizeof(int), 1, check_point_file);

    if (output_fieldData_precision_cmd == 0) { // single precision
        float* phi_p = NULL;
        long long size_phi = nxGlobal * nyGlobal * nzGlobal;
        phi_p = new float[size_phi];
        //float* phi_p = (float*)calloc(nx * ny * nz, sizeof(float));
        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    phi_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = float(phi[i_s4(i, j, k)]);
                }
            }
        }
        long long mem_size_tmp = size_phi * sizeof(float);
        fwrite(phi_p, mem_size_tmp, 1, check_point_file);
        delete[] phi_p;
    }
    else { // double precision
        double* phi_p = NULL;
        long long size_phi = nxGlobal * nyGlobal * nzGlobal;
        phi_p = new double[size_phi];
        //phi_p = (T_P*)calloc(nx * ny * nz, sizeof(T_P));
        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    phi_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = double(phi[i_s4(i, j, k)]);
                }
            }
        }
        long long mem_size_tmp = size_phi * sizeof(double);
        fwrite(phi_p, mem_size_tmp, 1, check_point_file);
        delete[] phi_p;
    }

    fclose(check_point_file);
    cout << "Phase field data saved!" << endl;
}

// ******************************* save data - macro variables *************************************
void save_macro(int nt) {
    int i, j, k;

    compute_macro_vars();

    ostringstream flnm;
    flnm << "full_nt" << setfill('0') << setw(9) << nt << "_id" << setfill('0') << setw(5) << 0;

    cout << "Start to save full flow field data." << endl;

    ostringstream filepath;
    filepath << "results/out3.field_data/full_flow_field/" << flnm.str();
    string fns = filepath.str();
    const char* fnc = fns.c_str();
    FILE* check_point_file = fopen(fnc, "wb+"); // open the file for the first time (create the file)
    if (check_point_file == NULL) { ERROR("Could not create save_checkpoint file!"); }

    fwrite(&nxGlobal, sizeof(int), 1, check_point_file);
    fwrite(&nyGlobal, sizeof(int), 1, check_point_file);
    fwrite(&nzGlobal, sizeof(int), 1, check_point_file);


    if (output_fieldData_precision_cmd == 0) { // single precision
        float* buff_p = (float*)calloc(nxGlobal * nyGlobal * nzGlobal, sizeof(float));
        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = float(u[i_s1(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(float), 1, check_point_file);

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = float(v[i_s1(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(float), 1, check_point_file);

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = float(w[i_s1(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(float), 1, check_point_file);

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = float(rho[i_s1(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(float), 1, check_point_file);

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = float(phi[i_s4(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(float), 1, check_point_file);
    }
    else { // double precision
        T_P* buff_p = (T_P*)calloc(nxGlobal * nyGlobal * nzGlobal, sizeof(T_P));
        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = T_P(u[i_s1(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(T_P), 1, check_point_file);

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = T_P(v[i_s1(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(T_P), 1, check_point_file);

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = T_P(w[i_s1(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(T_P), 1, check_point_file);

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = T_P(rho[i_s1(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(T_P), 1, check_point_file);

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    buff_p[(i - 1) + nxGlobal * ((j - 1) + nyGlobal * (k - 1))] = T_P(phi[i_s4(i, j, k)]);
                }
            }
        }
        fwrite(buff_p, nxGlobal * nyGlobal * nzGlobal * sizeof(T_P), 1, check_point_file);
    }

    fclose(check_point_file);
    cout << "Full flow field data saved!" << endl;
}

//===================================================================================================================================== =
//----------------------save data - vtk----------------------
//===================================================================================================================================== =
/* swap byte order (Little endian > Big endian) */
template <typename T>
void SwapEnd(T& var)
{
    char* varArray = reinterpret_cast<char*>(&var);
    for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
        std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}

//***********************3D legacy VTK writer * **********************
void VTK_legacy_writer_3D(int nt, int vtk_type) {

    string fmt;

    int i, j, k, l2, m2, n2, wall_indicator;

    T_P* dd = nullptr, * ff = nullptr, * utt = nullptr, * vtt = nullptr, * wtt = nullptr;
    float* ff4 = nullptr;   // single precision

    l2 = nxGlobal;
    m2 = nyGlobal;
    n2 = nzGlobal;

    // vtk_type 1: full flow field info for detailed analysis with options to save single / double precision data
    // vtk_type 2 : phase field info, single precision only
    // vtk_type 3 : force vectors from the CSF model with options to save single / double precision data

    if (vtk_type == 1 || vtk_type == 3) {
        dd = (T_P*)calloc(l2 * m2 * n2, sizeof(T_P));
        ff = (T_P*)calloc(l2 * m2 * n2, sizeof(T_P));
        utt = (T_P*)calloc(l2 * m2 * n2, sizeof(T_P));
        vtt = (T_P*)calloc(l2 * m2 * n2, sizeof(T_P));
        wtt = (T_P*)calloc(l2 * m2 * n2, sizeof(T_P));


        if (vtk_type == 1) {
            compute_macro_vars();

            for (k = 1; k <= nzGlobal; k++) {
                for (j = 1; j <= nyGlobal; j++) {
                    for (i = 1; i <= nxGlobal; i++) {
                        utt[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = u[i_s1(i, j, k)];
                        vtt[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = v[i_s1(i, j, k)];
                        wtt[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = w[i_s1(i, j, k)];
                        dd[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = rho[i_s1(i, j, k)];
                        ff[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = phi[i_s4(i, j, k)];
                    }
                }
            }

        }
        else {


            for (k = 1; k <= nzGlobal; k++) {
                for (j = 1; j <= nyGlobal; j++) {
                    for (i = 1; i <= nxGlobal; i++) {
                        utt[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = cn_x[i_s2(i, j, k)];
                        vtt[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = cn_y[i_s2(i, j, k)];
                        wtt[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = cn_z[i_s2(i, j, k)];
                        dd[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = c_norm[i_s2(i, j, k)];
                        ff[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = phi[i_s4(i, j, k)];
                    }
                }
            }
        }

    }
    else if (vtk_type == 2) { // phase field info, with single precision to save space
        ff4 = (float*)calloc(l2 * m2 * n2, sizeof(float));

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    wall_indicator = walls[i_s2(i, j, k)];
                    phi[i_s4(i, j, k)] = prc(0.) * wall_indicator + phi[i_s4(i, j, k)] * (1 - wall_indicator);
                }
            }
        }

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    ff4[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = float(phi[i_s4(i, j, k)]);
                }
            }
        }

    }

    ostringstream flnm;
    flnm << setfill('0') << setw(10) << nt << ".vtk";
    ostringstream filepath;
    string file_str;
    ofstream vtk_out_1;
    if (vtk_type == 1) {
        filepath << "results/out3.field_data/full_flow_field/full_flow_field_" << flnm.str();
        file_str = filepath.str();
        vtk_out_1.open(file_str.c_str(), ios_base::out | std::ios::binary);
    }
    else if (vtk_type == 2) {
        filepath << "results/out3.field_data/phase_distribution/small_" << flnm.str();
        file_str = filepath.str();
        vtk_out_1.open(file_str.c_str(), ios_base::out | std::ios::binary);
    }
    else if (vtk_type == 3) {
        filepath << "results/out3.field_data/full_flow_field/force_vector_" << flnm.str();
        file_str = filepath.str();
        vtk_out_1.open(file_str.c_str(), ios_base::out | std::ios::binary);
    }

    if (vtk_out_1.good()) {
        vtk_out_1 << "# vtk DataFile Version 3.0" << endl;
        vtk_out_1 << "vtk output" << endl;
        vtk_out_1 << "BINARY" << endl;
        vtk_out_1 << "DATASET STRUCTURED_POINTS" << endl;
        vtk_out_1 << "DIMENSIONS " << nxGlobal << " " << nyGlobal << " " << nzGlobal << endl;
        vtk_out_1 << "ORIGIN " << 1 << " " << 1 << " " << 1 << endl;
        vtk_out_1 << "SPACING " << 1 << " " << 1 << " " << 1 << endl;
        vtk_out_1 << "POINT_DATA " << nxGlobal * nyGlobal * nzGlobal << endl;

        if (vtk_type == 1 || vtk_type == 3) {
            if (output_fieldData_precision_cmd == 0) {
                fmt = "float";
            }
            else {
                fmt = "double";
            }
            vtk_out_1 << "SCALARS " << "phi " << fmt << endl;
            vtk_out_1 << "LOOKUP_TABLE" << " default" << endl;

            for (k = 0; k < nzGlobal; k++) {
                for (j = 0; j < nyGlobal; j++) {
                    for (i = 0; i < nxGlobal; i++) {
                        T_P save_value = ff[i + l2 * (j + m2 * k)];
                        SwapEnd(save_value);
                        vtk_out_1.write(reinterpret_cast<char*>(&save_value), sizeof(T_P));
                    }
                }
            }

            vtk_out_1 << "SCALARS " << "density " << fmt << endl;
            vtk_out_1 << "LOOKUP_TABLE" << " default" << endl;
            
            for (k = 0; k < nzGlobal; k++) {
                for (j = 0; j < nyGlobal; j++) {
                    for (i = 0; i < nxGlobal; i++) {
                        T_P save_value = dd[i + l2 * (j + m2 * k)];
                        SwapEnd(save_value);
                        vtk_out_1.write(reinterpret_cast<char*>(&save_value), sizeof(T_P));
                    }
                }
            }

            vtk_out_1 << "SCALARS " << "velocity_X " << fmt << endl;
            vtk_out_1 << "LOOKUP_TABLE" << " default" << endl;
            
            for (k = 0; k < nzGlobal; k++) {
                for (j = 0; j < nyGlobal; j++) {
                    for (i = 0; i < nxGlobal; i++) {
                        T_P save_value = utt[i + l2 * (j + m2 * k)];
                        SwapEnd(save_value);
                        vtk_out_1.write(reinterpret_cast<char*>(&save_value), sizeof(T_P));
                    }
                }
            }

            vtk_out_1 << "SCALARS " << "velocity_Y " << fmt << endl;
            vtk_out_1 << "LOOKUP_TABLE" << " default" << endl;
            
            for (k = 0; k < nzGlobal; k++) {
                for (j = 0; j < nyGlobal; j++) {
                    for (i = 0; i < nxGlobal; i++) {
                        T_P save_value = vtt[i + l2 * (j + m2 * k)];
                        SwapEnd(save_value);
                        vtk_out_1.write(reinterpret_cast<char*>(&save_value), sizeof(T_P));
                    }
                }
            }

            vtk_out_1 << "SCALARS " << "velocity_Z " << fmt << endl;
            vtk_out_1 << "LOOKUP_TABLE" << " default" << endl;
            
            for (k = 0; k < nzGlobal; k++) {
                for (j = 0; j < nyGlobal; j++) {
                    for (i = 0; i < nxGlobal; i++) {
                        T_P save_value = wtt[i + l2 * (j + m2 * k)];
                        SwapEnd(save_value);
                        vtk_out_1.write(reinterpret_cast<char*>(&save_value), sizeof(T_P));
                    }
                }
            }

            free(ff);
            free(dd);
            free(utt);
            free(vtt);
            free(wtt);

        }
        else if (vtk_type == 2) { // single precision phase field
            fmt = "float";
            vtk_out_1 << "SCALARS " << "phi " << fmt << endl;
            vtk_out_1 << "LOOKUP_TABLE" << " default" << endl;
            
            for (k = 0; k < nzGlobal; k++) {
                for (j = 0; j < nyGlobal; j++) {
                    for (i = 0; i < nxGlobal; i++) {
                        float save_value = ff4[i + l2 * (j + m2 * k)];
                        SwapEnd(save_value);
                        vtk_out_1.write(reinterpret_cast<char*>(&save_value), sizeof(float));
                    }
                }
            }

            free(ff4);
        }
        vtk_out_1.close();

    }
    else {
        ERROR("Could not open vtk outputl file!");
    }

}

//***********************2D legacy VTK writer * **********************
//used in 2D micromodel simulation by averaging phi along y(depth) direction
void VTK_phi_2d_micromodel(int nt) {
    float* ff;
    int i, j, k, l2, m2, n2, wall_indicator;
    T_P tmp;

    l2 = nxGlobal;
    m2 = nyGlobal;
    n2 = nzGlobal;

    ff = (float*)calloc(l2 * m2 * n2, sizeof(float));
    // [(i - 1) + l2 * ((j - 1) + m2 * (k - 1))]

    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                wall_indicator = walls[i_s2(i, j, k)];
                phi[i_s4(i, j, k)] = prc(0.) * wall_indicator + phi[i_s4(i, j, k)] * (1 - wall_indicator);
            }
        }
    }

    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                ff[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))] = float(phi[i_s4(i, j, k)]);
            }
        }
    }

    for (k = 1; k <= nzGlobal; k++) {
        for (i = 1; i <= nxGlobal; i++) {
            tmp = prc(0.);
            for (j = 2; j <= nxGlobal - 1; j++) {
                tmp = tmp + ff[(i - 1) + l2 * ((j - 1) + m2 * (k - 1))];
            }
            ff[(i - 1) + l2 * ((nyGlobal - 1) + m2 * (k - 1))] = 0.5f * (1.f + float(tmp) / float((nyGlobal - 2)));   // use j = nyGlobal(solid layer) to store the averaged phase field
        }
    }

    ostringstream flnm;
    flnm << setfill('0') << setw(10) << nt << ".vtk";
    ostringstream filepath;
    string file_str;
    ofstream vtk_out_1;
    filepath << "results/out3.field_data/phase_distribution/small_2d_" << flnm.str();
    file_str = filepath.str();
    vtk_out_1.open(file_str.c_str(), ios_base::out | ios::binary);

    if (vtk_out_1.good()) {
        vtk_out_1 << "# vtk DataFile Version 3.0" << endl;
        vtk_out_1 << "vtk output" << endl;
        vtk_out_1 << "BINARY" << endl;
        vtk_out_1 << "DATASET STRUCTURED_POINTS" << endl;
        vtk_out_1 << "DIMENSIONS " << nxGlobal << " " << 1 << " " << nzGlobal << endl;
        vtk_out_1 << "ORIGIN " << 1 << " " << 1 << " " << 1 << endl;
        vtk_out_1 << "SPACING " << 1 << " " << 0 << " " << 1 << endl;
        vtk_out_1 << "POINT_DATA " << nxGlobal * nzGlobal << endl;
        string fmt = "float";
        vtk_out_1 << "SCALARS " << "phi " << fmt << endl;
        vtk_out_1 << "LOOKUP_TABLE" << " default" << endl;
        
        for (k = 1; k <= nzGlobal; k++) {
            for (i = 1; i <= nxGlobal; i++) {
                float save_value = ff[(i - 1) + l2 * ((nyGlobal - 1) + m2 * (k - 1))];
                SwapEnd(save_value);
                vtk_out_1.write(reinterpret_cast<char*>(&save_value), sizeof(float));
            }
        }

        free(ff);
    }
    else {
        ERROR("results/out3.field_data/phase_distribution/small_2d");
    }

}

/* ****************** save geometry VTK *********************** */
void VTK_walls_bin() {  // solid geometry
    int i, j, k;
    bool ALIVE;

    string filepath = "results/out3.field_data/walls_bin.vtk";
    ofstream walls_vtk(filepath.c_str(), ios_base::out | ios::binary);
    ALIVE = walls_vtk.good();
    if (ALIVE) {

        walls_vtk << "# vtk DataFile Version 3.0" << endl;
        walls_vtk << "vtk output" << endl;
        walls_vtk << "BINARY" << endl;
        walls_vtk << "DATASET STRUCTURED_POINTS" << endl;
        walls_vtk << "DIMENSIONS " << nxGlobal << " " << nyGlobal << " " << nzGlobal << endl;
        walls_vtk << "ORIGIN " << 1 << " " << 1 << " " << 1 << endl;
        walls_vtk << "SPACING " << 1 << " " << 1 << " " << 1 << endl;
        walls_vtk << "POINT_DATA " << nxGlobal * nyGlobal * nzGlobal << endl;

        // scalar - walls
        walls_vtk << "SCALARS " << "walls" << " int" << endl;
        walls_vtk << "LOOKUP_TABLE" << " default" << endl;
        
        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    int save_value = walls_global[i_s0(i, j, k)];
                    SwapEnd(save_value);
                    walls_vtk.write(reinterpret_cast<char*>(&save_value), sizeof(int));
                }
            }
        }
        walls_vtk.close();
    }
    else {
        ERROR("Could not create 'results/out3.field_data/walls_bin.vtk' file!");
    }

}




