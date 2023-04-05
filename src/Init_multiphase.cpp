#include "externLib.h"
#include "solver_precision.h"
#include "Init_multiphase.h"
#include "preprocessor.h"
#include "Module_extern.h"
#include "Fluid_singlephase_extern.h"
#include "Fluid_multiphase_extern.h"
#include "utils.h"
#include "Idx_cpu.h"

#include "IO_multiphase.h"
#include "Misc.h"
#include "Geometry_preprocessing.h"

//=====================================================================================================================================
//----------------------initialization basic----------------------
//=====================================================================================================================================
void initialization_basic_multi() {

    read_parameter_multi();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // create folders
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "Creating directories if not exist" << endl;
    // create directory results
    string filepath = "results/"; // directory path
    bool file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results folder !"); } // check that the directory was created successfully
    }

    // create directory out2.checkpoint 
    filepath = "results/out2.checkpoint/"; // directory path
    file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results/out2_checkpoint folder !"); } // check that the directory was created successfully
    }

    // create directory out2.checkpoint/2rd_backup
    filepath = "results/out2.checkpoint/2rd_backup/"; // directory path
    file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results/out2.checkpoint/2rd_backup folder !"); } // check that the directory was created successfully
    }

    // create directory out1.output
    filepath = "results/out1.output/"; // directory path
    file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results/out1.output folder !"); } // check that the directory was created successfully
    }

    // create directory out1.output/profile
    filepath = "results/out1.output/profile/"; // directory path
    file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results/out1.output/profile folder !"); } // check that the directory was created successfully
    }

    // create directory out3.field_data
    filepath = "results/out3.field_data/"; // directory path
    file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results/out3.field_data folder !"); } // check that the directory was created successfully
    }

    // create directory out3.field_data/phase_distribution
    filepath = "results/out3.field_data/phase_distribution/"; // directory path
    file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results/out3.field_data/phase_distribution folder !"); } // check that the directory was created successfully
    }

    // create directory out3.field_data/full_flow_field
    filepath = "results/out3.field_data/full_flow_field/"; // directory path
    file_check = fs::is_directory(filepath); // check if directory exists
    if (!file_check) { // If directory does not exist, create one!
        bool file_create_err = fs::create_directory(filepath); // create a directory
        if (!file_create_err) { ERROR("Could not Create results/out3.field_data/full_flow_field folder !"); } // check that the directory was created successfully
    }
    
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << endl;

    /* initialize memory size */
    initMemSize();

    cout << "**************************** Processing geometry *****************************" << endl;
    cout << "........." << endl;
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // initializing walls
    MemAllocate_geometry(1);
    set_walls();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // geometry related preprocessing
    if (geometry_preprocess_cmd == 0) {
        cout << "------ Start processing boundary nodes info ------" << endl;
        geometry_preprocessing_new();
        cout << "------ End processing boundary nodes info --------" << endl;
    }
    else if (geometry_preprocess_cmd == 1) {
        ifstream file_geo(geo_boundary_file_path.c_str());
        if (file_geo.good()) {
            cout << "------ Start loading boundary nodes info ---------" << endl;
            geometry_preprocessing_load();
            cout << "------ End loading boundary nodes info -----------" << endl;
        }
        else {
            ERROR("Error! No precomputed boundary info file found! Exiting program!");
        }
    }
    else {
        ERROR("Wrong value of geometry_preprocess_cmd! Stop program!");
    }

    // ~~~~~~~~~~~~~~~~~~dimensions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // la_x, y, z only used in fluid displacement simulation
    // channel walls are considered below for effective sample volumeand cross sectional area
    la_z = T_P(nzGlobal - 1);
    la_y = T_P(nyGlobal - 1) - prc(0.5) - prc(0.5);    // half - way bounceback 0.5 + 0.5
    la_x = T_P(nxGlobal - 1) - prc(0.5) - prc(0.5);
    A_xy = la_x * la_y;  // cross - sectional area for an open duct
    volume_sample = A_xy * la_z;     // domain volume

    string File_info = "results/out1.output/info.txt";
    ofstream file_info(File_info.c_str(), ios_base::out);
    if (file_info.good()) {
        file_info << " Grid information:" << endl;
        file_info << "nxGlobal = " << nxGlobal << " , nyGlobal = " << nyGlobal << " , nzGlobal = " << nzGlobal << endl;
        file_info << "Inlet open cross sectional area = " << A_xy << endl;
        file_info << " Pore information:" << endl;
        file_info << "Total number of pore nodes = " << pore_sum << endl;
        file_info << "Total number of solid boundary nodes = " << num_solid_boundary_global << endl;
        file_info << "Total number of fluid boundary nodes = " << num_fluid_boundary_global << endl;

        file_info << "Total number of effective pore nodes (excluding inlet/outlet) = " << pore_sum_effective << endl;

        porosity_full = T_P(pore_sum) / T_P(((nxGlobal - 2) * (nyGlobal - 2) * (nzGlobal)));
        porosity_effective = T_P(pore_sum_effective) / ((nxGlobal - 2) * (nyGlobal - 2)) / T_P(nzGlobal - n_exclude_outlet - n_exclude_inlet);
        file_info << "Full domain porosity = " << porosity_full << endl;
        file_info << "Effective domain porosity  (excluding inlet/outlet) = " << porosity_effective << endl;
        file_info.close();

        cout << "Total number of pore nodes = " << pore_sum << endl;
        cout << "Inlet open cross sectional area = " << A_xy << endl;
        cout << "Total number of effective pore nodes (excluding inlet/outlet) = " << pore_sum_effective << endl;
        cout << "Full domain porosity = " << porosity_full << endl;
        cout << "Effective domain porosity  (excluding inlet/outlet) = " << porosity_effective << endl;
    }
    else {
        ERROR("Couldn't open the file results/out1.output/info.txt !");
    }
    cout << "************************** End Processing geometry ***************************" << endl;
    cout << endl;

    cout << "******************** Processing fluid and flow info **********************" << endl;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //allocate fluid memory
    MemAllocate_multi(1);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // parameters
    rt1 = prc(3.) * la_nu1 + prc(0.5);      // relaxation time in collision, related to viscosity
    rti1 = prc(1.) / rt1;
    rt2 = prc(3.) * la_nu2 + prc(0.5);
    rti2 = prc(2.) / rt2;
    la_nui1 = prc(1.) / la_nu1;
    la_nui2 = prc(1.) / la_nu2;

    cout << "Fluid 1 relaxation time = " << rt1 << endl;
    cout << "Fluid 2 relaxation time = " << rt2 << endl;

    // injection parameters
    phi_inlet = prc(2.) * sa_inject - prc(1.);     // inlet BC order parameter, consistent with injection fluid

     // body force, pressure gradient
    // if pressure or velocity BC is used to drive the flow, force_z should be set to 0
    force_z = force_z0;

    // outlet pressure(density) only used when outlet_BC == 2
    rho_out = prc(1.);

    cout << "Inlet boundary nodes order parameter = " << phi_inlet << endl;

    // open pressure or velocity inlet BC
    if (kper == 0 && domain_wall_status_z_min == 0 && domain_wall_status_z_max == 0) { // non - periodic BC along flow direction(z)
        if (inlet_BC == 1) { // velocity inlet bc
            force_z = prc(0.);
            initialization_open_velocity_inlet_BC();
        }
        else if (inlet_BC == 2) { // pressure inlet bc
            force_z = prc(0.);                       // M.S: This line is not part of the original MF-LBM and a potential bug in the original implementation!
            p_gradient = -force_z0 / prc(3.);        // pressure gradient
            if (rho_out_BC) {
                rho_out = prc(1.) - p_gradient * nzGlobal;
                cout << "Outlet density (pressure outlet BC) = " << rho_out << endl;
            }
            else {
                rho_in = rho_out - p_gradient * nzGlobal;
                cout << "Inlet density (pressure inlet BC) = " << rho_in << endl;
            }
            
        }
    }

    cout << "******************** End processing fluid and flow info **********************" << endl;

    if (d_vol_animation > prc(0.) && inlet_BC == 1 && kper == 0 && domain_wall_status_z_min == 0 && domain_wall_status_z_max == 0) {
        ntime_animation = int(T_P(d_vol_animation * pore_sum) / flowrate);
        // due to AA pattern streaming, PDFs after odd step is stored in oppotite way.
        // only use even step for outputs
        if (ntime_animation % 2 == 1) {
            ntime_animation++;
        }
        cout << "Animation VTK files timer is modified based on injected volume: " << ntime_animation << endl;
    }

    if (d_vol_detail > prc(0.) && inlet_BC == 1 && kper == 0 && domain_wall_status_z_min == 0 && domain_wall_status_z_max == 0) {
        ntime_visual = int(T_P(d_vol_detail * pore_sum) / flowrate);
        // due to AA pattern streaming, PDFs after odd step is stored in oppotite way.
        // only use even step for outputs
        if (ntime_visual % 2 == 1) {
            ntime_visual++;
        }
        cout << "Full VTK files timer is modified based on injected volume: " << ntime_visual << endl;
    }

    if (d_vol_monitor > prc(0.) && inlet_BC == 1 && kper == 0 && domain_wall_status_z_min == 0 && domain_wall_status_z_max == 0) {
        // due to AA pattern streaming, PDFs after odd step is stored in oppotite way.
        // only use even step for outputs
        ntime_monitor = int(T_P(d_vol_monitor * pore_sum) / flowrate);
        if (ntime_monitor % 2 == 1) {
            ntime_monitor++;
        }
        ntime_monitor_profile = ntime_monitor_profile_ratio * ntime_monitor;
        cout << "Monitor timer is modified based on injected volume: " << ntime_monitor << endl;
        cout << "Monitor_profile timer is modified based on injected volume: " << ntime_monitor_profile << endl;
    }

    monitor_previous_value = prc(0.);
    monitor_current_value = prc(0.);
    kinetic_energy[0] = kinetic_energy[1] = prc(0.);


}

// ~~~~~~~~~~~~~~~~~~~~~open velocity inlet BC initialization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void initialization_open_velocity_inlet_BC() {
    long long x, y;
    long long i, j;
    int n_vin;

    uin_avg_0 = ca_0 * lbm_gamma / la_nu1;            // common definition
    uin_avg = uin_avg_0;
    flowrate = uin_avg_0 * A_xy;

    cout << "Inlet average velocity = " << uin_avg_0 << endl;
    cout << "Inlet flowrate = " << flowrate << endl;

    // default uniform inlet velocity profile
    for (j = 1; j <= nyGlobal; j++) {
        for (i = 1; i <= nxGlobal; i++) {
            x = i;
            y = j;
            W_in[i_s1(i, j, 0)] = prc(0.);
            if (x > 1 && x < nxGlobal && y > 1 && y < nyGlobal) {
                W_in[i_s1(i, j, 0)] = uin_avg;
            }
        }
    }

    n_vin = 1000;
    // analytical velocity profile
    inlet_vel_profile_rectangular(uin_avg_0, n_vin);

    if (target_inject_pore_volume > 0) {// stop simulation based on injected volume is enabled
        ntime_max = int(T_P(target_inject_pore_volume * pore_sum) / flowrate);
        if (ntime_max % 2 == 1) {
            ntime_max = ntime_max + 1;
        }
        cout << "Maximum time step is modified based on target inject volume! ntime_max = " << ntime_max << endl;
    }

}

//=====================================================================================================================================
//----------------------initialization for new simulation - field variables----------------------
//=====================================================================================================================================
void initialization_new_multi() {
    long long i, j, k;
    T_P x, y, z, random;

    ntime0 = 1;

    // Use current time as seed for random generator
    srand((unsigned)time(NULL));
    // Initial fluid phase distribution
    for (k = 1 - 1; k <= nzGlobal + 1; k++) {
        for (j = 1 - 1; j <= nyGlobal + 1; j++) {
            for (i = 1 - 1; i <= nxGlobal + 1; i++) {
                x = T_P(i);
                y = T_P(j);
                z = T_P(k);
                if (initial_fluid_distribution_option == 1) { // drainage
                    phi[i_s4(i, j, k)] = prc(-1.);
                    if (z <= interface_z0) {
                        phi[i_s4(i, j, k)] = prc(1.);
                    }
                }
                else if (initial_fluid_distribution_option == 2) { // imbibition
                    phi[i_s4(i, j, k)] = prc(1.);
                    if (z <= interface_z0) {
                        phi[i_s4(i, j, k)] = prc(-1.);
                    }
                }
                else if (initial_fluid_distribution_option == 3) { // contact angle measurement
                    phi[i_s4(i, j, k)] = prc(-1.);
                    if (prc(pow)((x - (nxGlobal + 1) * prc(0.)), 2) + prc(pow)((z - (nzGlobal + 1) * prc(0.5)), 2) + prc(pow)((y - (nyGlobal + 1) * prc(0.5)), 2) <= prc(pow)(interface_z0, 2)) {
                        phi[i_s4(i, j, k)] = prc(1.);
                    }
                }
                else if (initial_fluid_distribution_option == 4) { // contact angle measurement
                    phi[i_s4(i, j, k)] = prc(1.);
                    if (prc(pow)((x - (nxGlobal + 1) * prc(0.)), 2) + prc(pow)((z - (nzGlobal + 1) * prc(0.5)), 2) + prc(pow)((y - (nyGlobal + 1) * prc(0.5)), 2) <= prc(pow)(interface_z0, 2)) {
                        phi[i_s4(i, j, k)] = prc(-1.);
                    }
                }
                else if (initial_fluid_distribution_option == 5) { // surface tension measurement
                    phi[i_s4(i, j, k)] = prc(-1.);
                    if (prc(pow)((x - (nxGlobal + 1) * prc(0.5)), 2) + prc(pow)((z - (nzGlobal + 1) * prc(0.5)), 2) + prc(pow)((y - (nyGlobal + 1) * prc(0.5)), 2) <= prc(pow)(interface_z0, 2)) {
                        phi[i_s4(i, j, k)] = prc(1.);
                    }
                }
                else if (initial_fluid_distribution_option == 6) { // randomly distributed : steady state relative perm measurement
                    phi[i_s4(i, j, k)] = prc(1.);
                    random = (T_P)rand() / RAND_MAX;
                    if (random > sa_target) {
                        phi[i_s4(i, j, k)] = prc(-1.);
                    }
                }
                else {
                    ERROR("Input parameter initial_fluid_distribution_option error! Stop program!!!");
                }
            }
        }
    }

    if (kper == 0 && domain_wall_status_z_min == 0 && domain_wall_status_z_max == 0) { // non - periodic BC along flow direction(z)
        for (k = 1 - 4; k <= nzGlobal + 4; k++) {
            for (j = 1 - 4; j <= nyGlobal + 4; j++) {
                for (i = 1 - 4; i <= nxGlobal + 4; i++) {
                    x = T_P(i);
                    y = T_P(j);
                    z = T_P(k);
                    if (z <= 0) {
                        phi[i_s4(i, j, k)] = phi_inlet; // phi_inlet = 2d0 * sa_inject - 1d0, inlet BC order parameter, consistent with injecting fluid saturation
                    }

                }
            }
        }
    }

    initialization_new_multi_pdf();

    if (steady_state_option == 2) {
        // using the phase field difference to indicate steady state
        for (k = 1 - 4; k <= nzGlobal + 4; k++) {
            for (j = 1 - 4; j <= nyGlobal + 4; j++) {
                for (i = 1 - 4; i <= nxGlobal + 4; i++) {
                    phi_old[i_s4(i, j, k)] = phi[i_s4(i, j, k)];
                }
            }
        }
    }
}

// initial particle distribution functions
void initialization_new_multi_pdf() {
    long long i, j, k;
    T_P usqrt, rho1, rho2;

    for (k = 1 - 1; k <= nzGlobal + 1; k++) {
        for (j = 1 - 1; j <= nyGlobal + 1; j++) {
            for (i = 1 - 1; i <= nxGlobal + 1; i++) {
                usqrt = u[i_s1(i, j, k)] * u[i_s1(i, j, k)] + v[i_s1(i, j, k)] * v[i_s1(i, j, k)] + w[i_s1(i, j, k)] * w[i_s1(i, j, k)];
                rho1 = rho[i_s1(i, j, k)] * (prc(1.0) + phi[i_s4(i, j, k)]) * prc(0.5);
                rho2 = rho[i_s1(i, j, k)] * (prc(1.0) - phi[i_s4(i, j, k)]) * prc(0.5);

                pdf[i_f1(i, j, k, 0, 0)] = rho1 * w_equ_0 + rho1 * w_equ_0 * (prc(-1.5) * usqrt);
                pdf[i_f1(i, j, k, 1, 0)] = rho1 * w_equ_1 + rho1 * w_equ_1 * (prc(3.0) * u[i_s1(i, j, k)] + prc(4.5) * u[i_s1(i, j, k)] * u[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 2, 0)] = rho1 * w_equ_1 + rho1 * w_equ_1 * (prc(3.0) * (-u[i_s1(i, j, k)]) + prc(4.5) * u[i_s1(i, j, k)] * u[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 3, 0)] = rho1 * w_equ_1 + rho1 * w_equ_1 * (prc(3.0) * v[i_s1(i, j, k)] + prc(4.5) * v[i_s1(i, j, k)] * v[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 4, 0)] = rho1 * w_equ_1 + rho1 * w_equ_1 * (prc(3.0) * (-v[i_s1(i, j, k)]) + prc(4.5) * v[i_s1(i, j, k)] * v[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 5, 0)] = rho1 * w_equ_1 + rho1 * w_equ_1 * (prc(3.0) * w[i_s1(i, j, k)] + prc(4.5) * w[i_s1(i, j, k)] * w[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 6, 0)] = rho1 * w_equ_1 + rho1 * w_equ_1 * (prc(3.0) * (-w[i_s1(i, j, k)]) + prc(4.5) * w[i_s1(i, j, k)] * w[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 7, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) + prc(4.5) * (u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) * (u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 8, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (-u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) + prc(4.5) * (-u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) * (-u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 9, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) + prc(4.5) * (u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) * (u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 10, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (-u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) + prc(4.5) * (-u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) * (-u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 11, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) + prc(4.5) * (u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) * (u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 12, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (-u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) + prc(4.5) * (-u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) * (-u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 13, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) + prc(4.5) * (u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) * (u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 14, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (-u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) + prc(4.5) * (-u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) * (-u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 15, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) + prc(4.5) * (v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) * (v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 16, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (-v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) + prc(4.5) * (-v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) * (-v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 17, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) + prc(4.5) * (v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) * (v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 18, 0)] = rho1 * w_equ_2 + rho1 * w_equ_2 * (prc(3.0) * (-v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) + prc(4.5) * (-v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) * (-v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) - prc(1.5) * usqrt);

                pdf[i_f1(i, j, k, 0, 1)] = rho2 * w_equ_0 + rho2 * w_equ_0 * (prc(-1.5) * usqrt);
                pdf[i_f1(i, j, k, 1, 1)] = rho2 * w_equ_1 + rho2 * w_equ_1 * (prc(3.0) * u[i_s1(i, j, k)] + prc(4.5) * u[i_s1(i, j, k)] * u[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 2, 1)] = rho2 * w_equ_1 + rho2 * w_equ_1 * (prc(3.0) * (-u[i_s1(i, j, k)]) + prc(4.5) * u[i_s1(i, j, k)] * u[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 3, 1)] = rho2 * w_equ_1 + rho2 * w_equ_1 * (prc(3.0) * v[i_s1(i, j, k)] + prc(4.5) * v[i_s1(i, j, k)] * v[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 4, 1)] = rho2 * w_equ_1 + rho2 * w_equ_1 * (prc(3.0) * (-v[i_s1(i, j, k)]) + prc(4.5) * v[i_s1(i, j, k)] * v[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 5, 1)] = rho2 * w_equ_1 + rho2 * w_equ_1 * (prc(3.0) * w[i_s1(i, j, k)] + prc(4.5) * w[i_s1(i, j, k)] * w[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 6, 1)] = rho2 * w_equ_1 + rho2 * w_equ_1 * (prc(3.0) * (-w[i_s1(i, j, k)]) + prc(4.5) * w[i_s1(i, j, k)] * w[i_s1(i, j, k)] - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 7, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) + prc(4.5) * (u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) * (u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 8, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (-u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) + prc(4.5) * (-u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) * (-u[i_s1(i, j, k)] + v[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 9, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) + prc(4.5) * (u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) * (u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 10, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (-u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) + prc(4.5) * (-u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) * (-u[i_s1(i, j, k)] - v[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 11, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) + prc(4.5) * (u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) * (u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 12, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (-u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) + prc(4.5) * (-u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) * (-u[i_s1(i, j, k)] + w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 13, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) + prc(4.5) * (u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) * (u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 14, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (-u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) + prc(4.5) * (-u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) * (-u[i_s1(i, j, k)] - w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 15, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) + prc(4.5) * (v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) * (v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 16, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (-v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) + prc(4.5) * (-v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) * (-v[i_s1(i, j, k)] + w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 17, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) + prc(4.5) * (v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) * (v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) - prc(1.5) * usqrt);
                pdf[i_f1(i, j, k, 18, 1)] = rho2 * w_equ_2 + rho2 * w_equ_2 * (prc(3.0) * (-v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) + prc(4.5) * (-v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) * (-v[i_s1(i, j, k)] - w[i_s1(i, j, k)]) - prc(1.5) * usqrt);

            }
        }
    }

    // ~~~~~~~~~~~~~~~~~~~convective outflow BC ~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (outlet_BC == 1) {

            for (j = 1 - 1; j <= nyGlobal + 1; j++) {
                for (i = 1 - 1; i <= nxGlobal + 1; i++) {
                    f_convec_bc[icnv_f1(i, j, 0)] = pdf[i_f1(i, j, nzGlobal, 0, 0)];
                    f_convec_bc[icnv_f1(i, j, 1)] = pdf[i_f1(i, j, nzGlobal, 1, 0)];
                    f_convec_bc[icnv_f1(i, j, 2)] = pdf[i_f1(i, j, nzGlobal, 2, 0)];
                    f_convec_bc[icnv_f1(i, j, 3)] = pdf[i_f1(i, j, nzGlobal, 3, 0)];
                    f_convec_bc[icnv_f1(i, j, 4)] = pdf[i_f1(i, j, nzGlobal, 4, 0)];
                    f_convec_bc[icnv_f1(i, j, 5)] = pdf[i_f1(i, j, nzGlobal, 5, 0)];
                    f_convec_bc[icnv_f1(i, j, 6)] = pdf[i_f1(i, j, nzGlobal, 6, 0)];
                    f_convec_bc[icnv_f1(i, j, 7)] = pdf[i_f1(i, j, nzGlobal, 7, 0)];
                    f_convec_bc[icnv_f1(i, j, 8)] = pdf[i_f1(i, j, nzGlobal, 8, 0)];
                    f_convec_bc[icnv_f1(i, j, 9)] = pdf[i_f1(i, j, nzGlobal, 9, 0)];
                    f_convec_bc[icnv_f1(i, j, 10)] = pdf[i_f1(i, j, nzGlobal, 10, 0)];
                    f_convec_bc[icnv_f1(i, j, 11)] = pdf[i_f1(i, j, nzGlobal, 11, 0)];
                    f_convec_bc[icnv_f1(i, j, 12)] = pdf[i_f1(i, j, nzGlobal, 12, 0)];
                    f_convec_bc[icnv_f1(i, j, 13)] = pdf[i_f1(i, j, nzGlobal, 13, 0)];
                    f_convec_bc[icnv_f1(i, j, 14)] = pdf[i_f1(i, j, nzGlobal, 14, 0)];
                    f_convec_bc[icnv_f1(i, j, 15)] = pdf[i_f1(i, j, nzGlobal, 15, 0)];
                    f_convec_bc[icnv_f1(i, j, 16)] = pdf[i_f1(i, j, nzGlobal, 16, 0)];
                    f_convec_bc[icnv_f1(i, j, 17)] = pdf[i_f1(i, j, nzGlobal, 17, 0)];
                    f_convec_bc[icnv_f1(i, j, 18)] = pdf[i_f1(i, j, nzGlobal, 18, 0)];

                    g_convec_bc[icnv_f1(i, j, 0)] = pdf[i_f1(i, j, nzGlobal, 0, 1)];
                    g_convec_bc[icnv_f1(i, j, 1)] = pdf[i_f1(i, j, nzGlobal, 1, 1)];
                    g_convec_bc[icnv_f1(i, j, 2)] = pdf[i_f1(i, j, nzGlobal, 2, 1)];
                    g_convec_bc[icnv_f1(i, j, 3)] = pdf[i_f1(i, j, nzGlobal, 3, 1)];
                    g_convec_bc[icnv_f1(i, j, 4)] = pdf[i_f1(i, j, nzGlobal, 4, 1)];
                    g_convec_bc[icnv_f1(i, j, 5)] = pdf[i_f1(i, j, nzGlobal, 5, 1)];
                    g_convec_bc[icnv_f1(i, j, 6)] = pdf[i_f1(i, j, nzGlobal, 6, 1)];
                    g_convec_bc[icnv_f1(i, j, 7)] = pdf[i_f1(i, j, nzGlobal, 7, 1)];
                    g_convec_bc[icnv_f1(i, j, 8)] = pdf[i_f1(i, j, nzGlobal, 8, 1)];
                    g_convec_bc[icnv_f1(i, j, 9)] = pdf[i_f1(i, j, nzGlobal, 9, 1)];
                    g_convec_bc[icnv_f1(i, j, 10)] = pdf[i_f1(i, j, nzGlobal, 10, 1)];
                    g_convec_bc[icnv_f1(i, j, 11)] = pdf[i_f1(i, j, nzGlobal, 11, 1)];
                    g_convec_bc[icnv_f1(i, j, 12)] = pdf[i_f1(i, j, nzGlobal, 12, 1)];
                    g_convec_bc[icnv_f1(i, j, 13)] = pdf[i_f1(i, j, nzGlobal, 13, 1)];
                    g_convec_bc[icnv_f1(i, j, 14)] = pdf[i_f1(i, j, nzGlobal, 14, 1)];
                    g_convec_bc[icnv_f1(i, j, 15)] = pdf[i_f1(i, j, nzGlobal, 15, 1)];
                    g_convec_bc[icnv_f1(i, j, 16)] = pdf[i_f1(i, j, nzGlobal, 16, 1)];
                    g_convec_bc[icnv_f1(i, j, 17)] = pdf[i_f1(i, j, nzGlobal, 17, 1)];
                    g_convec_bc[icnv_f1(i, j, 18)] = pdf[i_f1(i, j, nzGlobal, 18, 1)];


                    phi_convec_bc[i_s1(i, j, 0)] = phi[i_s4(i, j, nzGlobal)];

            }
        }
    }

}

//=====================================================================================================================================
//----------------------initialization for old simulation - field variables----------------------
//=====================================================================================================================================
void initialization_old_multi() {

    /* Open checkpoint file */
   /* File name */
    ostringstream flnm;
    flnm << "id" << setfill('0') << setw(4) << 0; // This line makes the width (4) and fills the values other than the required with (0)
    ostringstream filepath;
    filepath << "results/out2.checkpoint/" << flnm.str();
    string fns = filepath.str();
    const char* fnc = fns.c_str();
    FILE* check_point_file = fopen(fnc, "rb"); // open the file (rb: read binary)
    if (check_point_file == NULL) {
        ERROR("Checkpoint data not found! Exiting program!");
    }

    cout << "loading checkpoint data ... ";

    int nt_temp;
    if (!fread(&nt_temp, sizeof(int), 1, check_point_file)) { ERROR("Could not load from data check_point_file file!"); } ntime0 = nt_temp - 1;
    if (!fread(&force_z, sizeof(T_P), 1, check_point_file)) { ERROR("Could not load from data check_point_file file!"); }
    if (!fread(&rho_in, sizeof(T_P), 1, check_point_file)) { ERROR("Could not load from data check_point_file file!"); }

    if (!fread(pdf, mem_size_f1_TP, 1, check_point_file)) { ERROR("Could not load from data check_point_file file!"); }

    if (!fread(phi, mem_size_s4_TP, 1, check_point_file)) { ERROR("Could not load from data check_point_file file!"); }

    if (outlet_BC == 1) { // convective BC
        long long mem_size_tmp = NXG1 * NYG1 * 19 * sizeof(T_P);
        if (!fread(f_convec_bc, mem_size_tmp, 1, check_point_file)) { ERROR("Could not load from data check_point_file file!"); }
        if (!fread(g_convec_bc, mem_size_tmp, 1, check_point_file)) { ERROR("Could not load from data check_point_file file!"); }
        mem_size_tmp = NXG1 * NYG1 * sizeof(T_P);
        if(!fread(phi_convec_bc, mem_size_tmp, 1, check_point_file)) { ERROR("Could not load from data check_point_file file!"); }
    }
    fclose(check_point_file);

    cout << "Complete" << endl;

}


//=====================================================================================================================================
//----------------------memory allocate / deallocate----------------------
//=====================================================================================================================================
//***************geometry related memory allocate / deallocate * *******************************
void MemAllocate_geometry(int flag) {
    int FLAG = flag;
    if (FLAG == 1) {
        walls = (int*)calloc(num_cells_s2, sizeof(int));
        walls_global = (int*)calloc(num_cells_s0, sizeof(int));
        walls_type = (int*)calloc(num_cells_s4, sizeof(int));
        s_nx = (T_P*)calloc(num_cells_s4, sizeof(T_P));
        s_ny = (T_P*)calloc(num_cells_s4, sizeof(T_P));
        s_nz = (T_P*)calloc(num_cells_s4, sizeof(T_P));
    }
    else {
        free(walls);
        free(walls_global);
        free(walls_type);
        free(s_nx);
        free(s_ny);
        free(s_nz);
    }
}

//************* fluid flow related memory allocate/deallocate ******************************
void MemAllocate_multi(int flag) {
    int FLAG = flag;
    if (FLAG == 1) {
        u = (T_P*)calloc(num_cells_s1, sizeof(T_P));
        v = (T_P*)calloc(num_cells_s1, sizeof(T_P));
        w = (T_P*)calloc(num_cells_s1, sizeof(T_P));
        rho = (T_P*)calloc(num_cells_s1, sizeof(T_P));    fill_n(rho, num_cells_s1, prc(1.));
        curv = (T_P*)calloc(num_cells_s1, sizeof(T_P));
        W_in = (T_P*)calloc(NXG1 * NYG1, sizeof(T_P));

        pdf = (T_P*)calloc(num_cells_f1, sizeof(T_P));

        cn_x = (T_P*)calloc(num_cells_s2, sizeof(T_P));
        cn_y = (T_P*)calloc(num_cells_s2, sizeof(T_P));
        cn_z = (T_P*)calloc(num_cells_s2, sizeof(T_P));
        c_norm = (T_P*)calloc(num_cells_s2, sizeof(T_P));
        
        //convective BC
        if (outlet_BC == 1) {
            phi_convec_bc = (T_P*)calloc(NXG1 * NYG1, sizeof(T_P));
            g_convec_bc = (T_P*)calloc(NXG1 * NYG1 * 19, sizeof(T_P));
            f_convec_bc = (T_P*)calloc(NXG1 * NYG1 * 19, sizeof(T_P));
        }

        phi = (T_P*)calloc(num_cells_s4, sizeof(T_P));

        //phase field steady state
        if (steady_state_option == 2) {
            phi_old = (T_P*)calloc(num_cells_s4, sizeof(T_P));// using the phase field difference to indicate steady state
        }

        // temporary arrays
        fl1 = (T_P*)calloc(nzGlobal, sizeof(T_P));
        fl2 = (T_P*)calloc(nzGlobal, sizeof(T_P));
        pre = (T_P*)calloc(nzGlobal, sizeof(T_P));
        mass1 = (T_P*)calloc(nzGlobal, sizeof(T_P));
        mass2 = (T_P*)calloc(nzGlobal, sizeof(T_P));
        vol1 = (T_P*)calloc(nzGlobal, sizeof(T_P));
        vol2 = (T_P*)calloc(nzGlobal, sizeof(T_P));

        int tk_isize = 7 * int(nzGlobal) + 3;
        tk = (T_P*)calloc(tk_isize, sizeof(T_P));   // !isize = 5 * nz + 1          !fl1, fl2, pre, mass1, mass2, vol1, vol2 + umax + usq1 + usq2(kinetic energy)
    }
    else {
        free(u);
        free(v);
        free(w);
        free(rho);
        free(curv);
        free(W_in);
        free(pdf);
        free(cn_x);
        free(cn_y);
        free(cn_z);
        free(c_norm);
        if (outlet_BC == 1) {
            free(phi_convec_bc);
            free(g_convec_bc);
            free(f_convec_bc);
        }
        free(phi);
        if (steady_state_option == 2) {
            free(phi_old);
        }
        free(fl1);
        free(fl2);
        free(pre);
        free(mass1);
        free(mass2);
        free(vol1);
        free(vol2);
        free(tk);
    }

}



//------------- initialize memory size -------------
//=================================================================== 
void initMemSize() {

    /* Total number of array nodes after adding the ghost nodes */
    num_cells_s0 = NXG0 * NYG0 * NZG0;
    num_cells_s1 = NXG1 * NYG1 * NZG1;
    num_cells_s2 = NXG2 * NYG2 * NZG2;
    num_cells_s4 = NXG4 * NYG4 * NZG4;

    num_cells_v1 = NXG1 * NYG1 * NZG1 * 3;
    num_cells_v2 = NXG2 * NYG2 * NZG2 * 3;

    num_cells_f1 = NXG1 * NYG1 * NZG1 * 19 * 2;

    /* Memory size of arrays */
    mem_size_s0_int = num_cells_s0 * sizeof(int);
    mem_size_s2_int = num_cells_s2 * sizeof(int);
    mem_size_s4_int = num_cells_s4 * sizeof(int);
    mem_size_s0_TP = num_cells_s0 * sizeof(T_P);
    mem_size_s1_TP = num_cells_s1 * sizeof(T_P);
    mem_size_s2_TP = num_cells_s2 * sizeof(T_P);
    mem_size_s4_TP = num_cells_s4 * sizeof(T_P);

    mem_size_v1_TP = num_cells_v1 * sizeof(T_P);
    mem_size_v2_TP = num_cells_v2 * sizeof(T_P);

    mem_size_f1_TP = num_cells_f1 * sizeof(T_P);


}