#include "externLib.h"
#include "solver_precision.h"
#include "preprocessor.h"
#include "Module_extern.h"
#include "Fluid_singlephase_extern.h"
#include "Fluid_multiphase_extern.h"
//#include "utils.h"
#include "Idx_cpu.h"
#include "IO_multiphase.h"
#include "Misc.h"

#include "Monitor.h"

//=======================================================================================================================================================
//----------------------monitor_multiphase  unsteady flow----------------------
//=======================================================================================================================================================
void monitor() {
    T_P* fl1_0, * fl2_0, * pre_0, * mass1_0, * mass2_0, * vol1_0, * vol2_0;
    int i, j, k;
    int wall_indicator;
    T_P umax, temp, temp1, temp2, temp3, temp4, temp5, temp6, usq1, usq2;
    T_P prek, fl1_avg, fl2_avg, fl_avg, fl1_avg_whole, fl2_avg_whole, fl_avg_whole;

    // *********************************preperation***********************************
    prek = prc(0.);
    temp = prc(0.);
    umax = prc(0.);

    compute_macro_vars();

    usq1 = prc(0.);   // kinetic energy of each phase
    usq2 = prc(0.);

    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                wall_indicator = walls[i_s2(i, j, k)];
                temp = (u[i_s1(i, j, k)] * u[i_s1(i, j, k)] + v[i_s1(i, j, k)] * v[i_s1(i, j, k)] + w[i_s1(i, j, k)] * w[i_s1(i, j, k)]) * (1 - wall_indicator);
                if (umax < temp) {
                    umax = temp;
                }
                if (phi[i_s4(i, j, k)] > prc(0.999)) {
                    usq1 = usq1 + temp;
                }
                else if (phi[i_s4(i, j, k)] < prc(-0.999)) {
                    usq2 = usq2 + temp;
                }
            }
        }
    }

    for (k = 1; k <= nzGlobal; k++) {
        temp1 = prc(0.);
        temp2 = prc(0.);
        prek = prc(0.);
        temp3 = prc(0.);
        temp4 = prc(0.);
        temp5 = prc(0.);
        temp6 = prc(0.);

        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                wall_indicator = walls[i_s2(i, j, k)];;
                temp3 = temp3 + prc(0.5) * (prc(1.) + phi[i_s4(i, j, k)]) * (1 - wall_indicator);   //volume fraction
                temp4 = temp4 + prc(0.5) * (prc(1.) - phi[i_s4(i, j, k)]) * (1 - wall_indicator);
                temp5 = temp5 + rho[i_s1(i, j, k)] * prc(0.5) * (prc(1.) + phi[i_s4(i, j, k)]) * (1 - wall_indicator);  //mass fraction
                temp6 = temp6 + rho[i_s1(i, j, k)] * prc(0.5) * (prc(1.) - phi[i_s4(i, j, k)]) * (1 - wall_indicator);
                temp1 = temp1 + w[i_s1(i, j, k)] * prc(0.5) * (prc(1.) + phi[i_s4(i, j, k)]) * (1 - wall_indicator);     //volume fractional flow fluid 1
                temp2 = temp2 + w[i_s1(i, j, k)] * prc(0.5) * (prc(1.) - phi[i_s4(i, j, k)]) * (1 - wall_indicator);     // volume fractional flow fluid 2
                prek = prek + rho[i_s1(i, j, k)] * (1 - wall_indicator);    //calculate average bulk pressure
            }
        }
        pre[i_s0(k, 1, 1)] = prek;                       //pressure sum, further divided by wallz_prof to obtain averaged pressure profile along the flow direction z
        fl1[i_s0(k, 1, 1)] = temp1;
        fl2[i_s0(k, 1, 1)] = temp2;
        vol1[i_s0(k, 1, 1)] = temp3;
        vol2[i_s0(k, 1, 1)] = temp4;
        mass1[i_s0(k, 1, 1)] = temp5;
        mass2[i_s0(k, 1, 1)] = temp6;
    }

    fl1_0 = (T_P*)calloc(nzGlobal, sizeof(T_P));
    fl2_0 = (T_P*)calloc(nzGlobal, sizeof(T_P));
    vol1_0 = (T_P*)calloc(nzGlobal, sizeof(T_P));
    vol2_0 = (T_P*)calloc(nzGlobal, sizeof(T_P));
    mass1_0 = (T_P*)calloc(nzGlobal, sizeof(T_P));
    mass2_0 = (T_P*)calloc(nzGlobal, sizeof(T_P));
    pre_0 = (T_P*)calloc(nzGlobal, sizeof(T_P));

    for (k = 1; k <= nzGlobal; k++) {
        fl1_0[i_s0(k, 1, 1)] += fl1[i_s0(k, 1, 1)];
        fl2_0[i_s0(k, 1, 1)] += fl2[i_s0(k, 1, 1)];
        vol1_0[i_s0(k, 1, 1)] += vol1[i_s0(k, 1, 1)];
        vol2_0[i_s0(k, 1, 1)] += vol2[i_s0(k, 1, 1)];
        mass1_0[i_s0(k, 1, 1)] += mass1[i_s0(k, 1, 1)];
        mass2_0[i_s0(k, 1, 1)] += mass2[i_s0(k, 1, 1)];
        pre_0[i_s0(k, 1, 1)] += pre[i_s0(k, 1, 1)];
    }

    kinetic_energy[0] = prc(0.5) * usq1;
    kinetic_energy[1] = prc(0.5) * usq2;

    umax_global = prc(sqrt)(umax);

    // *********************************************saturation calculation********************************************
    mass1_sum = prc(0.);
    mass2_sum = prc(0.);
    vol1_sum = prc(0.);
    vol2_sum = prc(0.);

    for (k = n_exclude_inlet + 1; k <= nzGlobal - n_exclude_outlet; k++) {
        mass1_sum = mass1_sum + mass1_0[i_s0(k, 1, 1)];
        mass2_sum = mass2_sum + mass2_0[i_s0(k, 1, 1)];
        vol1_sum = vol1_sum + vol1_0[i_s0(k, 1, 1)];
        vol2_sum = vol2_sum + vol2_0[i_s0(k, 1, 1)];
    }

    saturation = vol1_sum / (vol1_sum + vol2_sum);
    string filepath = "results/out1.output/saturation.dat";
    ofstream sat_file;
    if (ntime0 == 1 && ntime == ntime_monitor) { // open for the first time, new file
        sat_file.open(filepath.c_str(), ios_base::out);
    }
    else { // open during simulation after timesteps, append file
        sat_file.open(filepath.c_str(), ios_base::app);
    }

    sat_file << ntime << " " << saturation << " " << vol1_sum << " " << vol2_sum << " " << mass1_sum << " " << mass2_sum << endl;
    sat_file.close();

    temp1 = prc(0.);
    temp2 = prc(0.);
    temp3 = prc(0.);
    temp4 = prc(0.);

    for (k = 1; k <= nzGlobal; k++) {
        temp1 = temp1 + mass1_0[i_s0(k, 1, 1)];
        temp2 = temp2 + mass2_0[i_s0(k, 1, 1)];
        temp3 = temp3 + vol1_0[i_s0(k, 1, 1)];
        temp4 = temp4 + vol2_0[i_s0(k, 1, 1)];
    }

    saturation_full_domain = temp3 / (temp3 + temp4);   // full domain saturation

     // ********************************************* flowrate calculation **********************************************
    fl1_avg = prc(0.);
    fl2_avg = prc(0.);
    fl1_avg_whole = prc(0.);
    fl2_avg_whole = prc(0.);

    for (k = 1; k <= nzGlobal; k++) { // average velocity of whole domain
        fl1_avg_whole = fl1_avg_whole + fl1_0[i_s0(k, 1, 1)];
        fl2_avg_whole = fl2_avg_whole + fl2_0[i_s0(k, 1, 1)];
    }

    fl1_avg_whole = fl1_avg_whole / T_P((nzGlobal));
    fl2_avg_whole = fl2_avg_whole / T_P((nzGlobal));
    fl_avg_whole = fl1_avg_whole + fl2_avg_whole;

    for (k = n_exclude_inlet + 1; k <= nzGlobal - n_exclude_outlet; k++) { // average velocity of sample domain
        fl1_avg = fl1_avg + fl1_0[i_s0(k, 1, 1)];
        fl2_avg = fl2_avg + fl2_0[i_s0(k, 1, 1)];
    }

    fl1_avg = fl1_avg / T_P((nzGlobal - n_exclude_outlet - n_exclude_inlet));
    fl2_avg = fl2_avg / T_P((nzGlobal - n_exclude_outlet - n_exclude_inlet));
    fl_avg = fl1_avg + fl2_avg;

    // flowrate based on darcy velocity
    temp = fl_avg / A_xy;  //averaged darcy velocity for the bulk fluid
    ca = temp * la_nu1 / lbm_gamma;  // based on bulk fluid velocity

     /* ********************************************* save data ************************************************************ */
    filepath = "results/out1.output/Ca_number.dat";
    ofstream ca_num_file;
    if (ntime0 == 1 && ntime == ntime_monitor) { // open for the first time, new file
        ca_num_file.open(filepath.c_str(), ios_base::out);
    }
    else { // open during simulation after timesteps, append file
        ca_num_file.open(filepath.c_str(), ios_base::app);
    }
    ca_num_file << ntime << " " << ca << " " << umax_global << " " << kinetic_energy[0] << " " << kinetic_energy[1] << endl;
    ca_num_file.close();

    filepath = "results/out1.output/flowrate_time.dat";
    ofstream flow_rate_file;
    if (ntime0 == 1 && ntime == ntime_monitor) { // open for the first time, new file
        flow_rate_file.open(filepath.c_str(), ios_base::out);
    }
    else { // open during simulation after timesteps, append file
        flow_rate_file.open(filepath.c_str(), ios_base::app);
    }
    flow_rate_file << ntime << " " << fl_avg_whole << " " << fl1_avg_whole << " " << fl2_avg_whole << " " << fl_avg << " " << fl1_avg << " " << fl2_avg << endl;
    flow_rate_file.close();

    filepath = "results/out1.output/saturation_full_domain.dat";
    ofstream sat_full_file;
    if (ntime0 == 1 && ntime == ntime_monitor) { // open for the first time, new file
        sat_full_file.open(filepath.c_str(), ios_base::out);
    }
    else { // open during simulation after timesteps, append file
        sat_full_file.open(filepath.c_str(), ios_base::app);
    }
    sat_full_file << ntime << " " << saturation_full_domain << " " << temp3 << " " << temp4 << " " << temp1 << " " << temp2 << endl;
    sat_full_file.close();

    if (kper == 0 && domain_wall_status_z_min == 0 && domain_wall_status_z_max == 0) { // non - periodic BC along flow direction(z)
        temp1 = pre_0[i_s0(1 + n_exclude_inlet, 1, 1)] / pore_profile_z[i_s0(1 + n_exclude_inlet, 1, 1)];
        temp2 = pre_0[i_s0(nzGlobal - n_exclude_outlet, 1, 1)] / pore_profile_z[i_s0(nzGlobal - n_exclude_outlet, 1, 1)];
        temp3 = temp1 - temp2;
        filepath = "results/out1.output/pre.dat";
        ofstream pre_file;
        if (ntime0 == 1 && ntime == ntime_monitor) { // open for the first time, new file
            pre_file.open(filepath.c_str(), ios_base::out);
        }
        else { // open during simulation after timesteps, append file
            pre_file.open(filepath.c_str(), ios_base::app);
        }
        pre_file << ntime << " " << temp1 << " " << temp2 << " " << temp3 << endl;
        pre_file.close();
    }

    if (ntime % ntime_monitor_profile == 0) {
        ostringstream flnm;
        flnm << setfill('0') << setw(8) << ntime;
        ostringstream filepath;
        filepath << "results/out1.output/profile/monitor " << flnm.str();
        string fns = filepath.str();
        ofstream monitor_file(fns.c_str(), ios_base::out);
        for (k = 0; k < nzGlobal; k++) {
            monitor_file << T_P(k) << " " << fl1_0[i_s0(k, 1, 1)] << " " << fl2_0[i_s0(k, 1, 1)] << " " << mass1_0[i_s0(k, 1, 1)] / (mass1_0[i_s0(k, 1, 1)] + mass2_0[i_s0(k, 1, 1)]) << " "
                << pre_0[i_s0(k, 1, 1)] / (T_P(pore_profile_z[i_s0(k, 1, 1)]) + eps) << endl;
        }
        monitor_file.close();
    }

    free(fl1_0);
    free(fl2_0);
    free(mass1_0);
    free(mass2_0);
    free(pre_0);

    // check simulation status
    if (isnan(saturation_full_domain) || isnan(ca)) {
        cout << "Simulation failed due to NAN of saturation or capillary number" << endl;
        simulation_end_indicator = 3;
    }
    else if (umax_global > prc(0.5)) {
        cout << "Simulation failed due to maximum velocity larger than 0.5 !" << endl;
        simulation_end_indicator = 3;
    }

    if (steady_state_option == 3) { // steady state based on saturation
        monitor_previous_value = monitor_current_value;  // store previous step
        monitor_current_value = saturation_full_domain;
        temp = fabs(monitor_current_value - monitor_previous_value) / (fabs(monitor_current_value) + eps);
        filepath = "results/out1.output/steady_monitor_saturation_error.dat";
        ofstream ss_monitor_file;
        if (ntime0 == 1 && ntime == ntime_monitor) { // open for the first time, new file
            ss_monitor_file.open(filepath.c_str(), ios_base::out);
        }
        else { // open during simulation after timesteps, append file
            ss_monitor_file.open(filepath.c_str(), ios_base::app);
        }
        ss_monitor_file << ntime << " " << temp << endl;
        ss_monitor_file.close();
        if (temp<convergence_criteria && ntime>ntime_monitor) {
            cout << "Simulation converged based on saturation! Relative error = " << temp << " ; maximum velocity = " << umax_global << endl;
            simulation_end_indicator = 1;   // successfully end simulation
        }
    }

}

//===============================================================================================================================================
//----------------------monitor_multiphase steady flow----------------------
//===============================================================================================================================================
//*******************************monitor_multiphase steady flow - based on phase field * ************************************
void monitor_multiphase_steady_phasefield() {
    int i, j, k;
    int wall_indicator;
    T_P tmp1, tmp2;
    T_P umax, d_phi_max, d_phi_max_global;

    umax = prc(0.);
    d_phi_max = prc(0.);

    compute_macro_vars();

    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                wall_indicator = walls[i_s2(i, j, k)];
                tmp1 = (u[i_s1(i, j, k)] * u[i_s1(i, j, k)] + v[i_s1(i, j, k)] * v[i_s1(i, j, k)] + w[i_s1(i, j, k)] * w[i_s1(i, j, k)]) * (1 - wall_indicator);

                if (umax < tmp1) {
                    umax = tmp1;
                }

                tmp2 = abs(phi[i_s4(i, j, k)] - phi_old[i_s4(i, j, k)]) * (1 - wall_indicator);

                if (d_phi_max < tmp2) {
                    d_phi_max = tmp2;
                }
            }
        }
    }

    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                phi_old[i_s4(i, j, k)] = phi[i_s4(i, j, k)];
            }
        }
    }

    d_phi_max_global = d_phi_max;
    umax_global = umax;

    if (ntime > ntime_relaxation) {
        if (isnan(umax) || isnan(d_phi_max)) {
            cout << "Simulation failed due to NAN!" << endl;
            simulation_end_indicator = 3;
        }
        else {
            umax_global = prc(sqrt)(umax_global);
            string filepath = "results/out1.output/steady_monitor_max_phi_change.dat";
            ofstream ss_monitor_max_file;
            if (ntime0 == 1 && ntime == ntime_monitor) { // open for the first time, new file
                ss_monitor_max_file.open(filepath.c_str(), ios_base::out);
            }
            else { // open during simulation after timesteps, append file
                ss_monitor_max_file.open(filepath.c_str(), ios_base::app);
            }
            ss_monitor_max_file << ntime << " " << d_phi_max_global << " " << umax_global << endl;
            ss_monitor_max_file.close();
            if (umax_global < 0.5) { // maximum velocity smaller than 0.5, otherwise, simulation is considered as failed
                if (d_phi_max_global<convergence_criteria && ntime>ntime_monitor) {
                    cout << "Simulation converged based on maximum local change of phi! Relative error =    " << d_phi_max_global << "; maximum velocity =  " << umax_global << endl;
                    simulation_end_indicator = 1;   // successfully end simulation
                }
            }
            else {
                cout << "Maximum velocity larger than 0.5, simulation failed!!!" << endl;
                simulation_end_indicator = 3;   // stop simulation due to failure
            }
        }
    }

}


// *************************** monitor_multiphase steady flow - based on capillary pressure **********************************
void monitor_multiphase_steady_capillarypressure() {
    int i, j, k, i_w, i_nw, i_w_sum, i_nw_sum;
    int wall_indicator;
    T_P umax, temp, pre_w, pre_nw, pre_w_sum, pre_nw_sum, pc;

    compute_macro_vars();

    umax = prc(0.);

    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                wall_indicator = walls[i_s2(i, j, k)];
                temp = (u[i_s1(i, j, k)] * u[i_s1(i, j, k)] + v[i_s1(i, j, k)] * v[i_s1(i, j, k)] + w[i_s1(i, j, k)] * w[i_s1(i, j, k)]) * (1 - wall_indicator);
                if (umax < temp) {
                    umax = temp;
                }
            }
        }
    }

    pre_w = prc(0.);
    pre_nw = prc(0.);
    i_w = 0;
    i_nw = 0;

    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                if (walls[i_s2(i, j, k)] == 0) {
                    if (phi[i_s4(i, j, k)] < prc(-0.99)) {
                        pre_w = pre_w + rho[i_s1(i, j, k)];
                        i_w = i_w + 1;
                    }
                    if (phi[i_s4(i, j, k)] > prc(0.99)) {
                        pre_nw = pre_nw + rho[i_s1(i, j, k)];
                        i_nw = i_nw + 1;
                    }
                }
            }
        }
    }

    pre_w_sum = pre_w;
    pre_nw_sum = pre_nw;
    i_w_sum = i_w;
    i_nw_sum = i_nw;
    umax_global = umax;

    umax_global = prc(sqrt)(umax_global);
    pre_w = pre_w_sum / (i_w_sum + eps);
    pre_nw = pre_nw_sum / (i_nw_sum + eps);
    pc = (pre_nw - pre_w) / prc(3.);
    monitor_previous_value = monitor_current_value;  // store previous step
    monitor_current_value = pc;
    temp = abs(monitor_current_value - monitor_previous_value) / (abs(monitor_current_value) + eps);

    string filepath = "results/out1.output/steady_monitor_capillary_pressure_error.dat";
    ofstream ss_monitor_cap_file;
    if (ntime0 == 1 && ntime == ntime_monitor) { // open for the first time, new file
        ss_monitor_cap_file.open(filepath.c_str(), ios_base::out);
    }
    else { // open during simulation after timesteps, append file
        ss_monitor_cap_file.open(filepath.c_str(), ios_base::app);
    }
    ss_monitor_cap_file << ntime << " " << temp << " " << pc << " " << pre_w << " " << pre_nw << " " << temp << " " << umax_global << endl;
    ss_monitor_cap_file.close();

    if (isnan(umax) || isnan(pc)) {
        cout << "Simulation failed due to NAN!" << endl;
        simulation_end_indicator = 3;
    }
    else {
        if (umax_global < prc(0.5)) { // maximum velocity smaller than 0.5, otherwise, simulation is considered as failed
            if (temp<convergence_criteria && ntime>ntime_monitor) {
                cout << "Simulation converged based on change of capillary pressure! Relative error =   " << temp << " ; maximum velocity = " << umax_global << endl;
                simulation_end_indicator = 1;   // successfully end simulation
            }
        }
        else {
            cout << "maximum velocity larger than 0.5, simulation failed!!!" << endl;
            simulation_end_indicator = 3;   //stop simulation due to failure
        }
    }



}

//=========================================================================================================================================== =
//----------------------misc subroutines----------------------
//=========================================================================================================================================== =
//*******************************monitor_breakthrough * ************************************
void monitor_breakthrough() {
    int i, j, obs_z, itemp;

    itemp = 0;

    obs_z = nzGlobal - 1;   // observation position
    for (j = 1; j <= nyGlobal; j++) {
        for (i = 1; i <= nxGlobal; i++) {
            if (walls[i_s2(i, j, obs_z)] == 0 && phi[i_s4(i, j, obs_z)] > prc(0.)) {
                itemp = itemp + 1;
            }
        }
    }

    outlet_phase1_sum = itemp;

    if (outlet_phase1_sum >= 1) { // break through detect fluid 1 only
        simulation_end_indicator = 1;
        cout << "Breakthrough point reached! Exiting program!" << endl;
    }

}


//*******************************calculate saturationm*************************************
void cal_saturation() {
    int i, j, k;
    int wall_indicator;
    T_P v1, v2;

    v1 = prc(0.);
    v2 = prc(0.);

    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                wall_indicator = walls[i_s2(i, j, k)];
                v1 = v1 + prc(0.5) * (prc(1.) + phi[i_s4(i, j, k)]) * (1 - wall_indicator);
                v2 = v2 + prc(0.5) * (prc(1.) - phi[i_s4(i, j, k)]) * (1 - wall_indicator);
            }
        }
    }

    vol1_sum = v1;
    vol2_sum = v2;

    saturation_full_domain = vol1_sum / (vol1_sum + vol2_sum + eps);

}
