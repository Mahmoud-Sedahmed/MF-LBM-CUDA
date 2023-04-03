//============================================================================= =
//PROGRAM: MF-LBM - C++/CUDA version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~ April 1st, 2023  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
// This is a C++/CUDA version of the MF-LBM solver.
// Author: Mahmoud Sedahmed (mahmoud.sedahmed@alexu.edu.eg)
// Github repository:
// The original MF-LBM solver - Fortran/MPI/OpenMP/OpenACC version by Authors: Dr. Yu Chen, Dr. Qinjun Kang
// Github repository: https://github.com/lanl/MF-LBM

#include "externLib.h"						/* C++/CUDA standard libraries */
#include "solver_precision.h"				/* Solver precision */
#include "preprocessor.h"					/* MRT parameters set */
#include "utils.h"							/* Helper functions (ERROR, read)*/
#include "Module_extern.h"					
#include "Module.h"							
#include "Fluid_singlephase_extern.h"		
#include "Fluid_singlephase.h"				
#include "Fluid_multiphase_extern.h"		
#include "Fluid_multiphase.h"				
#include "Init_multiphase.h"				
#include "Misc.h"
#include "Phase_gradient.h"
#include "Monitor.h"
#include "IO_multiphase.h"
#include "Init_multiphase_GPU.h"
#include "main_iteration_GPU.h"



int main(int argc, char* argv[]) {
	cout << "==============================================================================" << endl;
	cout << "MF-LBM solver - CUDA/C++ version" << endl;
	cout << "Author: M. Sedahmed" << endl;
	cout << "Github repository: " << endl;
	cout << "Original MF-LBM solver - Fortran/MPI/OpenMP/OpenACC version by Authors: Dr. Yu Chen, Dr. Qinjun Kang" << endl;
	cout << "Github repository: (https://github.com/lanl/MF-LBM)" << endl;
	cout << "==============================================================================" << endl;

	// indicator used to save extra backup checkpoint(pdf) data
	int save_checkpoint_data_indicator, save_2rd_checkpoint_data_indicator;
	int counter_checkpoint_save, counter_2rd_checkpoint_save;

	//################################################################################################################################
	//														 Preparation
	//################################################################################################################################

	simulation_end_indicator = 0;
	save_checkpoint_data_indicator = 0;          // default 0, saving data 1, after saving data 0
	save_2rd_checkpoint_data_indicator = 0;       // default 0, saving data 1, after saving data 0

	// initial value 1; after each checkpoint data saving, counter_checkpoint_save = counter_checkpoint_save + 1
	counter_checkpoint_save = 1;
	// initial value 1; after each checkpoint data saving, counter_2rd_checkpoint_save = counter_2rd_checkpoint_save + 1
	counter_2rd_checkpoint_save = 1;

	relaxation = prc(1.);   // no relaxation if 1; relaxation feature temporary disabled, should keep 1

	cout << " " << endl;

#if (PRECISION == SINGLE_PRECISION)
	cout << "Solver precision: Single precision" << endl;

#elif (PRECISION == DOUBLE_PRECISION)
	cout << "Solver precision: Double precision" << endl;
#endif

	cout << " " << endl;
	cout << "***************************** Initialization **********************************" << endl;

	initialization_basic_multi();
	if (job_status == "new_simulation") {
		initialization_new_multi();
	}
	else if (job_status == "continue_simulation") {
		initialization_old_multi();
	}
	else {
		ERROR("Error in job_status file content (new_simulation / contniue_simulation)! Existing program!");
	}

	if (change_inlet_fluid_phase_cmd != 0) {
		change_inlet_fluid_phase();
		cout << "Inlet fluid phase was changed (1 to fluid1; 2 to fluid2): " << change_inlet_fluid_phase_cmd << endl;
	}

	cout << "************************** Initialization ends ********************************" << endl;

	//t_all_sum = prc(0.);
	ntime = ntime0;

	color_gradient();

	cal_saturation();

	cout << "***************************** Initialization - GPU **********************************" << endl;
	initialization_GPU();
	/* copy constant data to GPU */
	copyConstantData();
	cout << "************************** Initialization ends - GPU ********************************" << endl;


	if (job_status == "new_simulation") {
		saturation_old = saturation;
		if (benchmark_cmd == 0) {
			if (extreme_large_sim_cmd == 0) { // initial distribution
				// vtk_type 1: full flow field info for detailed analysis
				// vtk_type 2 : phase field info, with single precision to save space
				// vtk_type 3 : force vectors from the CSF model
				VTK_legacy_writer_3D(ntime, 2);
				VTK_walls_bin();
			}
			else {
				ERROR("Extreme large simulation domain (extreme_large_sim_cmd = 1) is not supported yet!");
				// parallel I / O, distributed files, require post processing
				// should be replaced by parallel version of VTK in the future
				//save_phi(ntime);
			}
		}
	}
	else if (job_status == "continue_simulation") {
		saturation_old = prc(-1.);    // a negative value, in order to avoid convergence check at the first step after reading old data
		 // override the loaded rho_in value and use new input value (pressure BC)
		if (kper == 0 && domain_wall_status_z_min == 0 && domain_wall_status_z_max == 0 && inlet_BC == 2 && rho_in_new) {
			p_gradient = -force_z0 / prc(3.);        // pressure gradient
			if (rho_out_BC) {
				rho_out = prc(1.) - p_gradient * nzGlobal;
			}
			else {
				rho_in = rho_out - p_gradient * nzGlobal;
			}
			
		}
	}

	cout << "Initial saturation: " << saturation_full_domain << endl;

	//################################################################################################################################
	//													Main loop Starts
	//################################################################################################################################
	cout << "************************** Entering main loop *********************************" << endl;
	chrono::steady_clock::time_point ts_console = chrono::steady_clock::now(); // will be reset every specific interval
	chrono::steady_clock::time_point ts2_console = chrono::steady_clock::now(); // mark the start of the loop
	chrono::steady_clock::time_point ts3_console = chrono::steady_clock::now(); // will be reset every specific interval
	for (ntime = ntime0; ntime <= ntime0 + ntime_max; ntime++) {
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main kernel%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		main_iteration_kernel_GPU();
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main kernel%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// ---------- - computation time---------- -
		if (ntime % ntime_clock_sum == 0) {
			string filepath = "results/out1.output/time.dat";
			ofstream time_dat;
			if (ntime0 == 1 && ntime == ntime_clock_sum) { // open for the first time, new file
				time_dat.open(filepath.c_str(), ios_base::out);
			}
			else { // open during simulation after timesteps, append file
				time_dat.open(filepath.c_str(), ios_base::app);
			}
			if (time_dat.good()) {
				chrono::steady_clock::time_point tend1 = chrono::steady_clock::now();
				chrono::steady_clock::duration td_console = tend1 - ts_console; // result in nanoseconds
				double duration_console = td_console.count() * 1e-9; // result in seconds
				time_dat << ntime << " " << duration_console << " sec" << endl;
				time_dat.close();
				ts_console = chrono::steady_clock::now();
			}
			else {
				ERROR("Could not open results/out1.output/time.dat");
			}
		}

		// ---------- - monitors---------- -
		if (ntime % ntime_monitor == 0) {
			if (steady_state_option == 0 || steady_state_option == 3) {
				// displacement simulation : finished based on saturation reaches steady state or injected volume
				monitor();
				if (breakthrough_check == 1) {
					monitor_breakthrough();
				}
			}
			else if (steady_state_option == 1) { // steady state simulation based on capillary pressure
				monitor_multiphase_steady_capillarypressure();
			}
			else if (steady_state_option == 2) {// steady state simulation based on phase field
				monitor_multiphase_steady_phasefield();
			}
		}

		// ---------- - simulation progress - time steps---------- -
		if (ntime % ntime_display_steps == 0) {
			chrono::steady_clock::time_point te_console = chrono::steady_clock::now();
			chrono::steady_clock::duration td_console = te_console - ts3_console;
			double code_speed = static_cast<double>(nxGlobal * nyGlobal * nzGlobal) / static_cast<double>(td_console.count() * 1e-3); // 1e-3 >>> since the count in nano (1e-9) and result in (million) (1e6)
			code_speed *= static_cast<double>(ntime_display_steps);
			cout << "ntime =	" << ntime << endl;
			cout << "simulation speed: " << code_speed << " MLUPS" << endl;
			ts3_console = chrono::steady_clock::now();
		}

		// ---------- - phase field(vtk) files for visualizationand other analysis---------- -
		if (ntime % ntime_animation == 0) {
			if (extreme_large_sim_cmd == 0) {
				VTK_legacy_writer_3D(ntime, 2);
			}
			else {
				ERROR("Extreme large simulation domain (extreme_large_sim_cmd = 1) is not supported yet!");
				// parallel I / O, distributed files, require post processing
				// should be replaced by parallel version of VTK in the future
				//save_phi(ntime);
			}
		}

		// ---------- - full flow field(VTK) files for further analysis---------- -
		if (ntime % ntime_visual == 0) {
			if (extreme_large_sim_cmd == 0) {
				VTK_legacy_writer_3D(ntime, 1);
			}
			else {
				ERROR("Extreme large simulation domain (extreme_large_sim_cmd = 1) is not supported yet!");
				//save_macro(ntime);    // parallel I / O, distributed files, require post processing
			}
		}

		// ~~~~~~~~~~~~~~~~~~SAVE checkpoint PDF DATA for restarting simulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if (ntime % ntime_clock_sum == 0) { // frequency to check checkpoint data saving

			chrono::steady_clock::time_point tend2 = chrono::steady_clock::now();
			chrono::steady_clock::duration td2_console = tend2 - ts2_console; // result in nanoseconds
			double duration_console = td2_console.count() * 1e-9; // result in seconds
			duration_console *= prc(2.77777777777e-4);   // second to hour  1 / 3600

			cout << "simulation has run	" << duration_console << "	hours" << endl;
			if (duration_console >= simulation_duration_timer) {
				cout << "Time to save checkpoint data and exit the program!" << endl;
				simulation_end_indicator = 2;
			}
			if (duration_console >= counter_checkpoint_save * checkpoint_save_timer) {
				cout << "Time to save checkpoint data!" << endl;
				save_checkpoint_data_indicator = 1;
				counter_checkpoint_save = counter_checkpoint_save + 1;
			}
			if (duration_console >= counter_2rd_checkpoint_save * checkpoint_2rd_save_timer) {
				cout << "Time to save secondary checkpoint data!" << endl;
				save_2rd_checkpoint_data_indicator = 1;
				counter_2rd_checkpoint_save = counter_2rd_checkpoint_save + 1;
			}


			if (save_checkpoint_data_indicator == 1) {
				save_checkpoint(0);    // save pdf to the default location when option is 0
				save_checkpoint_data_indicator = 0;   //reset status
			}
			if (save_2rd_checkpoint_data_indicator == 1) {
				save_checkpoint(1);    // save pdf to the secondary backup location when option is 1
				save_2rd_checkpoint_data_indicator = 0;   //reset status
			}

		}
		// ~~~~~~~~~~~~~~~~~~SAVE checkpoint PDF DATA for restarting simulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if (simulation_end_indicator > 0) { break; }

	}
	cout << "************************** Exiting main iteration *********************************" << endl;

	/* print total simulation speed */
	chrono::steady_clock::time_point te2_console = chrono::steady_clock::now(); // mark the end of the loop
	chrono::steady_clock::duration td2_e_console = te2_console - ts2_console; // result in nanoseconds
	double duration_console_total = td2_e_console.count() * 1e-9; // result in seconds
	double code_speed = static_cast<double>(nxGlobal * nyGlobal * nzGlobal) * static_cast<double>((ntime - ntime0) - 1) / (1e6 * duration_console_total);
	cout << "Code speed:\t" << code_speed << " MLUPS" << endl;

	if (simulation_end_indicator == 0) {
		ntime = ntime - 1;  //dial back ntime
		save_checkpoint(0);
		if (extreme_large_sim_cmd == 0) {
			VTK_legacy_writer_3D(ntime, 2);
			VTK_legacy_writer_3D(ntime, 1);
		}
		else {
			ERROR("Extreme large simulation domain (extreme_large_sim_cmd = 1) is not supported yet!");
			//save_phi(ntime);   //parallel I / O, distributed files, require post processing
		}

		if (simulation_end_indicator == 0) { cout << "Simulation ended after	" << ntime << " iterations which reached the maximum time step!" << endl; }
		string filepath = "./job_status.txt";
		ofstream job_stat(filepath.c_str(), ios_base::out);
		if (job_stat.good()) {
			job_stat << "simulation_reached_max_step" << endl;
			job_stat.close();
		}
		else {
			ERROR("Could not open ./job_status.txt");
		}


	}
	else if (simulation_end_indicator == 1) {
		save_checkpoint(0);
		if (extreme_large_sim_cmd == 0) {
			VTK_legacy_writer_3D(ntime, 2);
			VTK_legacy_writer_3D(ntime, 1);
		}
		else {
			ERROR("Extreme large simulation domain (extreme_large_sim_cmd = 1) is not supported yet!");
			//save_phi(ntime);   // parallel I / O, distributed files, require post processing
		}

		cout << "Simulation ended successfully after	" << ntime << " iterations!" << endl;
		string filepath = "./job_status.txt";
		ofstream job_stat(filepath.c_str(), ios_base::out);
		if (job_stat.good()) {
			job_stat << "simulation_done" << endl;
			job_stat.close();
		}
		else {
			ERROR("Could not open ./job_status.txt");
		}

	}
	else if (simulation_end_indicator == 2) {
		save_checkpoint(0);

		cout << "Simulation data saved but not finished after	" << ntime << " iterations!" << endl;
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
	else if (simulation_end_indicator == 3) {
		if (extreme_large_sim_cmd == 0) {
			VTK_legacy_writer_3D(ntime, 1);
		}
		else {
			ERROR("Extreme large simulation domain (extreme_large_sim_cmd = 1) is not supported yet!");
			//save_phi(ntime);   // parallel I / O, distributed files, require post processing
		}

		cout << "Simulation failed after	" << ntime << " iterations!" << endl;
		string filepath = "./job_status.txt";
		ofstream job_stat(filepath.c_str(), ios_base::out);
		if (job_stat.good()) {
			job_stat << "simulation_failed" << endl;
			job_stat.close();

		}
		else {
			ERROR("Could not open ./job_status.txt");
		}

	}

	MemAllocate_geometry(2);
	MemAllocate_multi(2);

	MemAllocate_geometry_GPU(2);
	MemAllocate_multi_GPU(2);

	cout << endl << "Code Finished " << endl;
	return 0;
}