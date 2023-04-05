#include "externLib.h"
#include "solver_precision.h"
#include "Misc.h"
#include "preprocessor.h"
#include "Module_extern.h"
#include "Fluid_singlephase_extern.h"
#include "Fluid_multiphase_extern.h"
#include "utils.h"
#include "Idx_cpu.h"
#include "IO_multiphase.h"


//======================================================================================================================================= =
//----------------------geometry related----------------------
//======================================================================================================================================= =
//*******************************set walls * *****************************
void set_walls() {

    string flnm, dummy; //file name

    // ~~~~~~~~~~~~~~~~ read wall data ~~~~~~~~~~~~~~~~~~~~
    if (external_geometry_read_cmd == 1) {
        cout << "This simulation uses external geometry data!" << endl;
        string File_path = "input/Geometry_File_Path.txt";
        ifstream geo_file(File_path.c_str());
        if (geo_file.good()) {
            /* Geometry File Name */
            string geo_file_path_;
            READ_STRING(File_path.c_str(), geo_file_path_); // read geometry file path
            geo_file_path = geo_file_path_;
            while (!isdigit(geo_file_path.back())) {
                geo_file_path = geo_file_path.substr(0, geo_file_path.size() - 1);
            }
            geo_file_path += ".dat";

           
            string geo_boundary_file_path_;
            READ_STRING(File_path.c_str(), geo_boundary_file_path_); // read geometry boundary info file path
            geo_boundary_file_path = geo_boundary_file_path_;
            while (!isdigit(geo_boundary_file_path.back())) {
                geo_boundary_file_path = geo_boundary_file_path.substr(0, geo_boundary_file_path.size() - 1);
            }
            geo_boundary_file_path += ".dat";
        }
        else {
            ERROR("Error opening Geometry_File_Path.txt!");
        }

        read_walls();

    }
    else {
        cout << "This simulation does not use external geometry data!" << endl; // duct flow
        nx_sample = nxGlobal;
        ny_sample = nyGlobal;
        nz_sample = nzGlobal;
    }

    // ~~~~~~~~~~~~~~~~ modify geometry or create hard coded geometry ~~~~~~~~~~~~~~~~~~~~
    if (modify_geometry_cmd == 1) {
        modify_geometry();
    }

    // ~~~~~~~~~~~~~~~~specify channel walls for the global wall array ~~~~~~~~~~~~~~~~~~~~
    // walls = 1: solid;  walls = 0: fluid

    for (long long k = 1; k <= nzGlobal; k++) {
        for (long long j = 1; j <= nyGlobal; j++) {
            for (long long i = 1; i <= nxGlobal; i++) {
                if (domain_wall_status_z_min == 1) { walls_global[i_s0(i, j, 1)] = 1; }
                if (domain_wall_status_z_max == 1) { walls_global[i_s0(i, j, nzGlobal)] = 1; }
                if (domain_wall_status_x_min == 1) { walls_global[i_s0(1, j, k)] = 1; }
                if (domain_wall_status_x_max == 1) { walls_global[i_s0(nxGlobal, j, k)] = 1; }
                if (domain_wall_status_y_min == 1) { walls_global[i_s0(i, 1, k)] = 1; }
                if (domain_wall_status_y_max == 1) { walls_global[i_s0(i, nyGlobal, k)] = 1; }
            }
        }
    }

    /* copy the walls data from (walls_global) to (walls) */
    for (long long k = 1; k <= nzGlobal; k++) {
        for (long long j = 1; j <= nyGlobal; j++) {
            for (long long i = 1; i <= nxGlobal; i++) {
                walls[i_s2(i, j, k)] = walls_type[i_s4(i, j, k)] = walls_global[i_s0(i, j, k)];
            }
        }
    }

    // ~~~~~~~~~~~~~~~~ calculate open area at inlet ~~~~~~~~~~~~~~~~~~~~
    long long icount = 0;
    for (long long j = 1; j <= nyGlobal; j++) {
        for (long long i = 1; i <= nxGlobal; i++) {
            if (walls_global[i_s0(i, j, 1)] <= 0) {
                icount++;
            }
        }
    }
    A_xy_effective = T_P(icount);  // inlet area
    cout << "Inlet effective open area =    " << A_xy_effective << endl;

    pore_profile();   // pore profile info along the flow direction z

}
// ************************** modify geometry *******************************
void modify_geometry() {

    long long i, j, k, buffer;
    double xc, yc, zc, r1, r2;
    // walls = 1: solid;  walls = 0: fluid
    // below is a sample code to modify geometry

    // domain center, used as a reference point
    xc = 0.5 * double(nxGlobal + 1);
    yc = 0.5 * double(nyGlobal + 1);
    zc = 0.5 * double(nzGlobal + 1);
    r1 = 0.25 * nyGlobal; // used to creat simple obstacle
    r2 = nyGlobal * 0.5; // tube radius
    buffer = 10;    // inlet outlet reservior


    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {
                if (pow((i - xc), 2) + pow((j - yc), 2) + pow((k - zc), 2) < pow(r1, 2)) {
                    walls_global[i_s0(i, j, k)] = 1;
                }
                if (pow((i - xc), 2) + pow((j - yc), 2) + pow((k - zc), 2) > pow(r2, 2) && k > buffer && k < nzGlobal - buffer + 1) {
                    walls_global[i_s0(i, j, k)] = 1;
                }
            }
        }
    }

    cout << "Internal geometry modified!" << endl;

}

// ***************************** read walls ************************************
void read_walls() {
    long long i, j, k;


    const char* fnc = geo_file_path.c_str();
    FILE* geom_file = fopen(fnc, "r");
    if (geom_file == NULL) { ERROR("Could not open the geometry file !"); }

    if(!fread(&nx_sample, sizeof(long long), 1, geom_file)) { ERROR("Could not load from the geometry file!"); }
    if (!fread(&ny_sample, sizeof(long long), 1, geom_file)) { ERROR("Could not load from the geometry file!"); }
    if(!fread(&nz_sample, sizeof(long long), 1, geom_file)) { ERROR("Could not load from the geometry file!"); }

    cout << "Porous media sample size: nx = " << nx_sample << ", ny = " << ny_sample << ", nz = " << nz_sample << endl;

    if (nxGlobal < nx_sample || nyGlobal < ny_sample || nzGlobal < nz_sample) {
        ERROR("Error! Domain size is smaller than porous media sample size! Exiting program!");
    }
    else {
        char* geo_temp = new char[nx_sample * ny_sample * nz_sample]; // temporary bool array as the file is in the integer*1 format (size = 1 byte)
        if(!fread(geo_temp, sizeof(char), nx_sample * ny_sample * nz_sample, geom_file)) { ERROR("Could not load from the geometry file!"); }

        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    walls_global[i_s0(i, j, k)] = geo_temp[(i - 1) + nyGlobal * ((j - 1) + nxGlobal * (k - 1))];
                }
            }
        }

        delete[] geo_temp;
    }
    fclose(geom_file);

    if (domain_wall_status_x_max == 1 && domain_wall_status_y_max == 1) {
        for (k = 1; k <= nzGlobal; k++) {
            for (j = 1; j <= nyGlobal; j++) {
                for (i = 1; i <= nxGlobal; i++) {
                    if (j >= ny_sample || i >= nx_sample) {
                        walls_global[i_s0(i, j, k)] = 1;// pad with solid walls
                    }
                }
            }
        }
    }

}

// *******************************pore profile*************************************
void pore_profile() {

    pore_profile_z = (int*)calloc(nzGlobal, sizeof(int));

    for (long long k = 1; k <= nzGlobal; k++) {
        pore_profile_z[k - 1] = 0;
        for (long long j = 1; j <= nyGlobal; j++) {
            for (long long i = 1; i <= nxGlobal; i++) {
                if (walls[i_s2(i, j, k)] <= 0) {
                    pore_profile_z[k - 1] += 1;
                }
            }
        }
    }
    pore_sum = 0;
    for (int k = 1; k <= nzGlobal; k++) {
        pore_sum += pore_profile_z[k - 1];
    }
    for (int k = 1 + n_exclude_inlet; k <= nzGlobal - n_exclude_outlet; k++) {
        pore_sum_effective += pore_profile_z[k - 1];
    }

}

//===================================================================================================================================== =
//----------------------compute macroscopic varaibles from PDFs----------------------
//===================================================================================================================================== =
void compute_macro_vars() { // u,v,w,rho   (phi already known)
    long long i, j, k;
    int wall_indicator;
    T_P ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12, ft13, ft14, ft15, ft16, ft17, ft18, fx, fy, fz, tmp;

    for (k = 1; k <= nzGlobal; k++) {
        for (j = 1; j <= nyGlobal; j++) {
            for (i = 1; i <= nxGlobal; i++) {

                wall_indicator = walls[i_s2(i, j, k)];

                ft0 = pdf[i_f1(i, j, k, 0, 0)] + pdf[i_f1(i, j, k, 0, 1)];
                ft1 = pdf[i_f1(i, j, k, 1, 0)] + pdf[i_f1(i, j, k, 1, 1)];
                ft2 = pdf[i_f1(i, j, k, 2, 0)] + pdf[i_f1(i, j, k, 2, 1)];
                ft3 = pdf[i_f1(i, j, k, 3, 0)] + pdf[i_f1(i, j, k, 3, 1)];
                ft4 = pdf[i_f1(i, j, k, 4, 0)] + pdf[i_f1(i, j, k, 4, 1)];
                ft5 = pdf[i_f1(i, j, k, 5, 0)] + pdf[i_f1(i, j, k, 5, 1)];
                ft6 = pdf[i_f1(i, j, k, 6, 0)] + pdf[i_f1(i, j, k, 6, 1)];
                ft7 = pdf[i_f1(i, j, k, 7, 0)] + pdf[i_f1(i, j, k, 7, 1)];
                ft8 = pdf[i_f1(i, j, k, 8, 0)] + pdf[i_f1(i, j, k, 8, 1)];
                ft9 = pdf[i_f1(i, j, k, 9, 0)] + pdf[i_f1(i, j, k, 9, 1)];
                ft10 = pdf[i_f1(i, j, k, 10, 0)] + pdf[i_f1(i, j, k, 10, 1)];
                ft11 = pdf[i_f1(i, j, k, 11, 0)] + pdf[i_f1(i, j, k, 11, 1)];
                ft12 = pdf[i_f1(i, j, k, 12, 0)] + pdf[i_f1(i, j, k, 12, 1)];
                ft13 = pdf[i_f1(i, j, k, 13, 0)] + pdf[i_f1(i, j, k, 13, 1)];
                ft14 = pdf[i_f1(i, j, k, 14, 0)] + pdf[i_f1(i, j, k, 14, 1)];
                ft15 = pdf[i_f1(i, j, k, 15, 0)] + pdf[i_f1(i, j, k, 15, 1)];
                ft16 = pdf[i_f1(i, j, k, 16, 0)] + pdf[i_f1(i, j, k, 16, 1)];
                ft17 = pdf[i_f1(i, j, k, 17, 0)] + pdf[i_f1(i, j, k, 17, 1)];
                ft18 = pdf[i_f1(i, j, k, 18, 0)] + pdf[i_f1(i, j, k, 18, 1)];
                


                rho[i_s1(i, j, k)] = (ft0 + ft1 + ft2 + ft3 + ft4 + ft5 + ft6 + ft7 + ft8 + ft9 + ft10 + ft11 + ft12 + ft13 + ft14 + ft15 + ft16 + ft17 + ft18) * (1 - wall_indicator);

                tmp = prc(0.5) * lbm_gamma * curv[i_s1(i, j, k)] * c_norm[i_s2(i, j, k)];
                fx = tmp * cn_x[i_s2(i, j, k)];
                fy = tmp * cn_y[i_s2(i, j, k)];
                fz = tmp * cn_z[i_s2(i, j, k)] + force_z;

                //here "- 0.5fx" is due to that PDFs are after even step, which is post collision before streaming
                //to use the post collision PDFs to calculate the velocities, one must substruct the forcing terms applied during collision step
                //thus the fomular is u = f...f + 0.5fx - fx = f...f - 0.5fx
                u[i_s1(i, j, k)] = (ft1 - ft2 + ft7 - ft8 + ft9 - ft10 + ft11 - ft12 + ft13 - ft14 - prc(0.5) * fx) * (1 - wall_indicator);
                v[i_s1(i, j, k)] = (ft3 - ft4 + ft7 + ft8 - ft9 - ft10 + ft15 - ft16 + ft17 - ft18 - prc(0.5) * fy) * (1 - wall_indicator);
                w[i_s1(i, j, k)] = (ft5 - ft6 + ft11 + ft12 - ft13 - ft14 + ft15 + ft16 - ft17 - ft18 - prc(0.5) * fz) * (1 - wall_indicator);

                phi[i_s4(i, j, k)] = prc(0.) * wall_indicator + phi[i_s4(i, j, k)] * (1 - wall_indicator);

            }
        }
    }
}

// ************* change inlet fluid phase *********************************************************
void change_inlet_fluid_phase() {

    long long i, j, k, z;

    if (change_inlet_fluid_phase_cmd == 1) {
        for (k = 1 - 1; k <= nzGlobal + 1; k++) {
            for (j = 1 - 1; j <= nyGlobal + 1; j++) {
                for (i = 1 - 1; i <= nxGlobal - 1; i++) {
                    z = k;
                    if (z <= interface_z0) {
                        pdf[i_f1(i, j, k, 0, 0)] += pdf[i_f1(i, j, k, 0, 1)];
                        pdf[i_f1(i, j, k, 1, 0)] += pdf[i_f1(i, j, k, 1, 1)];
                        pdf[i_f1(i, j, k, 2, 0)] += pdf[i_f1(i, j, k, 2, 1)];
                        pdf[i_f1(i, j, k, 3, 0)] += pdf[i_f1(i, j, k, 3, 1)];
                        pdf[i_f1(i, j, k, 4, 0)] += pdf[i_f1(i, j, k, 4, 1)];
                        pdf[i_f1(i, j, k, 5, 0)] += pdf[i_f1(i, j, k, 5, 1)];
                        pdf[i_f1(i, j, k, 6, 0)] += pdf[i_f1(i, j, k, 6, 1)];
                        pdf[i_f1(i, j, k, 7, 0)] += pdf[i_f1(i, j, k, 7, 1)];
                        pdf[i_f1(i, j, k, 8, 0)] += pdf[i_f1(i, j, k, 8, 1)];
                        pdf[i_f1(i, j, k, 9, 0)] += pdf[i_f1(i, j, k, 9, 1)];
                        pdf[i_f1(i, j, k, 10, 0)] += pdf[i_f1(i, j, k, 10, 1)];
                        pdf[i_f1(i, j, k, 11, 0)] += pdf[i_f1(i, j, k, 11, 1)];
                        pdf[i_f1(i, j, k, 12, 0)] += pdf[i_f1(i, j, k, 12, 1)];
                        pdf[i_f1(i, j, k, 13, 0)] += pdf[i_f1(i, j, k, 13, 1)];
                        pdf[i_f1(i, j, k, 14, 0)] += pdf[i_f1(i, j, k, 14, 1)];
                        pdf[i_f1(i, j, k, 15, 0)] += pdf[i_f1(i, j, k, 15, 1)];
                        pdf[i_f1(i, j, k, 16, 0)] += pdf[i_f1(i, j, k, 16, 1)];
                        pdf[i_f1(i, j, k, 17, 0)] += pdf[i_f1(i, j, k, 17, 1)];
                        pdf[i_f1(i, j, k, 18, 0)] += pdf[i_f1(i, j, k, 18, 1)];

                        pdf[i_f1(i, j, k, 0, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 1, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 2, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 3, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 4, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 5, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 6, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 7, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 8, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 9, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 10, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 11, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 12, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 13, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 14, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 15, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 16, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 17, 1)] = prc(0.);
                        pdf[i_f1(i, j, k, 18, 1)] = prc(0.);

                    }
                }
            }
        }
    }
    else if (change_inlet_fluid_phase_cmd == 2) {
        for (k = 1 - 1; k <= nzGlobal + 1; k++) {
            for (j = 1 - 1; j <= nyGlobal + 1; j++) {
                for (i = 1 - 1; i <= nxGlobal - 1; i++) {
                    z = k;
                    if (z <= interface_z0) {

                        pdf[i_f1(i, j, k, 0, 1)] += pdf[i_f1(i, j, k, 0, 0)];
                        pdf[i_f1(i, j, k, 1, 1)] += pdf[i_f1(i, j, k, 1, 0)];
                        pdf[i_f1(i, j, k, 2, 1)] += pdf[i_f1(i, j, k, 2, 0)];
                        pdf[i_f1(i, j, k, 3, 1)] += pdf[i_f1(i, j, k, 3, 0)];
                        pdf[i_f1(i, j, k, 4, 1)] += pdf[i_f1(i, j, k, 4, 0)];
                        pdf[i_f1(i, j, k, 5, 1)] += pdf[i_f1(i, j, k, 5, 0)];
                        pdf[i_f1(i, j, k, 6, 1)] += pdf[i_f1(i, j, k, 6, 0)];
                        pdf[i_f1(i, j, k, 7, 1)] += pdf[i_f1(i, j, k, 7, 0)];
                        pdf[i_f1(i, j, k, 8, 1)] += pdf[i_f1(i, j, k, 8, 0)];
                        pdf[i_f1(i, j, k, 9, 1)] += pdf[i_f1(i, j, k, 9, 0)];
                        pdf[i_f1(i, j, k, 10, 1)] += pdf[i_f1(i, j, k, 10, 0)];
                        pdf[i_f1(i, j, k, 11, 1)] += pdf[i_f1(i, j, k, 11, 0)];
                        pdf[i_f1(i, j, k, 12, 1)] += pdf[i_f1(i, j, k, 12, 0)];
                        pdf[i_f1(i, j, k, 13, 1)] += pdf[i_f1(i, j, k, 13, 0)];
                        pdf[i_f1(i, j, k, 14, 1)] += pdf[i_f1(i, j, k, 14, 0)];
                        pdf[i_f1(i, j, k, 15, 1)] += pdf[i_f1(i, j, k, 15, 0)];
                        pdf[i_f1(i, j, k, 16, 1)] += pdf[i_f1(i, j, k, 16, 0)];
                        pdf[i_f1(i, j, k, 17, 1)] += pdf[i_f1(i, j, k, 17, 0)];
                        pdf[i_f1(i, j, k, 18, 1)] += pdf[i_f1(i, j, k, 18, 0)];

                        pdf[i_f1(i, j, k, 0, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 1, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 2, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 3, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 4, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 5, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 6, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 7, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 8, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 9, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 10, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 11, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 12, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 13, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 14, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 15, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 16, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 17, 0)] = prc(0.);
                        pdf[i_f1(i, j, k, 18, 0)] = prc(0.);
                    }
                }
            }
        }
    }

}

// *************inlet velocity - analytical solution**********************************************************
void inlet_vel_profile_rectangular(T_P vel_avg, int num_terms) {
    T_P a, b, xx, yy, tmp1, tmp2, tmp3;
    long long n, i, j, x, y;

    a = prc(0.5) * la_x;
    b = prc(0.5) * la_y;

    tmp1 = prc(0.);
    for (n = 1; n <= num_terms; n += 2) {
        tmp1 += (prc(tanh)(prc(0.5) * T_P(n) * Pi * b / a)) / prc(pow)(T_P(n), 5);
    }
    tmp2 = prc(1.) - prc(192.) / prc(pow)(Pi, 5) * (a / b) * tmp1;
    tmp2 = prc(-3.) * vel_avg / (tmp2 * prc(pow)(a, 2));
    for (j = 1; j <= nyGlobal; j++) {
        for (i = 1; i <= nxGlobal; i++) {
            x = i;
            y = j;
            if (x > 1 && x < nxGlobal && y > 1 && y < nyGlobal) {
                xx = x - prc(1.5) - a;
                yy = y - prc(1.5) - b;
                tmp3 = prc(0.);
                for (n = 1; n <= num_terms; n += 2) {
                    tmp3 += prc(pow)(prc(-1.), prc(0.5) * T_P((n - 1))) * prc(cos)(prc(0.5) * T_P(n) * Pi * xx / a) / prc(pow)(T_P(n), 3)

                        * (prc(1.) - (prc(exp)(prc(0.5) * T_P(n) * Pi * (yy - b) / a) + prc(exp)(prc(0.5) * T_P(n) * Pi * (-yy - b) / a)) / (prc(1.) + prc(exp)(prc(0.5) * T_P(n) * Pi * (-b - b) / a)));

                }
                W_in[i_s1(x, y, 0)] = tmp3 * (prc(-16.) * tmp2 * prc(pow)(a, 2) * prc(pow)(Pi, -3));
            }
        }
    }

}



