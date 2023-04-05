#include "externLib.h"
#include "solver_precision.h"
#include "Geometry_preprocessing.h"
#include "preprocessor.h"
#include "Module_extern.h"
#include "Fluid_singlephase_extern.h"
#include "Fluid_multiphase_extern.h"
#include "utils.h"
#include "Idx_cpu.h"
#include "IO_multiphase.h"

#define NXG_TMP_10	(nxGlobal + 20)
#define NYG_TMP_10	(nyGlobal + 20)
#define NZG_TMP_10	(nzGlobal + 20)

#define XG_TMP_10(x)	(x + 9)				// (x - 1 + 10): -1 for starting indexing from 1, +10 for considering ten ghost layers
#define YG_TMP_10(y)	(y + 9)
#define ZG_TMP_10(z)	(z + 9)

#define i_s_TMP_10(x, y, z)			(XG_TMP_10(x) + NXG_TMP_10 * (YG_TMP_10(y) + NYG_TMP_10 * ZG_TMP_10(z)))


// =====================================================================================================================================
// ----------------------geometry_preprocessing new ----------------------
// process the geometry before the start of the main iteration : normal directions of the solid surface; extrapolation weights from
// different directions not recommended for relatively large domain - the processing time before each new or old simulation could be
// too long(not paralllelized on CPU for the GPU version code); also not optimal for Xeon Phi as the on - package memory is limited
// =====================================================================================================================================
void geometry_preprocessing_new() {

    int iex[27] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1 };
    int iey[27] = { 0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1 };
    int iez[27] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1 };
    T_P we[4] = { prc(8.) / prc(27.), prc(2.) / prc(27.), prc(1.) / prc(54.), prc(1.) / prc(216.) };

    int i_max_iteration = 4; // smoothing iterations

    long long tmp_ghost_layers = 10;  // (6 + 4 , 6: used for smoothing operations , 4: additional layers used for solid/fluid boundary nodes that outside the domain)

    long long tmp_nx_gh = nxGlobal + 2 * tmp_ghost_layers;
    long long tmp_ny_gh = nyGlobal + 2 * tmp_ghost_layers;
    long long tmp_nz_gh = nzGlobal + 2 * tmp_ghost_layers;
    long long tmp_num_cells_gh = tmp_nx_gh * tmp_ny_gh * tmp_nz_gh;

    T_P* walls_smooth_1 = (T_P*)calloc(tmp_num_cells_gh, sizeof(T_P));
    T_P* walls_smooth_2 = (T_P*)calloc(tmp_num_cells_gh, sizeof(T_P));
    int* walls_temp1 = (int*)calloc(tmp_num_cells_gh, sizeof(int));

    // copy walls information from walls to walls_temp1
    for (long long k = 1; k <= nzGlobal; k++) {
        for (long long j = 1; j <= nyGlobal; j++) {
            for (long long i = 1; i <= nxGlobal; i++) {
                walls_temp1[i_s_TMP_10(i, j, k)] = walls[i_s2(i, j, k)];
            }
        }
    }

    // consider periodic/non-periodic BCs in ghost layers
    if (kper == 0) {// z direction, not periodic
        for (long long j = 1; j <= nyGlobal; j++) {
            for (long long i = 1; i <= nxGlobal; i++) {
                for (long long k = 1 - tmp_ghost_layers; k <= 0; k++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(i, j, 1)];
                }
                for (long long k = nzGlobal + 1; k <= nzGlobal + tmp_ghost_layers; k++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(i, j, nzGlobal)];
                }
            }
        }
    }
    else { // z direction, periodic
        for (long long j = 1; j <= nyGlobal; j++) {
            for (long long i = 1; i <= nxGlobal; i++) {
                for (long long k = 1 - tmp_ghost_layers; k <= 0; k++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(i, j, nzGlobal + k)]; // nzglobal-ghost_layers+1:nzglobal
                }
                for (long long k = nzGlobal + 1; k <= nzGlobal + tmp_ghost_layers; k++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(i, j, k - nzGlobal)];
                }
            }
        }
    }

    if (jper == 0) { // y direction, not periodic 
        for (long long k = 1 - tmp_ghost_layers; k <= nzGlobal + tmp_ghost_layers; k++) {
            for (long long i = 1; i <= nxGlobal; i++) {
                for (long long j = 1 - tmp_ghost_layers; j <= 0; j++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(i, 1, k)];
                }
                for (long long j = nyGlobal + 1; j <= nyGlobal + tmp_ghost_layers; j++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(i, nyGlobal, k)];
                }
            }
        }
    }
    else { // y direction,periodic 
        for (long long k = 1 - tmp_ghost_layers; k <= nzGlobal + tmp_ghost_layers; k++) {
            for (long long i = 1; i <= nxGlobal; i++) {
                for (long long j = 1 - tmp_ghost_layers; j <= 0; j++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(i, nyGlobal + j, k)];
                }
                for (long long j = nyGlobal + 1; j <= nyGlobal + tmp_ghost_layers; j++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(i, j - nyGlobal, k)];
                }
            }
        }
    }

    if (iper == 0) { // x direction, not periodic
        for (long long k = 1 - tmp_ghost_layers; k <= nzGlobal + tmp_ghost_layers; k++) {
            for (long long j = 1 - tmp_ghost_layers; j <= nyGlobal + tmp_ghost_layers; j++) {
                for (long long i = 1 - tmp_ghost_layers; i <= 0; i++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(1, j, k)];
                }
                for (long long i = nxGlobal + 1; i <= nxGlobal + tmp_ghost_layers; i++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(nxGlobal, j, k)];
                }
            }
        }
    }
    else { // x direction, periodic
        for (long long k = 1 - tmp_ghost_layers; k <= nzGlobal + tmp_ghost_layers; k++) {
            for (long long j = 1 - tmp_ghost_layers; j <= nyGlobal + tmp_ghost_layers; j++) {
                for (long long i = 1 - tmp_ghost_layers; i <= 0; i++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(nxGlobal + i, j, k)];
                }
                for (long long i = nxGlobal + 1; i <= nxGlobal + tmp_ghost_layers; i++) {
                    walls_temp1[i_s_TMP_10(i, j, k)] = walls_temp1[i_s_TMP_10(i - nxGlobal, j, k)];
                }
            }
        }
    }

    // copy walls information (including ghost layers) from (walls_temp1) to (walls)
    for (long long k = 1 - 2; k <= nzGlobal + 2; k++) {
        for (long long j = 1 - 2; j <= nyGlobal + 2; j++) {
            for (long long i = 1 - 2; i <= nxGlobal + 2; i++) {
                walls[i_s2(i, j, k)] = walls_temp1[i_s_TMP_10(i, j, k)];
            }
        }
    }

    // copy data from walls_temp1 to walls_smooth_1 and walls_smooth_2 (including ghost layers)
    for (long long k = 1 - tmp_ghost_layers; k <= nzGlobal + tmp_ghost_layers; k++) {
        for (long long j = 1 - tmp_ghost_layers; j <= nyGlobal + tmp_ghost_layers; j++) {
            for (long long i = 1 - tmp_ghost_layers; i <= nxGlobal + tmp_ghost_layers; i++) {
                walls_smooth_1[i_s_TMP_10(i, j, k)] = T_P(walls_temp1[i_s_TMP_10(i, j, k)]);
                walls_smooth_2[i_s_TMP_10(i, j, k)] = T_P(walls_temp1[i_s_TMP_10(i, j, k)]);
            }
        }
    }

    // extend wall information: change walls_temp1 values to (2) for solid boundary nodes or (-1) for fluid boundary nodes
    for (long long k = 1 - tmp_ghost_layers + 1; k <= nzGlobal + tmp_ghost_layers - 1; k++) {
        for (long long j = 1 - tmp_ghost_layers + 1; j <= nyGlobal + tmp_ghost_layers - 1; j++) {
            for (long long i = 1 - tmp_ghost_layers + 1; i <= nxGlobal + tmp_ghost_layers - 1; i++) {
                if (walls_temp1[i_s_TMP_10(i, j, k)] == 1) { // solid node
                    for (int n = 1; n <= 18; n++) {
                        if (walls_temp1[i_s_TMP_10(i + ex[n], j + ey[n], k + ez[n])] <= 0) { // check if neighbor is a fluid node (solid boundary node)
                            walls_temp1[i_s_TMP_10(i, j, k)] = 2;   //solid boundary nodes
                            break;
                        }
                    }
                }
                if (walls_temp1[i_s_TMP_10(i, j, k)] == 0) { // fluid node
                    for (int n = 1; n <= 18; n++) {
                        if (walls_temp1[i_s_TMP_10(i + ex[n], j + ey[n], k + ez[n])] >= 1) { // check if neighbor is a solid node (fluid boundary node)
                            walls_temp1[i_s_TMP_10(i, j, k)] = -1;   //fluid boundary nodes
                            break;
                        }
                    }
                }
            }
        }
    }

    // copy new walls information from (walls_temp1) to (walls_type)
    for (long long k = 1 - 4; k <= nzGlobal + 4; k++) {
        for (long long j = 1 - 4; j <= nyGlobal + 4; j++) {
            for (long long i = 1 - 4; i <= nxGlobal + 4; i++) {
                walls_type[i_s4(i, j, k)] = walls_temp1[i_s_TMP_10(i, j, k)];
            }
        }
    }

    // smooth the geometry from walls_smooth_2 and save in walls_smooth_1 
    for (int iteration = 1; iteration <= i_max_iteration; iteration++) {
        for (long long k = 1 - tmp_ghost_layers + 1; k <= nzGlobal + tmp_ghost_layers - 1; k++) {
            for (long long j = 1 - tmp_ghost_layers + 1; j <= nyGlobal + tmp_ghost_layers - 1; j++) {
                for (long long i = 1 - tmp_ghost_layers + 1; i <= nxGlobal + tmp_ghost_layers - 1; i++) {
                    walls_smooth_2[i_s_TMP_10(i, j, k)] = prc(0.);
                    for (int n = 0; n <= 26; n++) {
                        int m = iex[n] * iex[n] + iey[n] * iey[n] + iez[n] * iez[n];
                        walls_smooth_2[i_s_TMP_10(i, j, k)] += walls_smooth_1[i_s_TMP_10(i + iex[n], j + iey[n], k + iez[n])] * we[m];
                    }
                }
            }
        }
        for (long long k = 1 - tmp_ghost_layers + 1; k <= nzGlobal + tmp_ghost_layers - 1; k++) {
            for (long long j = 1 - tmp_ghost_layers + 1; j <= nyGlobal + tmp_ghost_layers - 1; j++) {
                for (long long i = 1 - tmp_ghost_layers + 1; i <= nxGlobal + tmp_ghost_layers - 1; i++) {
                    walls_smooth_1[i_s_TMP_10(i, j, k)] = walls_smooth_2[i_s_TMP_10(i, j, k)];
                }
            }
        }
    }

    num_solid_boundary_global = 0;
    num_fluid_boundary_global = 0;

    for (long long k = 1 - 4; k <= nzGlobal + 4; k++) { // smoothing process includs overlap_phi region used for cnx, cny, cnz calcaulation
        for (long long j = 1 - 4; j <= nyGlobal + 4; j++) {
            for (long long i = 1 - 4; i <= nxGlobal + 4; i++) {
                if (walls_temp1[i_s_TMP_10(i, j, k)] == 2) {
                    num_solid_boundary_global++;
                }
                if (walls_temp1[i_s_TMP_10(i, j, k)] == -1) {
                    num_fluid_boundary_global++;
                }
            }
        }
    }

    cout << "Total number of solid boundary nodes: " << num_solid_boundary_global << endl;
    cout << "Total number of fluid boundary nodes: " << num_fluid_boundary_global << endl;

    // Calculate nw
   
    for (long long k = 1 - 4; k <= nzGlobal + 4; k++) {
        for (long long j = 1 - 4; j <= nyGlobal + 4; j++) {
            for (long long i = 1 - 4; i <= nxGlobal + 4; i++) {
                if (walls_temp1[i_s_TMP_10(i, j, k)] == -1) {
                    T_P nwx = ISO8[1 - 1] * (walls_smooth_2[i_s_TMP_10(i + 1, j, k)] - walls_smooth_2[i_s_TMP_10(i - 1, j, k)]) +

                        ISO8[2 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j, k + 1)]) +

                        ISO8[3 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k + 1)]) +

                        ISO8[4 - 1] * (walls_smooth_2[i_s_TMP_10(i + 2, j, k)] - walls_smooth_2[i_s_TMP_10(i - 2, j, k)]) +

                        ISO8[5 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 2, j + 1, k)] - walls_smooth_2[i_s_TMP_10(i - 2, j - 1, k)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j - 1, k)] - walls_smooth_2[i_s_TMP_10(i - 2, j + 1, k)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 2, k)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 2, k)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 2, k)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 2, k)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j, k + 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j, k - 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j, k + 2)]) +

                        ISO8[6 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 2, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j + 1, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j - 1, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j - 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j + 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j - 1, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j + 1, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 2, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 2, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 2, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 2, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k - 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k + 2)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k - 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k + 2)]) +

                        ISO8[7 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 2, j + 2, k)] - walls_smooth_2[i_s_TMP_10(i - 2, j - 2, k)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j - 2, k)] - walls_smooth_2[i_s_TMP_10(i - 2, j + 2, k)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j, k + 2)] - walls_smooth_2[i_s_TMP_10(i - 2, j, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j, k - 2)] - walls_smooth_2[i_s_TMP_10(i - 2, j, k + 2)]);

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    T_P nwy =

                        ISO8[1 - 1] * (walls_smooth_2[i_s_TMP_10(i, j + 1, k)] - walls_smooth_2[i_s_TMP_10(i, j - 1, k)]) +

                        ISO8[2 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k)] +
                            walls_smooth_2[i_s_TMP_10(i, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i, j + 1, k - 1)] - walls_smooth_2[i_s_TMP_10(i, j - 1, k + 1)]) +

                        ISO8[3 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k - 1)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k - 1)]) +

                        ISO8[4 - 1] * (walls_smooth_2[i_s_TMP_10(i, j + 2, k)] - walls_smooth_2[i_s_TMP_10(i, j - 2, k)]) +

                        ISO8[5 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 2, j + 1, k)] - walls_smooth_2[i_s_TMP_10(i - 2, j - 1, k)] +
                            walls_smooth_2[i_s_TMP_10(i - 2, j + 1, k)] - walls_smooth_2[i_s_TMP_10(i + 2, j - 1, k)] +
                            walls_smooth_2[i_s_TMP_10(i, j + 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i, j - 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i, j + 2, k - 1)] - walls_smooth_2[i_s_TMP_10(i, j - 2, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 2, k)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 2, k)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 2, k)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 2, k)] +
                            walls_smooth_2[i_s_TMP_10(i, j + 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i, j - 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i, j + 1, k - 2)] - walls_smooth_2[i_s_TMP_10(i, j - 1, k + 2)]) +

                        ISO8[6 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 2, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j + 1, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j - 1, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 2, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 2, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 2, j + 1, k - 1)] - walls_smooth_2[i_s_TMP_10(i + 2, j - 1, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 2, k - 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 2, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 2, k - 1)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 2, k + 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k - 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k + 2)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k - 2)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k + 2)]) +

                        ISO8[7 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 2, j + 2, k)] - walls_smooth_2[i_s_TMP_10(i - 2, j - 2, k)] +
                            walls_smooth_2[i_s_TMP_10(i - 2, j + 2, k)] - walls_smooth_2[i_s_TMP_10(i + 2, j - 2, k)] +
                            walls_smooth_2[i_s_TMP_10(i, j + 2, k + 2)] - walls_smooth_2[i_s_TMP_10(i, j - 2, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i, j + 2, k - 2)] - walls_smooth_2[i_s_TMP_10(i, j - 2, k + 2)]);

                    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cz~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    T_P nwz =

                        ISO8[1 - 1] * (walls_smooth_2[i_s_TMP_10(i, j, k + 1)] - walls_smooth_2[i_s_TMP_10(i, j, k - 1)]) +

                        ISO8[2 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i, j - 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i, j + 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 1, j, k - 1)]) +

                        ISO8[3 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k - 1)]) +

                        ISO8[4 - 1] * (walls_smooth_2[i_s_TMP_10(i, j, k + 2)] - walls_smooth_2[i_s_TMP_10(i, j, k - 2)]) +

                        ISO8[5 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i, j + 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i, j - 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i, j - 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i, j + 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 2, j, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 2, j, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i, j + 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i, j - 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i, j - 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i, j + 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j, k + 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j, k + 2)] - walls_smooth_2[i_s_TMP_10(i + 1, j, k - 2)]) +

                        ISO8[6 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i + 2, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j - 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 2, j + 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 2, j + 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 2, j - 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 2, j - 1, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 2, j + 1, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j - 2, k + 1)] - walls_smooth_2[i_s_TMP_10(i + 1, j + 2, k - 1)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j + 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i + 1, j - 1, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i - 1, j - 1, k + 2)] - walls_smooth_2[i_s_TMP_10(i + 1, j + 1, k - 2)]) +

                        ISO8[7 - 1] * (
                            walls_smooth_2[i_s_TMP_10(i, j + 2, k + 2)] - walls_smooth_2[i_s_TMP_10(i, j - 2, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i, j - 2, k + 2)] - walls_smooth_2[i_s_TMP_10(i, j + 2, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i + 2, j, k + 2)] - walls_smooth_2[i_s_TMP_10(i - 2, j, k - 2)] +
                            walls_smooth_2[i_s_TMP_10(i - 2, j, k + 2)] - walls_smooth_2[i_s_TMP_10(i + 2, j, k - 2)]);

                    T_P tmp = prc(1.) / (prc(sqrt)(nwx * nwx + nwy * nwy + nwz * nwz) + eps); // normalized vector

                    s_nx[i_s4(i, j, k)] = nwx * tmp;
                    s_ny[i_s4(i, j, k)] = nwy * tmp;
                    s_nz[i_s4(i, j, k)] = nwz * tmp;

                }
            }
        }
    }

    /* calculate (num_solid_boundary) and (num_fluid_boundary) */
    num_solid_boundary = num_fluid_boundary = 0;
    for (long long k = 1 - 3; k <= nzGlobal + 3; k++) {
        for (long long j = 1 - 3; j <= nyGlobal + 3; j++) {
            for (long long i = 1 - 3; i <= nxGlobal + 3; i++) {
                if (walls_temp1[i_s_TMP_10(i, j, k)] == 2) {
                    num_solid_boundary++;
                }
                if (walls_temp1[i_s_TMP_10(i, j, k)] == -1) {
                    num_fluid_boundary++;
                }
            }
        }
    }
    
}

//= ====================================================================================================================================
//----------------------geometry_preprocessing based on existing preprocessed data----------------------
//load the geometry information from precomputed file.recommended for relatively large domain
//the geometry in the simulation must 100 % match the preprocessed geometry data!!!
//Due to the issue with different padding schemes used in different compilers for MPI_Type_create_struct,
//The compiler used to compute the geometry info must be the same with the compiler for simulation
//I.e., Intel - Intel, PGI - PGI, GCC - GCC
//=====================================================================================================================================
void geometry_preprocessing_load() {

    /* Geometry boundary info file */
    const char* fnc = geo_boundary_file_path.c_str();
    FILE* geom_file;

    geom_file = fopen(fnc, "r");
    if (geom_file == NULL) { ERROR("Could not open the precomputed boundary info file! Existing Program!"); }
    if (!fread(&num_solid_boundary_global, sizeof(long long), 1, geom_file)) { ERROR("Could not load from data precomputed boundary info file!"); }
    if (!fread(&num_fluid_boundary_global, sizeof(long long), 1, geom_file)) { ERROR("Could not load from data precomputed boundary info file!"); }
    cout << "total number of solid boundary nodes= " << num_solid_boundary_global;
    cout << "total number of fluid boundary nodes= " << num_fluid_boundary_global;

    ERROR(" geometry_preprocessing_load is not implemented yet!");

    printScalarBinary_gh<int>(walls_global, "walls", mem_size_s0_int);
}