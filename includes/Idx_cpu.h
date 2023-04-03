#ifndef IDX_CPU_H
#define IDX_CPU_H
#include "Module_extern.h"

/* Number of ghost nodes */
#define NGH_1	(1)
#define NGH_2	(2)
#define NGH_4	(4)

/* Total number of array nodes after adding the ghost nodes */
// zero ghost layers
#define NXG0	(nxGlobal)
#define NYG0	(nyGlobal)
#define NZG0	(nzGlobal)
// one ghost layer
#define NXG1	(nxGlobal + 2)
#define NYG1	(nyGlobal + 2)
#define NZG1	(nzGlobal + 2)
// two ghost layers
#define NXG2	(nxGlobal + 4)
#define NYG2	(nyGlobal + 4)
#define NZG2	(nzGlobal + 4)
// four ghost layer
#define NXG4	(nxGlobal + 8)
#define NYG4	(nyGlobal + 8)
#define NZG4	(nzGlobal + 8)

/* x,y,z indices considering the ghost nodes and indexing starting from 1 instead of 0 */
// zero ghost layers
#define XG0(x)	(x - 1)			// (x - 1): -1 for starting indexing from 1
#define YG0(y)	(y - 1)
#define ZG0(z)	(z - 1)
// one ghost layer 
#define XG1(x)	(x)				// (x - 1 + 1): -1 for starting indexing from 1, +1 for considering one ghost layer
#define YG1(y)	(y)
#define ZG1(z)	(z)
// two ghost layers
#define XG2(x)	(x + 1)			// (x - 1 + 2): -1 for starting indexing from 1, +2 for considering two ghost layer
#define YG2(y)	(y + 1)
#define ZG2(z)	(z + 1)
// four ghost layers
#define XG4(x)	(x + 3)			// (x - 1 + 4): -1 for starting indexing from 1, +4 for considering four ghost layer
#define YG4(y)	(y + 3)
#define ZG4(z)	(z + 3)

/* indexing functions for arrays */

// scalar array with no ghost layers
#define i_s0(x, y, z)			(XG0(x) + NXG0 * (YG0(y) + NYG0 * ZG0(z)))
// scalar array with one ghost layer					
#define i_s1(x, y, z)			(XG1(x) + NXG1 * (YG1(y) + NYG1 * ZG1(z)))
// scalar array with two ghost layers					
#define i_s2(x, y, z)			(XG2(x) + NXG2 * (YG2(y) + NYG2 * ZG2(z)))
// scalar array with four ghost layers					
#define i_s4(x, y, z)			(XG4(x) + NXG4 * (YG4(y) + NYG4 * ZG4(z)))

// vector array with one ghost layer
#define i_v1(x, y, z, d)		(XG1(x) + NXG1 * (YG1(y) + NYG1 * (ZG1(z) + NZG1 * (d))))
// vector array with two ghost layers
#define i_v2(x, y, z, d)		(XG2(x) + NXG2 * (YG2(y) + NYG2 * (ZG2(z) + NZG2 * (d))))

// distribution function array with one ghost layer
#define i_f1(x, y, z, e, g)		(XG1(x) + NXG1 * (YG1(y) + NYG1 * (ZG1(z) + NZG1 * (e + 19 * g))))

/* indexing functions for special arrays */
// f_convective & g_convective
#define icnv_f1(x, y, e)		(XG1(x) + NXG1 * (YG1(y) + NYG1 * e))


#endif