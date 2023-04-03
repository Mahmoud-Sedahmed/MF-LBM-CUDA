#ifndef SOLVER_PRECISION_H
#define SOLVER_PRECISION_H

/* Solver precision */
#define SINGLE_PRECISION (1)
#define DOUBLE_PRECISION (2)
/* Select solver precision: SINGLE_PRECISION / DOUBLE_PRECISION */
#define PRECISION (SINGLE_PRECISION)		

#if (PRECISION == SINGLE_PRECISION)
#define T_P float
#elif (PRECISION == DOUBLE_PRECISION)
#define T_P double
#endif

#if (PRECISION == SINGLE_PRECISION)
#define prc(x)	x##f
#define pprc(x)	f##x
#elif (PRECISION == DOUBLE_PRECISION)
#define prc(x)	x
#define pprc(x)	x
#endif





#endif