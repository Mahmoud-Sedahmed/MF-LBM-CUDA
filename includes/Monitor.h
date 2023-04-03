#ifndef Monitor_H
#define Monitor_H

//==============================================================================
//----------------------monitor_multiphase  unsteady flow----------------------
//==============================================================================
void monitor();
//===============================================================================================================================================
//----------------------monitor_multiphase steady flow----------------------
//===============================================================================================================================================
//*******************************monitor_multiphase steady flow - based on phase field * ************************************
void monitor_multiphase_steady_phasefield();
// *************************** monitor_multiphase steady flow - based on capillary pressure **********************************
void monitor_multiphase_steady_capillarypressure();

//=========================================================================================================================================== =
//----------------------misc subroutines----------------------
//=========================================================================================================================================== =
//*******************************monitor_breakthrough * ************************************
void monitor_breakthrough();

/* calculate saturation */
void cal_saturation();

#endif