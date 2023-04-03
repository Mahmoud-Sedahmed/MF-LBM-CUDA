#ifndef __UTILS_H__
#define __UTILS_H__

/* Error handling: ERROR(s) writes an error message and terminates the program */
#define ERROR(s)    errhandler(__LINE__, __FILE__, s)
void  errhandler(int nLine, const char* File, const char* err_string);

#define READ_INT( szFileName, VarName)		read_int( szFileName, #VarName, &(VarName) )
#define READ_FLOAT( szFileName, VarName)	read_float( szFileName, #VarName, &(VarName) )
#define READ_DOUBLE( szFileName, VarName)	read_double( szFileName, #VarName, &(VarName) )
#define READ_STRING( szFileName, VarName)	read_string( szFileName, #VarName,  (VarName) )


/* Read an int variable from the simulation setup file */
void read_int(const string& sim_par_file_name, const string varName, int* int_variable);

/* Read a float variable from the simulation setup file */
void read_float(const string& sim_par_file_name, const string varName, float* float_variable);

/* Read a double variable from the simulation setup file */
void read_double(const string& sim_par_file_name, const string varName, double* double_variable);

/* Read a string variable from the simulation setup file */
void read_string(const string& sim_par_file_name, string varName, string& string_variable);


#if (PRECISION == SINGLE_PRECISION)
#define READ_T_P READ_FLOAT
#elif(PRECISION == DOUBLE_PRECISION)
#define READ_T_P READ_DOUBLE
#endif



#endif
