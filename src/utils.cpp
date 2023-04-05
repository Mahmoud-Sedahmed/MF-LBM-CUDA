#include "externLib.h"
#include "solver_precision.h"
#include "Module_extern.h"
#include "utils.h"


/* Error handling: ERROR(s) writes an error message and terminates the program */
void errhandler(int nLine, const char* File, const char* err_string)
{
    cout << File << ":" << nLine << " Error: " << err_string << endl;

    exit(1);
}


/* Read a string variable from the simulation setup file */
void read_string(const string& sim_par_file_name, string varName, string& string_variable) {
    ifstream sim_par_file(sim_par_file_name);
    if (sim_par_file.good()) { // Check the file is open successfully
        string read_str; // Temporary string to read a line from the file;
        while (getline(sim_par_file, read_str)) { // Read a line from the file until end of file
            if (!read_str.empty() && *read_str.begin() != '#') { // Check the line is not empty or a comment (#)
                istringstream read_stream(read_str); // Temp. stream for the read line
                string read_var_str; // Temp. string to store the read the name of the read variable
                read_stream >> read_var_str; // store the name of the read variable
                if (read_var_str == varName) { // Matched variable name
                    string var_value_str; // Temp. string to store the value of the variable
                    read_stream >> var_value_str; // store the value of the read variable
                    string var = var_value_str;
                    string_variable = var_value_str;
                    return;
                }
            }
        }
        string err_str = "ERROR! Couldn't find the variable named: (" + varName + ") in the file\n";
        ERROR(err_str.c_str());
    }
    else {
        string err_str = "ERROR! Couldn't open the input file!" + varName + ") in the file\n";
        ERROR(err_str.c_str());
    }
}

/* Read an int variable from the simulation setup file */
void read_int(const string& sim_par_file_name, const string varName, int* int_variable){
    ifstream sim_par_file(sim_par_file_name);
    if (sim_par_file.good()) { // Check the file is open successfully
        string read_str; // Temporary string to read a line from the file;
        while (getline(sim_par_file, read_str)) { // Read a line from the file until end of file
            if (!read_str.empty() && *read_str.begin() != '#') { // Check the line is not empty or a comment (#)
                istringstream read_stream(read_str); // Temp. stream for the read line
                string read_var_str; // Temp. string to store the read the name of the read variable
                read_stream >> read_var_str; // store the name of the read variable
                if (read_var_str == varName) { // Matched variable name
                    string var_value_str; // Temp. string to store the value of the variable
                    read_stream >> var_value_str; // store the value of the read variable
                    int var = stoi(var_value_str);
                    *int_variable = var;
                    return;
                }
            }
        }
        string err_str = "ERROR! Couldn't find the variable named: (" + varName + ") in the file\n";
        ERROR(err_str.c_str()); 
    }
    else {
        string err_str = "ERROR! Couldn't open the input file!" + varName + ") in the file\n";
        ERROR(err_str.c_str());
    }
}

/* Read a long long variable from the simulation setup file */
void read_long_long(const string& sim_par_file_name, const string varName, long long* long_long_variable) {
    ifstream sim_par_file(sim_par_file_name);
    if (sim_par_file.good()) { // Check the file is open successfully
        string read_str; // Temporary string to read a line from the file;
        while (getline(sim_par_file, read_str)) { // Read a line from the file until end of file
            if (!read_str.empty() && *read_str.begin() != '#') { // Check the line is not empty or a comment (#)
                istringstream read_stream(read_str); // Temp. stream for the read line
                string read_var_str; // Temp. string to store the read the name of the read variable
                read_stream >> read_var_str; // store the name of the read variable
                if (read_var_str == varName) { // Matched variable name
                    string var_value_str; // Temp. string to store the value of the variable
                    read_stream >> var_value_str; // store the value of the read variable
                    long long var = stoll(var_value_str);
                    *long_long_variable = var;
                    return;
                }
            }
        }
        string err_str = "ERROR! Couldn't find the variable named: (" + varName + ") in the file\n";
        ERROR(err_str.c_str());
    }
    else {
        string err_str = "ERROR! Couldn't open the input file!" + varName + ") in the file\n";
        ERROR(err_str.c_str());
    }
}

/* Read a float variable from the simulation setup file */
void read_float(const string& sim_par_file_name, const string varName, float* float_variable) {
    ifstream sim_par_file(sim_par_file_name);
    if (sim_par_file.good()) { // Check the file is open successfully
        string read_str; // Temporary string to read a line from the file;
        while (getline(sim_par_file, read_str)) { // Read a line from the file until end of file
            if (!read_str.empty() && *read_str.begin() != '#') { // Check the line is not empty or a comment (#)
                istringstream read_stream(read_str); // Temp. stream for the read line
                string read_var_str; // Temp. string to store the read the name of the read variable
                read_stream >> read_var_str; // store the name of the read variable
                if (read_var_str == varName) { // Matched variable name
                    string var_value_str; // Temp. string to store the value of the variable
                    read_stream >> var_value_str; // store the value of the read variable
                    float var = stof(var_value_str);
                    *float_variable = var;
                    return;
                }
            }
        }
        string err_str = "ERROR! Couldn't find the variable named: (" + varName + ") in the file\n";
        ERROR(err_str.c_str());
    }
    else {
        string err_str = "ERROR! Couldn't open the input file!" + varName + ") in the file\n";
        ERROR(err_str.c_str());
    }
}

/* Read a double variable from the simulation setup file */
void read_double(const string& sim_par_file_name, const string varName, double* double_variable) {
    ifstream sim_par_file(sim_par_file_name);
    if (sim_par_file.good()) { // Check the file is open successfully
        string read_str; // Temporary string to read a line from the file;
        while (getline(sim_par_file, read_str)) { // Read a line from the file until end of file
            if (!read_str.empty() && *read_str.begin() != '#') { // Check the line is not empty or a comment (#)
                istringstream read_stream(read_str); // Temp. stream for the read line
                string read_var_str; // Temp. string to store the read the name of the read variable
                read_stream >> read_var_str; // store the name of the read variable
                if (read_var_str == varName) { // Matched variable name
                    string var_value_str; // Temp. string to store the value of the variable
                    read_stream >> var_value_str; // store the value of the read variable
                    double var = stod(var_value_str);
                    *double_variable = var;
                    return;
                }
            }
        }
        string err_str = "ERROR! Couldn't find the variable named: (" + varName + ") in the file\n";
        ERROR(err_str.c_str());
    }
    else {
        string err_str = "ERROR! Couldn't open the input file!" + varName + ") in the file\n";
        ERROR(err_str.c_str());
    }
}

