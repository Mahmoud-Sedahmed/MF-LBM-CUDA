######################################################################################################################
# Makefile for MF-LBM-CUDA (C++/CUDA version)
# Author: M. Sedahmed
######################################################################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Editable parameters 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GPU compute capability
COMPUTE_CAPABILITY := 86
# CUDA tookit library path (e.g., /usr/local/cuda/lib64)
CUDA_LIBRARY_LOCATION := /usr/local/cuda/lib64
CUDA_LNK_PATH := /usr/local/cuda/include
# Build type : (realase / debug)
BUILD := release

#Additional flags to compilers
CPPFLAGS_ADD := 
CUFLAGS_ADD := 		# --ptxas-options=-v
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

######################################################################################################################
# Directories
######################################################################################################################
SRC_DIR := src
OBJ_DIR := obj
DEP_DIR := dep

######################################################################################################################
# Files
######################################################################################################################
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DEPS := $(SOURCES:$(SRC_DIR)/%.cpp=$(DEP_DIR)/%.d)
	
SOURCES_CU := $(wildcard $(SRC_DIR)/*.cu)
OBJECTS_CU := $(SOURCES_CU:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
DEPS_CU := $(SOURCES_CU:$(SRC_DIR)/%.cu=$(DEP_DIR)/%.d)
OBJECTS_CU_LINK := $(OBJECTS_CU:$(OBJ_DIR)/%.o=$(OBJ_DIR)/%_link.o)

######################################################################################################################
# C++ compiler flags (GCC)
######################################################################################################################
EXECUTABLE := MF_LBM_CUDA
CC := g++
INCLUDES := -I includes -I $(CUDA_LNK_PATH) 
CPPFLAGS += -std=c++17
ifeq ($(strip $(BUILD)), release)
CPPFLAGS += -O3 -Wall
else ifeq ($(strip $(BUILD)), debug)
CPPFLAGS += -g
endif
CPPFLAGS += $(CPPFLAGS_ADD)
COMPILE.cpp := $(CC) $(CFLAGS) $(INCLUDES) $(CPPFLAGS) $(TARGET_ARCH)

######################################################################################################################
# CUDA compiler flags
######################################################################################################################
CC_CU := nvcc
CFLAGS_CU := -arch=sm_$(COMPUTE_CAPABILITY) -std=c++17 -rdc=true
ifeq ($(strip $(BUILD)), release)
CFLAGS_CU += -O3 -lineinfo
else ifeq ($(strip $(BUILD)), debug)
CFLAGS_CU += -g
endif
CFLAGS_CU += $(CUFLAGS_ADD)
LDFLAGS_CU := -lcudart
LDFLAGS := -lm
COMPILE.cu := $(CC_CU) $(INCLUDES) $(CFLAGS_CU) $(LDFLAGS_CU)

######################################################################################################################
# Make rules
######################################################################################################################

all: build_msg_start mk_dir $(EXECUTABLE) build_msg_end

$(EXECUTABLE): $(OBJECTS_CU) $(OBJECTS_CU_LINK) $(OBJECTS) #$(CUDA_LINKER)
	$(CC) $(OBJECTS_CU) $(OBJECTS_CU_LINK) $(OBJECTS) -o $@ -L$(CUDA_LIBRARY_LOCATION) $(LDFLAGS) $(LDFLAGS_CU)
	
$(OBJECTS_CU): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	$(CC_CU) $(INCLUDES) $(CFLAGS_CU) -c $< -o $@ $(LDFLAGS_CU)
# This step is required for CUDA separate compilation and linking with gcc compiler (Dynamic parallelism requires relocatable device code linking)
$(OBJECTS_CU_LINK): $(OBJ_DIR)/%_link.o : $(OBJ_DIR)/%.o
	$(CC_CU) -arch=sm_$(COMPUTE_CAPABILITY) -dlink $< -o $@ -lcudadevrt -lcudart

$(OBJECTS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(COMPILE.cpp) -c $< -o $@	

######################################################################################################################
# Generate Dependencies
######################################################################################################################
ifneq ($(MAKECMDGOALS),clean)	
include $(DEPS)
include $(DEPS_CU)

$(DEP_DIR)/%.d : $(SRC_DIR)/%.cpp
	@set -e; rm -f $@; \
	$(COMPILE.cpp) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,$(OBJ_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
	
$(DEP_DIR)/%.d : $(SRC_DIR)/%.cu
	@set -e; rm -f $@; \
	$(COMPILE.cu) -MM $(CFLAGS_CU) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,$(OBJ_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
endif	

######################################################################################################################
.PHONY: build_msg_start
build_msg_start: 
	@printf "#-------------------------------------------------\n# Building ($(EXECUTABLE)) Started! \n# Author: M. Sedahmed \n#-------------------------------------------------\n"
	
.PHONY: build_msg_end
build_msg_end:
	@printf "#-------------------------------------------------\n# Building ($(EXECUTABLE)) finished! \n#-------------------------------------------------\n"
.PHONY: mk_dir
mk_dir:
	@mkdir -p $(OBJ_DIR) $(DEP_DIR) results

.PHONY: clean
clean: 
	-rm -f $(OBJ_DIR)/* $(DEP_DIR)/* $(EXECUTABLE)
# print rule to print the expression of variables	
.PHONY: print-%
print-%  : ; @echo $* = $($*)	
















	