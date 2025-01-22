#
# photophorese makefile 0.01
#

CC       = g++
CFLAGS   = -c -std=c++11 -O3 
LDFLAGS  = -lm 


CUDA_DIR    = /usr/local/cuda-7.5

NVCC   = ${CUDA_DIR}/bin/nvcc
#NVFLAGS  = -x cu -c -dc -O3  -Xcompiler "-O3 -pthread" --ptxas-options=-v  -g -G
NVFLAGS  = -x cu -c -std=c++11 -dc -O3  -Xcompiler "-O3 -pthread" --ptxas-options=-v

CUDA_LINK_FLAGS = -dlink
CUDA_LINK_OBJ = cuLink.o


# please make sure, that GPU_ARCH corresponds to your hardware
# otherwise the code does not work!
# gtx 570
#GPU_ARCH = -arch=sm_20
# gtx titan
GPU_ARCH = -arch=sm_35
# gtx 970
#GPU_ARCH = -arch=sm_52
# gtx 750 Ti
#GPU_ARCH = -arch=sm_50

CUDA_LIB      = ${CUDA_DIR}
INCLUDE_DIRS += -I$(CUDA_LIB)/include -I$(CUDA_LIB)/include/thrust
LDFLAGS      += -L$(CUDA_LIB)/lib64 -lcudart -lpthread

# Default target
all: photophorese


# ---------------------------------------------------------
#         Set headers and object files 
# ---------------------------------------------------------

HEADERS = 
CUDA_HEADERS =  Photophorese_device_functions.h Photophorese_host_functions.h
OBJ = 
CUDA_OBJ = Photophorese_device_functions.o Photophorese_host_functions.o main.o

photophorese: $(OBJ) $(CUDA_OBJ)
#	$(NVCC) $(GPU_ARCH) $(CUDA_LINK_FLAGS) -o $(CUDA_LINK_OBJ) $(CUDA_OBJ)
	$(NVCC) $(GPU_ARCH) $(CUDA_OBJ) $(LDFLAGS) -o $@
#	$(CC) $(OBJ) $(CUDA_OBJ) $(CUDA_LINK_OBJ) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $<
%.o: %.cpp
	$(NVCC) $(CFLAGS) $(INCLUDE_DIRS) $<
	
%.o: %.cu
	$(NVCC) $(GPU_ARCH) $(NVFLAGS) $(INCLUDE_DIRS) $<

.PHONY: clean
clean:
	@rm -f	*.o
	@echo make clean: done

# ---------------------------------------------------------
#          Dependencies for object files
# ---------------------------------------------------------

$(OBJ):  $(HEADERS) Makefile
$(CUDA_OBJ): $(HEADERS) $(CUDA_HEADERS) Makefile

