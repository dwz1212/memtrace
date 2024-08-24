# 设置编译器和工具链
NVCC = nvcc -ccbin=$(CXX) -D_FORCE_INLINES
CXX = g++
PTXAS = ptxas

# 设置 NVCC 和 PTXAS 的最低版本要求
NVCC_VER_REQ = 10.1
PTXAS_VER_ADD_FLAG = 12.3

# 获取当前的 NVCC 和 PTXAS 版本
NVCC_VER = $(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER = $(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')

# 检查 NVCC 版本是否满足要求
NVCC_VER_CHECK = $(shell echo "$(NVCC_VER) >= $(NVCC_VER_REQ)" | bc)
ifeq ($(NVCC_VER_CHECK), 0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile this NVBit tool!)
endif

# 检查 PTXAS 版本是否满足要求
PTXAS_VER_CHECK = $(shell echo "$(PTXAS_VER) >= $(PTXAS_VER_ADD_FLAG)" | bc)
ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG = -maxrregcount=24
else
MAXRREGCOUNT_FLAG =
endif

# NVBit 工具和包含路径
NVBIT_PATH = ../../core
INCLUDES = -I$(NVBIT_PATH)

# 库链接路径和库
LIBS = -L$(NVBIT_PATH) -lnvbit
NVCC_PATH = -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

# C++ 编译器和链接器标志
CXXFLAGS = -std=c++17 -O3 -fPIC
LDFLAGS = -shared

# 更新源文件列表
CU_SOURCES = $(wildcard *.cu)
CPP_SOURCES = $(wildcard *.cpp)

CU_OBJECTS = $(CU_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS)
ARCH ?= sm_60  # 根据需要修改目标架构

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL = $(current_dir).so

# 默认目标
all: $(NVBIT_TOOL)

# 编译共享库
$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static $(LDFLAGS) -o $@

# 编译 CUDA 文件
%.o: %.cu
	$(NVCC) -dc -c $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

# 编译 C++ 文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 编译 inject_funcs 文件
inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

# 清理目标
clean:
	rm -f *.so *.o
