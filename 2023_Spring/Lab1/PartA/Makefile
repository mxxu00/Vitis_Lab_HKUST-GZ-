# Points to top directory of Git repository
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%host_xrt/hello_world_xrt/*}')
PWD = $(shell readlink -f .)
XF_PROJ_ROOT = $(shell readlink -f $(COMMON_REPO))
PLATFORM ?= xilinx_u55c_gen3x16_xdma_3_202210_1

########################## Checking if PLATFORM in allowlist #######################
# PLATFORM_BLOCKLIST += nodma 
# # PLATFORM ?= xilinx_u55c_gen3x16_xdma_3_202210_1
# DEV_ARCH := $(shell platforminfo -p $(PLATFORM) | grep 'FPGA Family' | sed 's/.*://' | sed '/ai_engine/d' | sed 's/^[[:space:]]*//')
# # platforminfo -p xilinx_u55c_gen3x16_xdma_3_202210_1 | grep 'FPGA Family'  | sed 's/.*://' | sed '/ai_engine/d' | sed 's/^[[:space:]]*//'
# # ↑ virtexuplusHBM
# CPU_TYPE := $(shell platforminfo -p $(PLATFORM) | grep 'CPU Type' | sed 's/.*://' | sed '/ai_engine/d' | sed 's/^[[:space:]]*//')
# # platforminfo -p xilinx_u55c_gen3x16_xdma_3_202210_1 | grep 'CPU Type' | sed 's/.*://' | sed '/ai_engine/d' | sed 's/^[[:space:]]*//'
# # ↑ x86

# ifeq ($(CPU_TYPE), cortex-a9)
# HOST_ARCH := aarch32
# else ifneq (,$(findstring cortex-a, $(CPU_TYPE)))
# HOST_ARCH := aarch64
# else
# HOST_ARCH := x86
# endif

# ifeq ($(DEV_ARCH), zynquplus)
# ifeq ($(HOST_ARCH), aarch64)
# include makefile_zynqmp.mk
# else
# include makefile_us_alveo.mk
# endif
# else ifeq ($(DEV_ARCH), versal)
# ifeq ($(HOST_ARCH), x86)
# include makefile_versal_alveo.mk
# else
# include makefile_versal_ps.mk 
# endif
# else
# include makefile_us_alveo.mk # <<< this one
# endif

############################## Setting up Project Variables ##############################
TARGET := hw
VPP_LDFLAGS :=
include ./utils.mk

CURRENT_DIR := $(shell pwd)
ROOT_DIR := $(abspath $(CURRENT_DIR)/../..)
COMMON_DIR := $(ROOT_DIR)/common

TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

LINK_OUTPUT := $(BUILD_DIR)/real_matmul.link.xclbin
PACKAGE_OUT = ./package.$(TARGET)

CXXFLAGS += -I$(COMMON_DIR)  -I$(XILINX_XRT)/include/CL

VPP_PFLAGS := 
CMD_ARGS = $(BUILD_DIR)/real_matmul.xclbin
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -I$(XILINX_HLS)/include -Wall -O0 -g -std=c++1y
LDFLAGS += -L$(XILINX_XRT)/lib -pthread -lOpenCL

########################## Checking if PLATFORM in allowlist #######################
PLATFORM_BLOCKLIST += nodma 
############################## Setting up Host Variables ##############################
# Include Required Host Source Files
# CXXFLAGS += -I$(XF_PROJ_ROOT)/common/includes/cmdparser
# CXXFLAGS += -I$(XF_PROJ_ROOT)/common/includes/logger
# HOST_SRCS += $(XF_PROJ_ROOT)/common/includes/cmdparser/cmdlineparser.cpp $(XF_PROJ_ROOT)/common/includes/logger/logger.cpp ./src/host.cpp 

HOST_SRCS += $(COMMON_DIR)/xilinx_ocl_helper.cpp
HOST_SRCS += ./src/host.cpp
# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 
LDFLAGS += -luuid -lxrt_coreutil 
LDFLAGS += -lxrt_core

############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += --save-temps 


EXECUTABLE = ./real_matmul
EMCONFIG_DIR = $(TEMP_DIR)

############################## Setting Targets ##############################
.PHONY: all clean cleanall docs emconfig
all: check-platform check-device check-vitis $(EXECUTABLE) $(BUILD_DIR)/real_matmul.xclbin emconfig

.PHONY: host
host: $(EXECUTABLE)

.PHONY: build
build: check-vitis check-device $(BUILD_DIR)/real_matmul.xclbin

.PHONY: xclbin
xclbin: build

############################## Setting Rules for Binary Containers (Building Kernels) ##############################
$(TEMP_DIR)/real_matmul.xo: src/real_matmul.cpp
	mkdir -p $(TEMP_DIR)
	v++ -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k real_matmul --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'

$(BUILD_DIR)/real_matmul.xclbin: $(TEMP_DIR)/real_matmul.xo
	mkdir -p $(BUILD_DIR)
	v++ -l $(VPP_FLAGS) $(VPP_LDFLAGS) -t $(TARGET) --platform $(PLATFORM) --temp_dir $(TEMP_DIR) -o'$(LINK_OUTPUT)' $(+)
	v++ -p $(LINK_OUTPUT) $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/real_matmul.xclbin

############################## Setting Rules for Host (Building Host Executable) ##############################
$(EXECUTABLE): $(HOST_SRCS) | check-xrt
		g++ -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(EMCONFIG_DIR)

############################## Setting Essential Checks and Running Rules ##############################
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	cp -rf $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(EXECUTABLE) $(CMD_ARGS)
endif

.PHONY: test
test: $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(EXECUTABLE) $(CMD_ARGS)
endif

############################## Cleaning Rules ##############################
# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin

	-$(RMDIR) *.compile_summary *.ltx *.info *.link_summary *.package_summary *.xo



