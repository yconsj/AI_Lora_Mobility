#
# OMNeT++/OMNEST Makefile for tflite-micro-arduino-examples
#
# This file was generated with the command:
#  opp_makemake -f --deep -O out -I. -L../out/$$\(CONFIGNAME\)
#

# Name of target to be created (-o option)
TARGET_DIR = .
TARGET_NAME = tflite-micro-arduino-examples$(D)
TARGET = $(TARGET_NAME)$(EXE_SUFFIX)
TARGET_IMPLIB = $(TARGET_NAME)$(IMPLIB_SUFFIX)
TARGET_IMPDEF = $(TARGET_NAME)$(IMPDEF_SUFFIX)
TARGET_FILES = $(TARGET_DIR)/$(TARGET)

# User interface (uncomment one) (-u option)
USERIF_LIBS = $(ALL_ENV_LIBS) # that is, $(QTENV_LIBS) $(CMDENV_LIBS)
#USERIF_LIBS = $(CMDENV_LIBS)
#USERIF_LIBS = $(QTENV_LIBS)

# C++ include paths (with -I)
INCLUDE_PATH = -I.

# Additional object and library files to link with
EXTRA_OBJS =

# Additional libraries (-L, -l options)
LIBS = $(LDFLAG_LIBPATH)../out/$(CONFIGNAME)

# Output directory
PROJECT_OUTPUT_DIR = ../out
PROJECTRELATIVE_PATH = src
O = $(PROJECT_OUTPUT_DIR)/$(CONFIGNAME)/$(PROJECTRELATIVE_PATH)

# Object files for local .cpp, .msg and .sm files
OBJS = \
    $O/tensorflow/lite/core/api/error_reporter.o \
    $O/tensorflow/lite/core/api/flatbuffer_conversions.o \
    $O/tensorflow/lite/core/api/op_resolver.o \
    $O/tensorflow/lite/core/api/tensor_utils.o \
    $O/tensorflow/lite/core/c/common.o \
    $O/tensorflow/lite/experimental/microfrontend/lib/fft.o \
    $O/tensorflow/lite/experimental/microfrontend/lib/fft_util.o \
    $O/tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.o \
    $O/tensorflow/lite/kernels/kernel_util.o \
    $O/tensorflow/lite/kernels/internal/portable_tensor_utils.o \
    $O/tensorflow/lite/kernels/internal/quantization_util.o \
    $O/tensorflow/lite/kernels/internal/tensor_utils.o \
    $O/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.o \
    $O/tensorflow/lite/micro/all_ops_resolver.o \
    $O/tensorflow/lite/micro/debug_log.o \
    $O/tensorflow/lite/micro/fake_micro_context.o \
    $O/tensorflow/lite/micro/flatbuffer_utils.o \
    $O/tensorflow/lite/micro/memory_helpers.o \
    $O/tensorflow/lite/micro/micro_allocation_info.o \
    $O/tensorflow/lite/micro/micro_allocator.o \
    $O/tensorflow/lite/micro/micro_context.o \
    $O/tensorflow/lite/micro/micro_graph.o \
    $O/tensorflow/lite/micro/micro_interpreter.o \
    $O/tensorflow/lite/micro/micro_log.o \
    $O/tensorflow/lite/micro/micro_profiler.o \
    $O/tensorflow/lite/micro/micro_resource_variable.o \
    $O/tensorflow/lite/micro/micro_string.o \
    $O/tensorflow/lite/micro/micro_time.o \
    $O/tensorflow/lite/micro/micro_utils.o \
    $O/tensorflow/lite/micro/mock_micro_graph.o \
    $O/tensorflow/lite/micro/recording_micro_allocator.o \
    $O/tensorflow/lite/micro/system_setup.o \
    $O/tensorflow/lite/micro/test_helper_custom_ops.o \
    $O/tensorflow/lite/micro/test_helpers.o \
    $O/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.o \
    $O/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.o \
    $O/tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.o \
    $O/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.o \
    $O/tensorflow/lite/micro/kernels/activations.o \
    $O/tensorflow/lite/micro/kernels/activations_common.o \
    $O/tensorflow/lite/micro/kernels/add_common.o \
    $O/tensorflow/lite/micro/kernels/arg_min_max.o \
    $O/tensorflow/lite/micro/kernels/assign_variable.o \
    $O/tensorflow/lite/micro/kernels/batch_to_space_nd.o \
    $O/tensorflow/lite/micro/kernels/broadcast_args.o \
    $O/tensorflow/lite/micro/kernels/broadcast_to.o \
    $O/tensorflow/lite/micro/kernels/call_once.o \
    $O/tensorflow/lite/micro/kernels/cast.o \
    $O/tensorflow/lite/micro/kernels/ceil.o \
    $O/tensorflow/lite/micro/kernels/circular_buffer.o \
    $O/tensorflow/lite/micro/kernels/circular_buffer_common.o \
    $O/tensorflow/lite/micro/kernels/comparisons.o \
    $O/tensorflow/lite/micro/kernels/concatenation.o \
    $O/tensorflow/lite/micro/kernels/conv_common.o \
    $O/tensorflow/lite/micro/kernels/cumsum.o \
    $O/tensorflow/lite/micro/kernels/depth_to_space.o \
    $O/tensorflow/lite/micro/kernels/depthwise_conv_common.o \
    $O/tensorflow/lite/micro/kernels/dequantize.o \
    $O/tensorflow/lite/micro/kernels/dequantize_common.o \
    $O/tensorflow/lite/micro/kernels/detection_postprocess.o \
    $O/tensorflow/lite/micro/kernels/div.o \
    $O/tensorflow/lite/micro/kernels/elementwise.o \
    $O/tensorflow/lite/micro/kernels/elu.o \
    $O/tensorflow/lite/micro/kernels/ethosu.o \
    $O/tensorflow/lite/micro/kernels/exp.o \
    $O/tensorflow/lite/micro/kernels/expand_dims.o \
    $O/tensorflow/lite/micro/kernels/fill.o \
    $O/tensorflow/lite/micro/kernels/floor.o \
    $O/tensorflow/lite/micro/kernels/floor_div.o \
    $O/tensorflow/lite/micro/kernels/floor_mod.o \
    $O/tensorflow/lite/micro/kernels/fully_connected_common.o \
    $O/tensorflow/lite/micro/kernels/gather.o \
    $O/tensorflow/lite/micro/kernels/gather_nd.o \
    $O/tensorflow/lite/micro/kernels/hard_swish.o \
    $O/tensorflow/lite/micro/kernels/hard_swish_common.o \
    $O/tensorflow/lite/micro/kernels/if.o \
    $O/tensorflow/lite/micro/kernels/kernel_runner.o \
    $O/tensorflow/lite/micro/kernels/kernel_util.o \
    $O/tensorflow/lite/micro/kernels/l2_pool_2d.o \
    $O/tensorflow/lite/micro/kernels/l2norm.o \
    $O/tensorflow/lite/micro/kernels/leaky_relu.o \
    $O/tensorflow/lite/micro/kernels/leaky_relu_common.o \
    $O/tensorflow/lite/micro/kernels/log_softmax.o \
    $O/tensorflow/lite/micro/kernels/logical.o \
    $O/tensorflow/lite/micro/kernels/logical_common.o \
    $O/tensorflow/lite/micro/kernels/logistic.o \
    $O/tensorflow/lite/micro/kernels/logistic_common.o \
    $O/tensorflow/lite/micro/kernels/lstm_eval.o \
    $O/tensorflow/lite/micro/kernels/maximum_minimum.o \
    $O/tensorflow/lite/micro/kernels/micro_tensor_utils.o \
    $O/tensorflow/lite/micro/kernels/mirror_pad.o \
    $O/tensorflow/lite/micro/kernels/mul_common.o \
    $O/tensorflow/lite/micro/kernels/neg.o \
    $O/tensorflow/lite/micro/kernels/pack.o \
    $O/tensorflow/lite/micro/kernels/pad.o \
    $O/tensorflow/lite/micro/kernels/pooling_common.o \
    $O/tensorflow/lite/micro/kernels/prelu.o \
    $O/tensorflow/lite/micro/kernels/prelu_common.o \
    $O/tensorflow/lite/micro/kernels/quantize.o \
    $O/tensorflow/lite/micro/kernels/quantize_common.o \
    $O/tensorflow/lite/micro/kernels/read_variable.o \
    $O/tensorflow/lite/micro/kernels/reduce.o \
    $O/tensorflow/lite/micro/kernels/reduce_common.o \
    $O/tensorflow/lite/micro/kernels/reshape.o \
    $O/tensorflow/lite/micro/kernels/resize_bilinear.o \
    $O/tensorflow/lite/micro/kernels/resize_nearest_neighbor.o \
    $O/tensorflow/lite/micro/kernels/round.o \
    $O/tensorflow/lite/micro/kernels/select.o \
    $O/tensorflow/lite/micro/kernels/shape.o \
    $O/tensorflow/lite/micro/kernels/slice.o \
    $O/tensorflow/lite/micro/kernels/softmax_common.o \
    $O/tensorflow/lite/micro/kernels/space_to_batch_nd.o \
    $O/tensorflow/lite/micro/kernels/space_to_depth.o \
    $O/tensorflow/lite/micro/kernels/split.o \
    $O/tensorflow/lite/micro/kernels/split_v.o \
    $O/tensorflow/lite/micro/kernels/squared_difference.o \
    $O/tensorflow/lite/micro/kernels/squeeze.o \
    $O/tensorflow/lite/micro/kernels/strided_slice.o \
    $O/tensorflow/lite/micro/kernels/sub.o \
    $O/tensorflow/lite/micro/kernels/sub_common.o \
    $O/tensorflow/lite/micro/kernels/svdf_common.o \
    $O/tensorflow/lite/micro/kernels/tanh.o \
    $O/tensorflow/lite/micro/kernels/transpose.o \
    $O/tensorflow/lite/micro/kernels/transpose_conv.o \
    $O/tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.o \
    $O/tensorflow/lite/micro/kernels/unpack.o \
    $O/tensorflow/lite/micro/kernels/var_handle.o \
    $O/tensorflow/lite/micro/kernels/while.o \
    $O/tensorflow/lite/micro/kernels/zeros_like.o \
    $O/tensorflow/lite/micro/kernels/cmsis_nn/add.o \
    $O/tensorflow/lite/micro/kernels/cmsis_nn/conv.o \
    $O/tensorflow/lite/micro/kernels/cmsis_nn/depthwise_conv.o \
    $O/tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.o \
    $O/tensorflow/lite/micro/kernels/cmsis_nn/mul.o \
    $O/tensorflow/lite/micro/kernels/cmsis_nn/pooling.o \
    $O/tensorflow/lite/micro/kernels/cmsis_nn/softmax.o \
    $O/tensorflow/lite/micro/kernels/cmsis_nn/svdf.o \
    $O/tensorflow/lite/micro/memory_planner/greedy_memory_planner.o \
    $O/tensorflow/lite/micro/memory_planner/linear_memory_planner.o \
    $O/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.o \
    $O/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.o \
    $O/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.o \
    $O/tensorflow/lite/micro/tflite_bridge/op_resolver_bridge.o \
    $O/tensorflow/lite/schema/schema_utils.o \
    $O/test_over_serial/base64.o \
    $O/test_over_serial/test_over_serial.o

# Message files
MSGFILES =

# SM files
SMFILES =

#------------------------------------------------------------------------------

# Pull in OMNeT++ configuration (Makefile.inc)

ifneq ("$(OMNETPP_CONFIGFILE)","")
CONFIGFILE = $(OMNETPP_CONFIGFILE)
else
CONFIGFILE = $(shell opp_configfilepath)
endif

ifeq ("$(wildcard $(CONFIGFILE))","")
$(error Config file '$(CONFIGFILE)' does not exist -- add the OMNeT++ bin directory to the path so that opp_configfilepath can be found, or set the OMNETPP_CONFIGFILE variable to point to Makefile.inc)
endif

include $(CONFIGFILE)

# Simulation kernel and user interface libraries
OMNETPP_LIBS = $(OPPMAIN_LIB) $(USERIF_LIBS) $(KERNEL_LIBS) $(SYS_LIBS)
ifneq ($(PLATFORM),win32)
LIBS += -Wl,-rpath,$(abspath ../out/$(CONFIGNAME))
endif

COPTS = $(CFLAGS) $(IMPORT_DEFINES)  $(INCLUDE_PATH) -I$(OMNETPP_INCL_DIR)
MSGCOPTS = $(INCLUDE_PATH)
SMCOPTS =

# we want to recompile everything if COPTS changes,
# so we store COPTS into $COPTS_FILE (if COPTS has changed since last build)
# and make the object files depend on it
COPTS_FILE = $O/.last-copts
ifneq ("$(COPTS)","$(shell cat $(COPTS_FILE) 2>/dev/null || echo '')")
  $(shell $(MKPATH) "$O")
  $(file >$(COPTS_FILE),$(COPTS))
endif

#------------------------------------------------------------------------------
# User-supplied makefile fragment(s)
-include makefrag

#------------------------------------------------------------------------------

# Main target
all: $(TARGET_FILES)

$(TARGET_DIR)/% :: $O/%
	@mkdir -p $(TARGET_DIR)
	$(Q)$(LN) $< $@
ifeq ($(TOOLCHAIN_NAME),clang-msabi)
	-$(Q)-$(LN) $(<:%.dll=%.lib) $(@:%.dll=%.lib) 2>/dev/null

$O/$(TARGET_NAME).pdb: $O/$(TARGET)
endif

$O/$(TARGET): $(OBJS)  $(wildcard $(EXTRA_OBJS)) Makefile $(CONFIGFILE)
	@$(MKPATH) $O
	@echo Creating executable: $@
	$(Q)echo >.tmp$$$$ $(OBJS) $(EXTRA_OBJS) && $(CXX) -o $O/$(TARGET) @.tmp$$$$ $(AS_NEEDED_OFF) $(WHOLE_ARCHIVE_ON) $(LIBS) $(WHOLE_ARCHIVE_OFF) $(OMNETPP_LIBS) $(LDFLAGS) ; rm .tmp$$$$

.PHONY: all clean cleanall depend msgheaders smheaders

# disabling all implicit rules
.SUFFIXES :
.PRECIOUS : %_m.h %_m.cpp

$O/%.o: %.cpp $(COPTS_FILE) | msgheaders smheaders
	@$(MKPATH) $(dir $@)
	$(qecho) "$<"
	$(Q)$(CXX) -c $(CXXFLAGS) $(COPTS) -o $@ $<

%_m.cpp %_m.h: %.msg
	$(qecho) MSGC: $<
	$(Q)$(MSGC) -s _m.cpp -MD -MP -MF $O/$(basename $<)_m.h.d $(MSGCOPTS) $?

%_sm.cpp %_sm.h: %.sm
	$(qecho) SMC: $<
	$(Q)$(SMC) -c++ -suffix cpp $(SMCOPTS) $?

msgheaders: $(MSGFILES:.msg=_m.h)

smheaders: $(SMFILES:.sm=_sm.h)

clean:
	$(qecho) Cleaning $(TARGET)
	$(Q)-rm -rf $O
	$(Q)-rm -f $(TARGET_FILES)
	$(Q)-rm -f $(call opp_rwildcard, . , *_m.cpp *_m.h *_sm.cpp *_sm.h)

cleanall:
	$(Q)$(CLEANALL_COMMAND)
	$(Q)-rm -rf $(PROJECT_OUTPUT_DIR)

help:
	@echo "$$HELP_SYNOPSYS"
	@echo "$$HELP_TARGETS"
	@echo "$$HELP_VARIABLES"
	@echo "$$HELP_EXAMPLES"

# include all dependencies
-include $(OBJS:%=%.d) $(MSGFILES:%.msg=$O/%_m.h.d)
