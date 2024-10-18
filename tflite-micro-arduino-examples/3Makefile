
# Base directory for TensorFlow Lite Micro Arduino (adjust this to your path)
TFLITE_MICRO_ARDUINO_DIR = ../tflite-micro-arduino

# Define include paths for TensorFlow Lite Micro Arduino
TFLITE_MICRO_INCLUDE = \
    -I$(TFLITE_MICRO_ARDUINO_DIR)/src \
    -I$(TFLITE_MICRO_ARDUINO_DIR)/src/tensorflow/lite/micro \
    -I$(TFLITE_MICRO_ARDUINO_DIR)/src/tensorflow/lite/core/api \
    -I$(TFLITE_MICRO_ARDUINO_DIR)/src/tensorflow/lite/kernels \
    -I$(TFLITE_MICRO_ARDUINO_DIR)/src/third_party/flatbuffers/include \
    -I$(TFLITE_MICRO_ARDUINO_DIR)/src/third_party/cmsis/CMSIS/Core/Include \
    -I$(TFLITE_MICRO_ARDUINO_DIR)/src/third_party/cmsis_nn/Include
    

# Define the library path for TensorFlow Lite Micro Arduino (if applicable)
# TFLITE_MICRO_LIB_PATH = $(TFLITE_MICRO_ARDUINO_DIR)/lib
# TFLITE_MICRO_LIBS = -L$(TFLITE_MICRO_LIB_PATH) -ltflite_micro

# Add all .c files from the entire project (recursively)
TFLITE_MICRO_ALL_C_SRCS = '$(TFLITE_MICRO_ARDUINO_DIR)/src/**/*.c)'

# Add preprocessor definitions
CPPFLAGS = -DARDUINO_EXCLUDE_CODE -DTF_LITE_USE_CTIME -DARDUINO $(TFLITE_MICRO_ALL_C_SRCS) 


.PHONY: all clean cleanall makefiles makefiles-so makefiles-lib makefiles-exe checkenvir checkmakefiles submodule-init

all: checkmakefiles
	@cd src && $(MAKE)

clean: checkmakefiles
	@cd src && $(MAKE) clean

cleanall: checkmakefiles
	@cd src && $(MAKE) MODE=release clean
	@cd src && $(MAKE) MODE=debug clean
	@rm -f src/Makefile

# Modify the MAKE options to include TensorFlow Lite Micro Arduino's include paths (removed $(TFLITE_MICRO_LIBS))
MAKEMAKE_OPTIONS := -f --deep -o TFLM -O out -pTFLM -I. $(TFLITE_MICRO_INCLUDE)  $(CPPFLAGS)

makefiles: makefiles-so

makefiles-so: checkenvir
	@cd src && opp_makemake --make-so $(MAKEMAKE_OPTIONS)

makefiles-lib: checkenvir
	@cd src && opp_makemake --make-lib $(MAKEMAKE_OPTIONS)

makefiles-exe: checkenvir
	@cd src && opp_makemake $(MAKEMAKE_OPTIONS)

checkenvir:
	@if [ "$(TFLITE_MICRO_ARDUINO_DIR)" = "" ]; then \
	echo; \
	echo '==========================================================================='; \
	echo 'TensorFlow Lite Micro Arduino path is not set. Please adjust the path!'; \
	echo '==========================================================================='; \
	echo; \
	exit 1; \
	fi

checkmakefiles:
	@if [ ! -f src/Makefile ]; then \
	echo; \
	echo '========================================================================'; \
	echo 'src/Makefile does not exist. Please use "make makefiles" to generate it!'; \
	echo '========================================================================'; \
	echo; \
	exit 1; \
	fi
