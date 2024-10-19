//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 


#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <omnetpp.h>




#include "LearningModel.h"
#include "tensorflow/lite/micro/system_setup.h"


#include "constants.h"
#include "main_functions.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


namespace inet {

Define_Module(LearningModel);

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];



LearningModel::LearningModel()
{
    // stub
}

void LearningModel::initialize()
{

    EV_TRACE << "initializing LearningModel " << omnetpp::endl;


    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.

    model = tflite::GetModel(g_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
      MicroPrintf(
          "Model provided is schema version %d not equal "
          "to supported version %d.",
          model->version(), TFLITE_SCHEMA_VERSION);
      return;
    }

    // This pulls in all the operation implementations we need.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::AllOpsResolver resolver;

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      MicroPrintf("AllocateTensors() failed");
      return;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Keep track of how many inferences we have performed.
    inference_count = 0;
    /*

    */

}

double LearningModel::pollModel()
{
    return uniform(0,1);
}

void LearningModel::PrintOutput(float x_value, float y_value) {

    EV_INFO << "x: " << x_value << ", y:" << y_value << omnetpp::endl;

}

void LearningModel::Test() {

    EV << "LearningModel.Test()" << omnetpp::endl;
    // Calculate an x value to feed into the model. We compare the current
   // inference_count to the number of inferences per cycle to determine
   // our position within the range of possible x values the model was
   // trained on, and use this to calculate a value.

    float position = static_cast<float>(inference_count) /
                    static_cast<float>(kInferencesPerCycle);
   float x = position * kXrange;

   // Quantize the input from floating-point to integer
   int8_t x_quantized = x / input->params.scale + input->params.zero_point;
   // Place the quantized input in the model's input tensor
   input->data.int8[0] = x_quantized;



   // Run inference, and report any error
   TfLiteStatus invoke_status = interpreter->Invoke();

   if (invoke_status != kTfLiteOk) {
       EV << "Invoke failed on x: " << static_cast<double>(x) << "\n" << omnetpp::endl;
     return;
   }

   // Obtain the quantized output from model's output tensor
   int8_t y_quantized = output->data.int8[0];
   // Dequantize the output from integer to floating-point
   float y = (y_quantized - output->params.zero_point) * output->params.scale;

   // Output the results. A custom HandleOutput function can be implemented
   // for each supported hardware target.
   PrintOutput(x, y);

   // Increment the inference_counter, and reset it if we have reached
   // the total number per cycle
   inference_count += 1;
   if (inference_count >= kInferencesPerCycle) inference_count = 0;
   /*
   */
}


LearningModel::~LearningModel() {
    // TODO Auto-generated destructor stub
}




} /* namespace inet */
