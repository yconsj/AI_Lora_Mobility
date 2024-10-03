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

#include "LearningModel.h"
#include <onnxruntime_cxx_api.h>

#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


namespace inet {

Define_Module(LearningModel);

LearningModel::LearningModel()
{
    // stub
}

void LearningModel::initialize(int stage)
{

    EV_TRACE << "initializing LearningModel stage " << stage << endl;
    if (stage == INITSTAGE_LOCAL) {
        // TODO: initialize ONNX model session here
        EV << "ONNX Runtime version: " << Ort::GetVersionString() << endl;
    }
}

double LearningModel::pollModel()
{
    return uniform(0,1);
}


void LearningModel::Test() {



}


LearningModel::~LearningModel() {
    // TODO Auto-generated destructor stub
}



} /* namespace inet */
