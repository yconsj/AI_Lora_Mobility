#ifdef __cplusplus
extern "C" {
#endif

/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_1x1_s8_fast.c
 * Description:  Fast q7 version of 1x1 convolution (non-square shape)
 *
 * $Date:        20 june 2022
 * $Revision:    V.3.0.1
 *
 * Target Processor:  Cortex-M Processors
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"
#include <stdio.h>

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Fast s8 version for 1x1 convolution (non-square shape)
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_convolve_1x1_s8_fast(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const q7_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const q7_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const int32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             q7_t *output_data)
{
    if (conv_params->padding.w != 0 || conv_params->padding.h != 0 || conv_params->stride.w != 1 ||
        conv_params->stride.h != 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    (void)ctx;
    (void)filter_dims;
    (void)bias_dims;

#if defined(ARM_MATH_MVEI)

    const int32_t col_len = input_dims->w * input_dims->h * input_dims->n;
    const int32_t output_ch = output_dims->c;
    const int32_t input_ch = input_dims->c;

    for (int i_items = 0; i_items <= (col_len - 4); i_items += 4)
    {
        output_data = arm_nn_mat_mul_core_4x_s8(input_ch,
                                                input_ch,
                                                input_data + i_items * input_ch,
                                                filter_data,
                                                output_ch,
                                                conv_params,
                                                quant_params,
                                                bias_data,
                                                output_data);
    }

    /* Handle left over elements */
    for (int i_items = (col_len & ~0x3); i_items < col_len; i_items++)
    {
        arm_nn_mat_mul_core_1x_s8(input_ch,
                                  0,
                                  input_data + i_items * input_ch,
                                  filter_data,
                                  output_ch,
                                  conv_params,
                                  quant_params,
                                  bias_data,
                                  output_data);
        output_data += output_ch;
    }

#else
    /* Run the following code as reference implementation for Cortex-M processors with or without DSP extension */

    const int32_t lhs_rows = input_dims->w * input_dims->h * input_dims->n;
    const int32_t rhs_rows = output_dims->c;
    const int32_t rhs_cols = input_dims->c;

    arm_nn_mat_mult_nt_t_s8(input_data,
                            filter_data,
                            bias_data,
                            output_data,
                            quant_params->multiplier,
                            quant_params->shift,
                            lhs_rows,
                            rhs_rows,
                            rhs_cols,
                            conv_params->input_offset,
                            conv_params->output_offset,
                            conv_params->activation.min,
                            conv_params->activation.max);

#endif

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const cmsis_nn_dims *input_dims)
{
    (void)input_dims;
    return 0;
}

/**
 * @} end of NNConv group
 */

#ifdef __cplusplus
}
#endif
