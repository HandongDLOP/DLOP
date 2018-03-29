#ifndef RECURRENT_H_
#define RECURRENT_H_    value

#include "..//Operator.h"
#include "..//Operator_utils.h"

template<typename DTYPE> class Recurrent : public Operator<DTYPE>{
private:
    int m_hiddenSize;

    Tensor<DTYPE> *input_Result;
    Tensor<DTYPE> *pre_hidden_Result;
    Tensor<DTYPE> *pre_net_Result;
    Tensor<DTYPE> *net_Result;
    Tensor<DTYPE> *hidden_Result;
    Tensor<DTYPE> *output_Result;

    Tensor<DTYPE> *input_temp;
    Tensor<DTYPE> *hidden_temp;
    Tensor<DTYPE> *output_temp;

    Tensor<DTYPE> *output_Gradient;
    Tensor<DTYPE> *linear_output_Gradient;
    Tensor<DTYPE> *next_linear_hidden_Gradient;
    Tensor<DTYPE> *pre_tanhDer_Gradient;
    Tensor<DTYPE> *DerTanh;
    Tensor<DTYPE> *hidden_Gradient;
    Tensor<DTYPE> *linear_hidden_Gradient;

    Tensor<DTYPE> *transpose_input;
    Tensor<DTYPE> *transpose_hidden;
    Tensor<DTYPE> *transpose_output;

    Tensor<DTYPE> *zero_tensor;

public:
    Recurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightInput, Operator<DTYPE> *pWeightHidden, Operator<DTYPE> *pWeightOutput, Operator<DTYPE> *pBiasInput, Operator<DTYPE> *pBiasHidden, Operator<DTYPE> *pBiasOutput, int pHiddenSize, std::string pName)
        : Operator<DTYPE>(pInput, pWeightInput, pWeightHidden, pWeightOutput, pBiasInput, pBiasHidden, pBiasOutput, pName) {
        Alloc(pInput, pWeightInput, pWeightHidden, pWeightOutput, pBiasInput, pBiasHidden, pBiasOutput, pHiddenSize);
    }

    ~Recurrent() {
        std::cout << "Recurrent::~Recurrent()" << '\n';
        Delete();
    }

    void Delete() {
        if (input_Result) {
            delete input_Result;
            input_Result = NULL;
        }

        if (pre_hidden_Result) {
            delete pre_hidden_Result;
            pre_hidden_Result = NULL;
        }

        if (pre_net_Result) {
            delete pre_net_Result;
            pre_net_Result = NULL;
        }

        if (net_Result) {
            delete net_Result;
            net_Result = NULL;
        }

        if (hidden_Result) {
            delete hidden_Result;
            hidden_Result = NULL;
        }

        if (output_Result) {
            delete output_Result;
            output_Result = NULL;
        }

        if(input_temp){
          delete input_temp;
          input_temp = NULL;
        }

        if(hidden_temp){
          delete hidden_temp;
          hidden_temp = NULL;
        }

        if(output_temp){
          delete output_temp;
          output_temp = NULL;
        }

        // if(output_Gradient){
        //   delete output_Gradient;
        //   output_Gradient = NULL;
        // }

        if(linear_output_Gradient){
          delete linear_output_Gradient;
          linear_output_Gradient = NULL;
        }

        if(next_linear_hidden_Gradient){
          delete next_linear_hidden_Gradient;
          next_linear_hidden_Gradient = NULL;
        }

        if(pre_tanhDer_Gradient){
          delete pre_tanhDer_Gradient;
          pre_tanhDer_Gradient = NULL;
        }

        if(DerTanh){
          delete DerTanh;
          DerTanh = NULL;
        }

        if(hidden_Gradient){
          delete hidden_Gradient;
          hidden_Gradient = NULL;
        }

        if(linear_hidden_Gradient){
          delete linear_hidden_Gradient;
          linear_hidden_Gradient = NULL;
        }

        if(zero_tensor){
          delete zero_tensor;
          zero_tensor = NULL;
        }

        if(transpose_input){
          delete transpose_input;
          transpose_input = NULL;
        }

        if(transpose_hidden){
          delete transpose_hidden;
          transpose_hidden = NULL;
        }

        if(transpose_output){
          delete transpose_output;
          transpose_output = NULL;
        }

    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightInput, Operator<DTYPE> *pWeightHidden, Operator<DTYPE> *pWeightOutput,
              Operator<DTYPE> *pBiasInput, Operator<DTYPE> *pBiasHidden, Operator<DTYPE> *pBiasOutput, int pHiddenSize) {
        // input은 onehot이어야 한다.
        Shape *shapeOfInput        = pInput->GetResult()->GetShape();
        Shape *shapeOfWeightInput  = pWeightInput->GetResult()->GetShape();
        Shape *shapeOfWeightHidden = pWeightInput->GetResult()->GetShape();
        Shape *shapeOfWeightOutput = pWeightOutput->GetResult()->GetShape();

        output_Result = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightOutput)[4]);
        this->AddResult(output_Result);

        input_Result = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]); //(*shapeOfWeightInput)[4] == pHiddenSize
        this->AddResult(input_Result);

        pre_hidden_Result = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddResult(pre_hidden_Result);

        pre_net_Result = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddResult(pre_net_Result);

        net_Result = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]);
        this->AddResult(net_Result);

        hidden_Result = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddResult(hidden_Result);

        // -----------bias temp-----------
        // dependency 문제 때문에 temp 생성
        input_temp = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]);
        this->AddResult(input_temp);

        hidden_temp = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]);
        this->AddResult(hidden_temp);

        output_temp = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]);
        this->AddResult(output_temp);
        // ---------------------------------

        // outputGradient * outputWeight
        // output_Gradient = this->GetGradient();
        output_Gradient = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddGradient(output_Gradient);

        linear_output_Gradient = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddGradient(linear_output_Gradient);

        next_linear_hidden_Gradient = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddGradient(next_linear_hidden_Gradient);

        pre_tanhDer_Gradient = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]);
        this->AddGradient(pre_tanhDer_Gradient);

        // vector
        DerTanh = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]);
        this->AddGradient(DerTanh);

        hidden_Gradient = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddGradient(hidden_Gradient);

        linear_hidden_Gradient = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddGradient(linear_hidden_Gradient);

        zero_tensor = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddGradient(zero_tensor);

        // transpose (ro, co 교체)
        transpose_input = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[4], (*shapeOfInput)[3], 0);
        this->AddGradient(transpose_input);

        transpose_hidden = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfWeightInput)[4], (*shapeOfInput)[3], 0);
        this->AddGradient(transpose_hidden);

        transpose_output = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfWeightInput)[4], (*shapeOfInput)[3],  0);
        this->AddGradient(transpose_output);


        // AddDelta
        // output_Gradient = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightOutput)[4]);
        // this->AddGradient(output_Gradient);

        m_hiddenSize = pHiddenSize;

        return TRUE;
    }

    int ComputeForwardPropagate() {
        printf("ComputeForwardPropagate()\n");
        Tensor<DTYPE> *input        = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weightInput  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weightHidden = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *weightOutput = this->GetInput()[3]->GetResult();
        Tensor<DTYPE> *biasInput    = this->GetInput()[4]->GetResult();
        Tensor<DTYPE> *biasHidden   = this->GetInput()[5]->GetResult();
        Tensor<DTYPE> *biasOutput   = this->GetInput()[6]->GetResult();

        Shape *shapeOfInput = input->GetShape();
        int timesize = (*shapeOfInput)[0];

        for (int ti = 0; ti < timesize; ti++) {
            MatMul(input, weightInput, input_temp, ti);
            BiasAdd(input_temp, biasInput, input_Result, ti);

            MatMul(pre_hidden_Result, weightHidden, hidden_temp, ti);
            BiasAdd(hidden_temp, biasHidden, pre_net_Result, ti);

            Add(input_Result, pre_net_Result, net_Result, ti);

            Tanh(net_Result, hidden_Result, ti);
            if(ti < timesize - 1){
                  Copy(hidden_Result, pre_hidden_Result, ti, ti + 1);   //src ,dst
            }

            MatMul(hidden_Result, weightOutput, output_temp, ti);
            BiasAdd(output_temp, biasOutput, output_Result, ti);
          }
        return TRUE;
    }

    int ComputeBackPropagate() {
        printf("ComputeBackPropagate()\n");
        Tensor<DTYPE> *input        = this->GetInput()[0]->GetResult();

        Tensor<DTYPE> *weightInput  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weightHidden = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *weightOutput = this->GetInput()[3]->GetResult();
        Tensor<DTYPE> *biasInput    = this->GetInput()[4]->GetResult();
        Tensor<DTYPE> *biasHidden   = this->GetInput()[5]->GetResult();
        Tensor<DTYPE> *biasOutput   = this->GetInput()[6]->GetResult();

        Tensor<DTYPE> *weightInputGradient  = this->GetInput()[1]->GetGradient();
        Tensor<DTYPE> *weightHiddenGradient = this->GetInput()[2]->GetGradient();
        Tensor<DTYPE> *weightOutputGradient = this->GetInput()[3]->GetGradient();
        Tensor<DTYPE> *biasInputGradient    = this->GetInput()[4]->GetGradient();
        Tensor<DTYPE> *biasHiddenGradient   = this->GetInput()[5]->GetGradient();
        Tensor<DTYPE> *biasOutputGradient   = this->GetInput()[6]->GetGradient();

        Tensor<DTYPE> *input_delta          = this->GetInput()[0]->GetGradient();

        Shape *shapeOfInput = input->GetShape();
        int timesize = (*shapeOfInput)[0];

        output_Gradient = this->GetGradient();

        for(int ti = timesize - 1; ti >= 0; ti--){
          MatMul(output_Gradient, weightOutput, linear_output_Gradient, ti);

          Add(linear_output_Gradient, next_linear_hidden_Gradient, pre_tanhDer_Gradient, ti);

          DerivativeTanh(net_Result, DerTanh, ti);

          Elementwise(pre_tanhDer_Gradient, DerTanh, hidden_Gradient, ti);

          MatMul(hidden_Gradient, weightHidden, linear_hidden_Gradient, ti);

          if(ti > 0){
            Copy(linear_hidden_Gradient, next_linear_hidden_Gradient, ti, ti - 1);  // ti -> ti - 1
          }

          Transpose(input, transpose_input, ti);
          CellMatMul(transpose_input, hidden_Gradient, weightInputGradient, ti);
          CellBiasAdd(zero_tensor, hidden_Gradient, biasInputGradient, ti);

          Transpose(pre_hidden_Result, transpose_hidden, ti);
          CellMatMul(transpose_hidden, hidden_Gradient, weightHiddenGradient, ti);
          CellBiasAdd(zero_tensor, hidden_Gradient, biasHiddenGradient, ti);

          Transpose(hidden_Result, transpose_output, ti);
          CellMatMul(transpose_output, output_Gradient, weightOutputGradient, ti);
          CellBiasAdd(zero_tensor, output_Gradient, biasOutputGradient, ti);

          CellMatMul(weightInput, hidden_Gradient, input_delta, ti);
        }

        return TRUE;
    }

    int Copy(Tensor<DTYPE> *input, Tensor<DTYPE> *output, int input_ti, int output_ti){
        Shape *shapeOfInput  = input->GetShape();
        Shape *shapeOfOutput = output->GetShape();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                          (*output)[Index5D(shapeOfOutput, output_ti, ba, ch, ro, co)] = (*input)[Index5D(shapeOfInput, input_ti, ba, ch, ro, co)];
                    }
                }
            }
        }

        return TRUE;
    }

    int Tanh(Tensor<DTYPE> *input, Tensor<DTYPE> *result, int ti){
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int capacity          = input->GetCapacity();

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                          (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = this->TANH((*input)[Index5D(input->GetShape(), ti, ba, ch, ro, co)]);
                    }
                }
            }
        }

        return TRUE;
    }

    inline DTYPE TANH(DTYPE data){
        return ((DTYPE)exp(data) - (DTYPE)exp(-data)) / ((DTYPE)exp(data) + (DTYPE)exp(-data));
    }

    int DerivativeTanh(Tensor<DTYPE> *input, Tensor<DTYPE> *result, int ti){
std::cout<< "DerivativeTanh" << '\n';
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int capacity          = result->GetCapacity();

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                            (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = (1 - (*input)[Index5D(input->GetShape(), ti, ba, ch, ro, co)] * (*input)[Index5D(input->GetShape(), ti, ba, ch, ro, co)]);
                    }
                }
            }
        }

        return TRUE;
    }

    int MatMul(Tensor<DTYPE> *input, Tensor<DTYPE> *weight, Tensor<DTYPE> *result, int ti) {
        Shape *shapeOfInput  = input->GetShape();
        Shape *shapeOfWeight = weight->GetShape();

        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = input->GetColSize();

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        DTYPE temp = 0.f;

        // ------------MatMul------------
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++)   {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                          // index5D로 바꾸기
                            input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * hiddensize + hid;
                            weight_index = hid * colsize + co;
                            temp        += (*input)[input_index] * (*weight)[weight_index];
                            }
                            (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = temp;
                            temp = 0.f;
                        }
                    }
                }
            }
        return TRUE;
    }

    int CellMatMul(Tensor<DTYPE> *input, Tensor<DTYPE> *weight, Tensor<DTYPE> *result, int ti) {
        Shape *shapeOfInput  = input->GetShape();
        Shape *shapeOfWeight = weight->GetShape();

        int timesize    = input->GetTimeSize();
        int batchsize   = input->GetBatchSize();    // 이 부분이 달라짐

        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = input->GetColSize();

        int resulttimesize = result->GetTimeSize();
        int resultbatchsize = result->GetBatchSize();

std::cout << "CellMatMul()\n" << '\n';
        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        DTYPE temp = 0.f;

        // ------------MatMul------------
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++)   {
                    for (int co = 0; co < colsize; co++) {
                        for (int hid = 0; hid < hiddensize; hid++) {
                          // index5D로 바꾸기
                            input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * hiddensize + hid;
                            weight_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * hiddensize) + hid) * colsize + co;

                            temp        += (*input)[input_index] * (*weight)[weight_index];
                            }

                            (*result)[Index5D(result->GetShape(), resulttimesize - 1, resultbatchsize - 1, ch, ro, co)] += temp;
                            temp = 0.f;
                        }
                    }
                }
            }
        return TRUE;
    }

    int Elementwise(Tensor<DTYPE> *input, Tensor<DTYPE> *weight, Tensor<DTYPE> *result, int ti) {
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

std::cout << "Elementwise()\n" << '\n';
        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++)   {
                    for (int co = 0; co < colsize; co++) {
                            input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * colsize + co;
                            // index5D로 바꾸기
                            weight_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * colsize + co;
                          (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = (*input)[input_index] * (*weight)[weight_index];
                    }
                }
            }
        }
        return TRUE;
    }

    int BiasAdd(Tensor<DTYPE> *input, Tensor<DTYPE> *bias, Tensor<DTYPE> *result, int ti){
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();
        int input_index = 0, bias_index = 0;

        Shape *m_pInputTenShape = input->GetShape();
        Shape *m_pBiasTenShape  = bias->GetShape();

        int m_timesize    = (*m_pInputTenShape)[0];
        int m_batchsize   = (*m_pInputTenShape)[1];
        int m_channelsize = (*m_pInputTenShape)[2];
        int m_rowsize     = (*m_pInputTenShape)[3];
        int m_colsize     = (*m_pInputTenShape)[4];

        int m_ti = ti;
        int m_ba = 0;
        int m_ch = 0;
        int m_ro = 0;
        int m_co = 0;

        int *m_ti_bias = &m_ti;
        int *m_ba_bias = &m_ba;
        int *m_ch_bias = &m_ch;
        int *m_ro_bias = &m_ro;
        int *m_co_bias = &m_co;

        int m_zero = 0;

        if ((*m_pBiasTenShape)[0] == 1) m_ti_bias = &m_zero;

        if ((*m_pBiasTenShape)[1] == 1) m_ba_bias = &m_zero;

        if ((*m_pBiasTenShape)[2] == 1) m_ch_bias = &m_zero;

        if ((*m_pBiasTenShape)[3] == 1) m_ro_bias = &m_zero;

        if ((*m_pBiasTenShape)[4] == 1) m_co_bias = &m_zero;

            for (m_ba = 0; m_ba < m_batchsize; m_ba++) {
                for (m_ch = 0; m_ch < m_channelsize; m_ch++) {
                    for (m_ro = 0; m_ro < m_rowsize; m_ro++) {
                        for (m_co = 0; m_co < m_colsize; m_co++) {
                            (*result)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                            = (*input)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                                      + (*bias)[Index5D(m_pBiasTenShape, *m_ti_bias, *m_ba_bias, *m_ch_bias, *m_ro_bias, *m_co_bias)];
                        }
                    }
                }
            }


        return TRUE;
    }

    int CellBiasAdd(Tensor<DTYPE> *input, Tensor<DTYPE> *bias, Tensor<DTYPE> *result, int ti){
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();
        int input_index = 0, bias_index = 0;

        Shape *m_pInputTenShape = input->GetShape();
        Shape *m_pBiasTenShape  = bias->GetShape();

        int m_timesize    = (*m_pInputTenShape)[0];
        int m_batchsize   = (*m_pInputTenShape)[1];
        int m_channelsize = (*m_pInputTenShape)[2];
        int m_rowsize     = (*m_pInputTenShape)[3];
        int m_colsize     = (*m_pInputTenShape)[4];

        int m_ti = ti;
        int m_ba = 0;
        int m_ch = 0;
        int m_ro = 0;
        int m_co = 0;

        int *m_ti_bias = &m_ti;
        int *m_ba_bias = &m_ba;
        int *m_ch_bias = &m_ch;
        int *m_ro_bias = &m_ro;
        int *m_co_bias = &m_co;

        int m_zero = 0;

        if ((*m_pBiasTenShape)[0] == 1) m_ti_bias = &m_zero;

        if ((*m_pBiasTenShape)[1] == 1) m_ba_bias = &m_zero;

        if ((*m_pBiasTenShape)[2] == 1) m_ch_bias = &m_zero;

        if ((*m_pBiasTenShape)[3] == 1) m_ro_bias = &m_zero;

        if ((*m_pBiasTenShape)[4] == 1) m_co_bias = &m_zero;

            for (m_ba = 0; m_ba < m_batchsize; m_ba++) {
                for (m_ch = 0; m_ch < m_channelsize; m_ch++) {
                    for (m_ro = 0; m_ro < m_rowsize; m_ro++) {
                        for (m_co = 0; m_co < m_colsize; m_co++) {
                            input_index = ((((((ti * m_batchsize) + m_ba) * m_channelsize) + m_ch) * m_rowsize) + m_ro) * m_colsize + m_co;
                            bias_index  = m_co;
                            (*result)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                    = (*input)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                        + (*bias)[Index5D(m_pBiasTenShape, *m_ti_bias, *m_ba_bias, *m_ch_bias, *m_ro_bias, *m_co_bias)];
                        }
                    }
                }
            }

        return TRUE;
    }

    int Add(Tensor<DTYPE> *input0, Tensor<DTYPE> *input1, Tensor<DTYPE> *result, int ti){
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();
        int input_index0 = 0, input_index1 = 0;

        Shape *shapeOfResult = result->GetShape();

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                    (*result)[Index5D(shapeOfResult, ti, ba, ch, ro, co)] = (*input0)[Index5D(input0->GetShape(), ti, ba, ch, ro, co)] + (*input1)[Index5D(input1->GetShape(), ti, ba, ch, ro, co)];

                    }
                }
            }
        }

        return TRUE;
    }

    int Transpose(Tensor<DTYPE> *input, Tensor<DTYPE>* result, int ti){
      int inputRowsize     = input->GetRowSize();
      int inputChannelsize = input->GetChannelSize();

      int batchsize   = result->GetBatchSize();
      int channelsize = result->GetChannelSize();
      int rowsize     = result->GetRowSize();
      int colsize     = result->GetColSize(); // 1

      // try catch로 바꾸기
      if(inputRowsize == 1 && inputChannelsize == 1){
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int co = 0; co < colsize; co++) {  // 4
                    for (int ro = 0; ro < rowsize; ro++) {   // 2
                            (*result)[Index5D(result->GetShape(),ti, ba, ch, ro, co)] = (*input)[Index5D(input->GetShape(), ti, ba, ch, co, ro)];
                    }
                }
              }
          }
      }

      return TRUE;
    }
};

#endif  // ifndef RECURRENT_H_
