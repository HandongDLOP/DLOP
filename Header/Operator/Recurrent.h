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

    // AddGradient
    Tensor<DTYPE> *output_Gradient;
    Tensor<DTYPE> *hidden_Gradient;

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

        if(output_Gradient){
          delete output_Gradient;
          output_Gradient = NULL;
        }

        if(hidden_Gradient){
          delete hidden_Gradient;
          hidden_Gradient = NULL;
        }
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightInput, Operator<DTYPE> *pWeightHidden, Operator<DTYPE> *pWeightOutput,
              Operator<DTYPE> *pBiasInput, Operator<DTYPE> *pBiasHidden, Operator<DTYPE> *pBiasOutput, int pHiddenSize) {
        // input은 onehot이어야 한다.
        Shape *shapeOfInput        = pInput->GetResult()->GetShape();
        Shape *shapeOfWeightInput  = pWeightInput->GetResult()->GetShape();
        Shape *shapeOfWeightHidden = pWeightInput->GetResult()->GetShape();
        Shape *shapeOfWeightOutput = pWeightOutput->GetResult()->GetShape();

        input_Result = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]);
        this->AddResult(input_Result);

        pre_hidden_Result = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddResult(pre_hidden_Result);

        pre_net_Result = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddResult(pre_net_Result);

        net_Result = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]);
        this->AddResult(net_Result);

        hidden_Result = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4], 0);
        this->AddResult(hidden_Result);

        output_Result = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightOutput)[4]);
        this->AddResult(output_Result);

        // AddDelta
        output_Gradient = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightOutput)[4]);
        this->AddGradient(output_Gradient);

        hidden_Gradient = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightInput)[4]);
        this->AddGradient(hidden_Gradient);

        m_hiddenSize = pHiddenSize;

        return TRUE;
    }

    int ComputeForwardPropagate() {
        printf("ComputeForwardPropagate()\n");
        Tensor<DTYPE> *input        = this->GetInput()[0] -> GetResult();

        Tensor<DTYPE> *weightInput  = this->GetInput()[1] -> GetResult();
        Tensor<DTYPE> *weightHidden = this->GetInput()[2] -> GetResult();
        Tensor<DTYPE> *weightOutput = this->GetInput()[3] -> GetResult();

        Tensor<DTYPE> *biasInput  = this->GetInput()[4] -> GetResult();
        Tensor<DTYPE> *biasHidden = this->GetInput()[5] -> GetResult();
        Tensor<DTYPE> *biasOutput = this->GetInput()[6] -> GetResult();

        int timesize = (*shapeOfInput)[0];

        //===================================for test============================================
        // Operator<DTYPE> *R_input   = new Tensorholder<DTYPE>(input_Result,"R_input");
        // Operator<DTYPE> *R_pre_net = new Tensorholder<DTYPE>(pre_net_Result,"R_pre_net");
        // Operator<DTYPE> *R_net    = new Tensorholder<DTYPE>(net_Result,"R_net");
        // Operator<DTYPE> *R_hidden = new Tensorholder<DTYPE>(hidden_Result,"R_hidden");
        // Operator<DTYPE> *R_hidden_to_prehidden = new Tensorholder<DTYPE>(pre_hidden_Result,"R_hidden_to_prehidden");
        // Operator<DTYPE> *R_output = new Tensorholder<DTYPE>(output_Result,"R_output");

        for (int ti = 0; ti < timesize; ti++) {
            Linear(input, weightInput, biasInput, input_Result, ti, TRUE);
            printf("Result_of_input\n");
            std::cout<< input_Result << std::endl;

            Linear(pre_hidden_Result, weightHidden, biasHidden, pre_net_Result, ti, TRUE);    // WX + b
            printf("Result_of_pre_net\n");
            std::cout<< pre_net_Result << std::endl;

            Add(input_Result, pre_net_Result, net_Result, ti, TRUE); // net_result   //Tensor + Tensor
            printf("Result_of_net\n");
            std::cout<< net_Result << std::endl;

            Tanh(net_Result, hidden_Result, ti, TRUE);          // hidden_result
            printf("Result_of_hidden\n");
            std::cout<< hidden_Result << std::endl;

            if(ti < timesize - 1){
                  CopyNextTensor(hidden_Result, pre_hidden_Result, ti, TRUE);   //src ,dst
                  printf("Result_of_hidden_to_prehidden\n");
                  std::cout<< pre_hidden_Result << std::endl;
            }// hidden_result -> prehidden_result

            Linear(hidden_Result, weightOutput, biasOutput, output_Result, ti, TRUE);
            printf("Result_of_output\n");
            std::cout<< output_Result << std::endl;
          }

        return TRUE;
    }

    int ComputeBackPropagate() {
        // 델타 3개
        output_Gradient =
        for(ti = timesize; ti > -1; ti++){
          Linear()
        }

        return TRUE;
    }


    int CopyNextTensor(Tensor<DTYPE> *hidden_Result, Tensor<DTYPE> *pre_hidden_Result, int ti, int isForward = TRUE){

        Shape *shapeOfHiddenResult  = hidden_Result->GetShape();
        Shape *shapeOfPreHiddenResult = pre_hidden_Result->GetShape();

        int batchsize   = hidden_Result->GetBatchSize();
        int channelsize = hidden_Result->GetChannelSize();
        int rowsize     = hidden_Result->GetRowSize();
        int colsize     = hidden_Result->GetColSize();


        if(isForward){
          for (int ba = 0; ba < batchsize; ba++) {
              for (int ch = 0; ch < channelsize; ch++) {
                  for (int ro = 0; ro < rowsize; ro++) {
                      for (int co = 0; co < colsize; co++) {
                              (*pre_hidden_Result)[Index5D(shapeOfPreHiddenResult, ti + 1, ba, ch, ro, co)] = (*hidden_Result)[Index5D(shapeOfHiddenResult, ti, ba, ch, ro, co)];
                      }
                  }
              }
          }
        }

        else{
          //backpropagate



        }
        return TRUE;
    }



    int Tanh(Tensor<DTYPE> *input, Tensor<DTYPE> *result, int ti, int isForward = TRUE){

        // result->GetBatchSize();
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int capacity          = input->GetCapacity();

        if(isForward){
          for (int ba = 0; ba < batchsize; ba++) {
              for (int ch = 0; ch < channelsize; ch++) {
                  for (int ro = 0; ro < rowsize; ro++) {
                      for (int co = 0; co < colsize; co++) {
                              (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = this->TANH((*input)[Index5D(input->GetShape(), ti, ba, ch, ro, co)]);
                      }
                  }
              }
          }
        }

      else{
        //backpropagate





      }
        return TRUE;
    }

    inline DTYPE TANH(DTYPE data){
      //DTYPE sinh = ((DTYPE)exp(data) - (DTYPE)exp(-data)) / 2;
      //DTYPE cosh = ((DTYPE)exp(data) + (DTYPE)exp(-data)) / 2;
      //return sinh/cosh;
      return ((DTYPE)exp(data) - (DTYPE)exp(-data)) / ((DTYPE)exp(data) + (DTYPE)exp(-data));
    }

//Linear(pre_hidden_Result, weightHidden, biasHidden, pre_net_Result, ti, TRUE);    // WX + b

    int Linear(Tensor<DTYPE> *input, Tensor<DTYPE> *weight, Tensor<DTYPE> *bias, Tensor<DTYPE> *result, int ti, int isForward = TRUE) {
        Shape *shapeOfInput  = input->GetShape();
        Shape *shapeOfWeight = weight->GetShape();

        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = input->GetColSize();
        std::cout << "result colsize : " << colsize;

        //int bias_capacity = bias->GetCapacity();

        if(isForward){

          int input_index  = 0;
          int weight_index = 0;
          int result_index = 0;
          int index = 0;

          DTYPE temp = 0.f;

          // ------------MatMul------------
          for (int ba = 0; ba < batchsize; ba++) {
              for (int ch = 0; ch < channelsize; ch++) {
                  for (int ro = 0; ro < rowsize; ro++) {
                      for (int co = 0; co < colsize; co++) {
                          for (int hid = 0; hid < hiddensize; hid++) {
                              input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * hiddensize + hid;

                              weight_index = hid * colsize + co;
                              temp        += (*input)[input_index] * (*weight)[weight_index];
                          }

                          (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = temp;

                          //std::cout<<(*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)]<<'\n';
                          temp = 0.f;
                      }
                  }
              }
          }

          // -----------------------------

          // ---------- Add --------------
          for (int ba = 0; ba < batchsize; ba++) {
              for (int ch = 0; ch < channelsize; ch++) {
                  for (int ro = 0; ro < rowsize; ro++) {
                      for (int co = 0; co < colsize; co++) {
                              index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * colsize + co;

                              (*result)[index] += (*bias)[co];

                      }
                  }
              }
          }

        }

        else{
          //backpropagation




        }
        // -----------------------------
        return TRUE;
    }

    int Add(Tensor<DTYPE> *input0, Tensor<DTYPE> *input1, Tensor<DTYPE> *result, int ti, int isForward = TRUE){
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *shapeOfResult = result->GetShape();

        if(isForward){
          for (int ba = 0; ba < batchsize; ba++) {
              for (int ch = 0; ch < channelsize; ch++) {
                  for (int ro = 0; ro < rowsize; ro++) {
                      for (int co = 0; co < colsize; co++) {
                              (*result)[Index5D(shapeOfResult, ti, ba, ch, ro, co)] = (*input0)[Index5D(input0->GetShape(), ti, ba, ch, ro, co)] + (*input1)[Index5D(input1->GetShape(), ti, ba, ch, ro, co)];
                      }
                  }
              }
          }
        }

        else{
          //backpropagation


        }

        return TRUE;
    }




};

#endif  // ifndef RECURRENT_H_
