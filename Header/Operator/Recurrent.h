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

    Tensor<DTYPE> *zero_tensor;


    // AddGradient



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

        // -----------bias temp-----------
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

        // AddDelta
        // output_Gradient = new Tensor<DTYPE>((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfInput)[3], (*shapeOfWeightOutput)[4]);
        // this->AddGradient(output_Gradient);

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

        Shape *shapeOfInput = input->GetShape();
        int timesize = (*shapeOfInput)[0];

        for (int ti = 0; ti < timesize; ti++) {
            //Linear(input, weightInput, biasInput, input_Result, ti, TRUE);
            MatMul(input, weightInput, input_temp, ti, TRUE);
            Add(input_temp, biasInput, input_Result, ti, TRUE);
// printf("input_temp\n");
// std::cout<< input_temp << std::endl;
// printf("Result_of_input\n");
// std::cout<< input_Result << std::endl;

            //Linear(pre_hidden_Result, weightHidden, biasHidden, pre_net_Result, ti, TRUE);    // WX + b
            MatMul(pre_hidden_Result, weightHidden, hidden_temp, ti, TRUE);
            Add(hidden_temp, biasHidden, pre_net_Result, ti, TRUE);
// printf("Result_of_pre_net\n");
// std::cout<< pre_net_Result << std::endl;

            Add(input_Result, pre_net_Result, net_Result, ti, TRUE); // net_result   //Tensor + Tensor
// printf("Result_of_net\n");
// std::cout<< net_Result << std::endl;

            Tanh(net_Result, hidden_Result, ti, TRUE);          // hidden_result
// printf("Result_of_hidden\n");
// std::cout<< hidden_Result << std::endl;

            if(ti < timesize - 1){
                  CopyTensor(hidden_Result, pre_hidden_Result, ti, ti + 1, TRUE);   //src ,dst
// printf("Result_of_hidden_to_prehidden\n");
// std::cout<< pre_hidden_Result << std::endl;
            }// hidden_result -> prehidden_result

            //Linear(hidden_Result, weightOutput, biasOutput, output_Result, ti, TRUE);
            MatMul(hidden_Result, weightOutput, output_temp, ti, TRUE);
            Add(output_temp, biasOutput, output_Result, ti, TRUE);
// printf("Result_of_output\n");
// std::cout<< output_Result << std::endl;
          }

        return TRUE;
    }

    int ComputeBackPropagate() {
        printf("ComputeBackPropagate()\n");
        Tensor<DTYPE> *input        = this->GetInput()[0]->GetResult();

        Tensor<DTYPE> *weightInput  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *weightHidden = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *weightOutput = this->GetInput()[3]->GetResult();

        Tensor<DTYPE> *biasInput  = this->GetInput()[4] -> GetResult();
        Tensor<DTYPE> *biasHidden = this->GetInput()[5] -> GetResult();
        Tensor<DTYPE> *biasOutput = this->GetInput()[6] -> GetResult();

        Tensor<DTYPE> *weightInputGradient  = this->GetInput()[1]->GetGradient();
        Tensor<DTYPE> *weightHiddenGradient = this->GetInput()[2]->GetGradient();
        Tensor<DTYPE> *weightOutputGradient = this->GetInput()[3]->GetGradient();

        Tensor<DTYPE> *biasInputGradient  = this->GetInput()[4]->GetGradient();
        Tensor<DTYPE> *biasHiddenGradient = this->GetInput()[5]->GetGradient();
        Tensor<DTYPE> *biasOutputGradient = this->GetInput()[6]->GetGradient();

        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetGradient();

        Shape *shapeOfInput = input->GetShape();
        int timesize = (*shapeOfInput)[0];


        output_Gradient = this->GetGradient();



// Container<Tensor<DTYPE> *>   *gradient_container = this->GetGradientContainer();
// printf("Container gradient Size: ");
// std::cout << gradient_container->GetSize() << '\n';

        for(int ti = timesize - 1; ti >= 0; ti--){
          // ti에서 hiddenGradient

          printf("output_Gradient\n");
          std::cout<< output_Gradient << std::endl;

          printf("weightOutput\n");
          std::cout<< weightOutput << std::endl;
          //
          // printf("before\n");
          // std::cout<< linear_output_Gradient << std::endl;

          MatMul(output_Gradient, weightOutput, linear_output_Gradient, ti, TRUE);
          printf("linear_output_Gradient\n");
          std::cout<< linear_output_Gradient << std::endl;

          Add(linear_output_Gradient, next_linear_hidden_Gradient, pre_tanhDer_Gradient, ti, TRUE);
          printf("pre_tanhDer_Gradient\n");
          std::cout<< pre_tanhDer_Gradient << std::endl;

          DerivativeTanh(net_Result, DerTanh, ti, TRUE);
          printf("DerTanh\n");
          std::cout<< DerTanh << std::endl;

          Elementwise(pre_tanhDer_Gradient, DerTanh, hidden_Gradient, ti, TRUE);   //Matrix Multiplication ?? or Tensor * scalar ??  // hidden_Gradient
          printf("hidden_Gradient\n");
          std::cout<< hidden_Gradient << std::endl;

          MatMul(hidden_Gradient, weightHidden, linear_hidden_Gradient, ti, TRUE);
          printf("linear_hidden_Gradient\n");
          std::cout<< linear_hidden_Gradient << std::endl;

          //////
          if(ti > 0){
            CopyTensor(linear_hidden_Gradient, next_linear_hidden_Gradient, ti, ti - 1, TRUE);  // ti -> ti - 1
            printf("next_linear_hidden_Gradient\n");
            std::cout<< next_linear_hidden_Gradient << std::endl;
          }

          //w1,w2,w3,b1,b2,b3의 gradient
          // 1 ~ ti까지 돌리기
          // weight_gradient bias_gradient 구하기
          MatMul(input, hidden_Gradient, weightInputGradient, ti, TRUE);
          printf("weightInputGradient\n");
          std::cout<< weightInputGradient << std::endl;

          //MatMul(hidden_Gradient, biasInput, biasInputGradient, ti);
          Add(zero_tensor, hidden_Gradient, biasInputGradient, ti, TRUE);
          printf("biasInputGradient\n");
          std::cout<< biasInputGradient << std::endl;

          MatMul(pre_hidden_Result, hidden_Gradient, weightHiddenGradient, ti, TRUE);         // ti - 1
          printf("weightHiddenGradient\n");
          std::cout<< weightHiddenGradient << std::endl;

          Add(zero_tensor, hidden_Gradient, biasHiddenGradient, ti, TRUE);
          printf("biasHiddenGradient\n");
          std::cout<< biasHiddenGradient << std::endl;

          MatMul(hidden_Result, output_Gradient, weightOutputGradient, ti, TRUE);
          printf("weightOutputGradient\n");
          std::cout<< weightOutputGradient << std::endl;

          Add(zero_tensor, output_Gradient, biasOutputGradient, ti, TRUE);
          printf("biasOutputGradient\n");
          std::cout<< biasOutputGradient << std::endl;

          MatMul(weightInput, hidden_Gradient, input_delta, ti, TRUE);
          printf("input_delta\n");
          std::cout<< input_delta << std::endl;
        }
        // input_detla 2개 넘겨주기.

        return TRUE;
    }

//=========================================================================================

    int CopyTensor(Tensor<DTYPE> *input, Tensor<DTYPE> *output, int input_ti, int output_ti, int isForward = TRUE){
        Shape *shapeOfInput  = input->GetShape();
        Shape *shapeOfOutput = output->GetShape();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        if(isForward){
          // ti -> ti + 1
          for (int ba = 0; ba < batchsize; ba++) {
              for (int ch = 0; ch < channelsize; ch++) {
                  for (int ro = 0; ro < rowsize; ro++) {
                      for (int co = 0; co < colsize; co++) {
                              (*output)[Index5D(shapeOfOutput, output_ti, ba, ch, ro, co)] = (*input)[Index5D(shapeOfInput, input_ti, ba, ch, ro, co)];
                      }
                  }
              }
          }
        }

        // else{
        //   // backpropagate
        //   // Operator가 아니고 Tensor를 직접 계산하는 형식이라 델타(Gradient)가 쓰이지 않음.
        //   // ti + 1 -> ti
        //   for (int ba = 0; ba < batchsize; ba++) {
        //       for (int ch = 0; ch < channelsize; ch++) {
        //           for (int ro = 0; ro < rowsize; ro++) {
        //               for (int co = 0; co < colsize; co++) {
        //                       (*output)[Index5D(shapeOfOutput, output_ti, ba, ch, ro, co)] = (*input)[Index5D(shapeOfInput, ti, ba, ch, ro, co)];
        //               }
        //           }
        //       }
        //   }
        // }
        return TRUE;
    }

//=========================================================================================
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
      //
      //   else{
      //     //backpropagate
      //     std::cout << "Tanh BackwardPropagate()\n" << '\n';
      //     //Tensor<DTYPE> *result      = this->GetResult();
      //     Tensor<DTYPE> *this_delta  = this->GetDelta();
      //     Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
      //     int capacity               = result->GetData()->GetCapacity();
      //
      //     for (int i = 0; i < capacity; i++){
      //       //printf("%d\n", i);
      //     //  printf("(*result)[%d] : %f", i, (*result)[i]);
      //       // std::cout << result << '\n';
      //       (*input_delta)[i] = (1 - ((*input)[i] * (*input)[i])) * (*this_delta)[i];
      //     }
      // }
        return TRUE;
    }

    inline DTYPE TANH(DTYPE data){
      return ((DTYPE)exp(data) - (DTYPE)exp(-data)) / ((DTYPE)exp(data) + (DTYPE)exp(-data));
    }

    //=========================================================================================
    int DerivativeTanh(Tensor<DTYPE> *input, Tensor<DTYPE> *result, int ti, int isForward = TRUE){

            // result->GetBatchSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int capacity          = result->GetCapacity();

            //backpropagate
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                                (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = (*input)[Index5D(input->GetShape(), ti, ba, ch, ro, co)] * (*input)[Index5D(input->GetShape(), ti, ba, ch, ro, co)];
                        }
                    }
                }
            }
            return TRUE;
        }

//=========================================================================================

    int MatMul(Tensor<DTYPE> *input, Tensor<DTYPE> *weight, Tensor<DTYPE> *result, int ti, int isForward = TRUE) {
      Shape *shapeOfInput  = input->GetShape();
      Shape *shapeOfWeight = weight->GetShape();

      int batchsize   = result->GetBatchSize();
      int channelsize = result->GetChannelSize();
      int rowsize     = result->GetRowSize();
      int colsize     = result->GetColSize();

      int hiddensize = input->GetColSize();

      //output_Gradient = this->GetGradient();
      if(isForward){
          std::cout << "MatMul()\n" << '\n';
          int input_index  = 0;
          int weight_index = 0;
          int result_index = 0;

          DTYPE temp = 0.f;

          // printf("result\n");
          // std::cout<< result << std::endl;


//std:: cout<< "ba, ch, ro, co, result, temp"<< std:: endl;
          // ------------MatMul------------
          for (int ba = 0; ba < batchsize; ba++) {
              for (int ch = 0; ch < channelsize; ch++) {
                  for (int ro = 0; ro < rowsize; ro++)   {
                      for (int co = 0; co < colsize; co++) {
                          for (int hid = 0; hid < hiddensize; hid++) {
                              input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * hiddensize + hid;
                              // index5D로 바꾸기
                              weight_index = hid * colsize + co;
                              std::cout << input->GetShape() << '\n' << weight->GetShape() << '\n';
                              //weight_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * hiddensize) + hid) * colsize + co;

                              std::cout << "input[" << input_index <<"] : " << (*input)[input_index] << "weight[" << weight_index <<"] : " << (*weight)[weight_index] << '\n';
                              temp        += (*input)[input_index] * (*weight)[weight_index];
                          }

//std:: cout<< std:: fixed<< ba << ", "<< ch<< ", "<< ro<< ", "<< co<< ", "<< (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)]<< ", ";
                          (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = temp;
//std:: cout<< temp<< std:: endl;
                          std::cout<<(*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)]<<'\n';
                          temp = 0.f;
                      }
                  }
              }
          }
        }

      // else{
      //     //backpropagation
      //     int input_index  = 0;
      //     int weight_index = 0;
      //     int result_index = 0;
      //
      //
      //     for (int ba = 0; ba < batchsize; ba++) {
      //         for (int ch = 0; ch < channelsize; ch++) {
      //             for (int ro = 0; ro < rowsize; ro++) {
      //                 for (int co = 0; co < colsize; co++) {
      //                     for (int hid = 0; hid < hiddensize; hid++) {
      //                         input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * hiddensize + hid;
      //                         //input_index  = (i * rowsize + ro) * hiddensize + hid;
      //                         weight_index = hid * colsize + co;
      //                         result_index = Index5D(result->GetShape(), ti, ba, ch, ro, co);
      //                         // 다음 Operator로 넘겨주는거라 필요 없음
      //                         (*input_delta)[input_index]      += (*weight)[weight_index] * (*this_delta)[result_index];
      //                         (*hidden_Gradient)[weight_index] += (*input)[input_index] * (*this_delta)[result_index];
      //                       }
      //                   }
      //               }
      //           }
      //     }
      //   }


      // -----------------------------
      return TRUE;
  }

  //=========================================================================================
  int Elementwise(Tensor<DTYPE> *input, Tensor<DTYPE> *weight, Tensor<DTYPE> *result, int ti, int isForward = TRUE) {
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    if(isForward){
        std::cout << "Elementwise()\n" << '\n';
        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        // ------------MatMul------------
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++)   {
                    for (int co = 0; co < colsize; co++) {
                            input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * colsize + co;
                            // index5D로 바꾸기
                            weight_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * colsize + co;

                            std::cout << input->GetShape() << '\n' << weight->GetShape() << '\n';
                            //weight_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * hiddensize) + hid) * colsize + co;

                            std::cout << "input[" << input_index <<"] : " << (*input)[input_index] << "weight[" << weight_index <<"] : " << (*weight)[weight_index] << '\n';
                              (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = (*input)[input_index] * (*weight)[weight_index];
                        }

//std:: cout<< std:: fixed<< ba << ", "<< ch<< ", "<< ro<< ", "<< co<< ", "<< (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)]<< ", ";
//                        = temp;
// //std:: cout<< temp<< std:: endl;
//                         std::cout<<(*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)]<<'\n';
//                         temp = 0.f;
                    }
                }
            }
        }

    // -----------------------------
    return TRUE;
}

//=========================================================================================
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
        // else{
        //   //backpropagation
        //   for (int ba = 0; ba < batchsize; ba++) {
        //       for (int ch = 0; ch < channelsize; ch++) {
        //           for (int ro = 0; ro < rowsize; ro++) {
        //               for (int co = 0; co < colsize; co++) {
        //                       (*result)[Index5D(shapeOfResult, ti, ba, ch, ro, co)] = (*input0)[Index5D(input0->GetShape(), ti, ba, ch, ro, co)] + (*input1)[Index5D(input1->GetShape(), ti, ba, ch, ro, co)];
        //               }
        //           }
        //       }
        //   }
        // }
        return TRUE;
    }
    //=========================================================================================
};

#endif  // ifndef RECURRENT_H_
