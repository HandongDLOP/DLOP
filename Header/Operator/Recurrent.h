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
    Tensor<DTYPE> *transpose_inputweight;

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

        if(transpose_inputweight){
          delete transpose_inputweight;
          transpose_inputweight = NULL;
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

        transpose_output = Tensor<DTYPE>::Constants((*shapeOfInput)[0], (*shapeOfInput)[1], (*shapeOfInput)[2], (*shapeOfWeightInput)[4], (*shapeOfInput)[3], 0);
        this->AddGradient(transpose_output);

        transpose_inputweight = Tensor<DTYPE>::Constants((*shapeOfWeightInput)[0], (*shapeOfWeightInput)[1], (*shapeOfWeightInput)[2], (*shapeOfWeightInput)[4], (*shapeOfWeightInput)[3], 0);
        this->AddGradient(transpose_inputweight);

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
            //Linear(input, weightInput, biasInput, input_Result, ti, TRUE);
            MatMul(input, weightInput, input_temp, ti);
printf("input_temp\n");
std::cout<< input_temp << std::endl;
            BiasAdd(input_temp, biasInput, input_Result, ti);
printf("input_Result\n");
std::cout<< input_Result << std::endl;

            //Linear(pre_hidden_Result, weightHidden, biasHidden, pre_net_Result, ti, TRUE);    // WX + b
            MatMul(pre_hidden_Result, weightHidden, hidden_temp, ti);
            BiasAdd(hidden_temp, biasHidden, pre_net_Result, ti);
printf("pre_net_Result\n");
std::cout<< pre_net_Result << std::endl;

            Add(input_Result, pre_net_Result, net_Result, ti); // net_result   //Tensor + Tensor
printf("net_Result\n");
std::cout<< net_Result << std::endl;

            Tanh(net_Result, hidden_Result, ti);          // hidden_result
printf("hidden_Result\n");
std::cout<< hidden_Result << std::endl;

            if(ti < timesize - 1){
                  Copy(hidden_Result, pre_hidden_Result, ti, ti + 1);   //src ,dst
printf("pre_hidden_Result\n");
std::cout<< pre_hidden_Result << std::endl;
            }// hidden_result -> prehidden_result

            //Linear(hidden_Result, weightOutput, biasOutput, output_Result, ti, TRUE);
            MatMul(hidden_Result, weightOutput, output_temp, ti);
            BiasAdd(output_temp, biasOutput, output_Result, ti);
printf("output_Result\n");
std::cout<< output_Result << std::endl;
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

// Container<Tensor<DTYPE> *>   *gradient_container = this->GetGradientContainer();
// printf("Container gradient Size: ");
// std::cout << gradient_container->GetSize() << '\n';

        for(int ti = timesize - 1; ti >= 0; ti--){
          // ti에서 hiddenGradient

printf("output_Gradient\n");
std::cout<< output_Gradient << std::endl;

printf("weightOutput\n");
std::cout<< weightOutput << std::endl;

// printf("before\n")
// std::cout<< linear_output_Gradient << std::endl;

          MatMul(output_Gradient, weightOutput, linear_output_Gradient, ti);
printf("linear_output_Gradient\n");
std::cout<< linear_output_Gradient << std::endl;

          Add(linear_output_Gradient, next_linear_hidden_Gradient, pre_tanhDer_Gradient, ti);
printf("pre_tanhDer_Gradient\n");
std::cout<< pre_tanhDer_Gradient << std::endl;

          DerivativeTanh(net_Result, DerTanh, ti);
printf("DerTanh\n");
std::cout<< DerTanh << std::endl;

          Elementwise(pre_tanhDer_Gradient, DerTanh, hidden_Gradient, ti);   //Matrix Multiplication ?? or Tensor * scalar ??  // hidden_Gradient
printf("hidden_Gradient\n");
std::cout<< hidden_Gradient << std::endl;

          MatMul(hidden_Gradient, weightHidden, linear_hidden_Gradient, ti);
printf("linear_hidden_Gradient\n");
std::cout<< linear_hidden_Gradient << std::endl;


          if(ti > 0){
            Copy(linear_hidden_Gradient, next_linear_hidden_Gradient, ti, ti - 1);  // ti -> ti - 1
printf("next_linear_hidden_Gradient\n");
std::cout<< next_linear_hidden_Gradient << std::endl;
          }

          //w1,w2,w3,b1,b2,b3의 gradient
          // 1 ~ ti까지 돌리기
          // weight_gradient bias_gradient 구하기
          //w1
          Transpose(input, transpose_input, ti);
printf("transpose_input111111111111111111111111111111\n");
std::cout<< transpose_input << std::endl;

std::cout << "Transpose_input Shape : " << transpose_input->GetShape() << '\n';
std::cout << "hidden_gradient : " << hidden_Gradient->GetShape() << '\n';
std::cout << "weightInputGradient : " << weightInputGradient->GetShape() << '\n';

          CellMatMul(transpose_input, hidden_Gradient, weightInputGradient, ti);
printf("weightInputGradient\n");
std::cout<< weightInputGradient << std::endl;

          //MatMul(hidden_Gradient, biasInput, biasInputGradient, ti);
          //b1
          CellBiasAdd(zero_tensor, hidden_Gradient, biasInputGradient, ti);
printf("biasInputGradient\n");
std::cout<< biasInputGradient << std::endl;

          //w2
          Transpose(pre_hidden_Result, transpose_hidden, ti);
printf("transpose_hidden2222222222222222222222222222222\n");
std::cout<< transpose_hidden << std::endl;

          CellMatMul(transpose_hidden, hidden_Gradient, weightHiddenGradient, ti);         // ti - 1
printf("weightHiddenGradient\n");
std::cout<< weightHiddenGradient << std::endl;

          //b2
          CellBiasAdd(zero_tensor, hidden_Gradient, biasHiddenGradient, ti);
printf("biasHiddenGradient\n");
std::cout<< biasHiddenGradient << std::endl;


Transpose(hidden_Result, transpose_output, ti);
printf("transpose_output33333333333333333333333333333333333\n");
std::cout<< transpose_output << std::endl;

          //w3
          CellMatMul(transpose_output, output_Gradient, weightOutputGradient, ti);
printf("weightOutputGradient\n");
std::cout<< weightOutputGradient << std::endl;

          //b3
          CellBiasAdd(zero_tensor, output_Gradient, biasOutputGradient, ti);
printf("biasOutputGradient\n");
std::cout<< biasOutputGradient << std::endl;

          //x
          printf("-------------------------------------------------\n");
printf("weightInput\n");
std::cout<< weightInput << std::endl;

Transpose(weightInput, transpose_inputweight, ti);
printf("transpose_inputweight444444444444444444444444444444\n");

printf("transpose_inputweight\n");
std::cout<< transpose_inputweight << std::endl;
printf("-------------------------------------------------\n");

std::cout<< transpose_inputweight << std::endl;
          MatMul(hidden_Gradient, transpose_inputweight, input_delta, ti);
printf("input_delta\n");
std::cout<< input_delta << std::endl;
        }

        return TRUE;
    }

    int Copy(Tensor<DTYPE> *input, Tensor<DTYPE> *output, int input_ti, int output_ti){
std::cout<< "CopyTensor" << '\n';
        Shape *shapeOfInput  = input->GetShape();
        Shape *shapeOfOutput = output->GetShape();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

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

        return TRUE;
    }

    int Tanh(Tensor<DTYPE> *input, Tensor<DTYPE> *result, int ti){
std::cout<< "Tanh" << '\n';
        // result->GetBatchSize();
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
std::cout<<(*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)]<<'\n';
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

        //output_Gradient = this->GetGradient();
std::cout << "MatMul()\n" << '\n';
            int input_index  = 0;
            int weight_index = 0;
            int result_index = 0;

            DTYPE temp = 0.f;
std::cout << "Batchsize : " << batchsize << '\n';
std::cout << "channelsize : " << channelsize << '\n';
std::cout << "rowsize : " << rowsize << '\n';
std::cout << "colsize : " << colsize << '\n';
std::cout << "hiddensize : " << hiddensize << '\n';

// printf("result\n");
// std::cout<< result << std::endl;

  //std:: cout<< "ba, ch, ro, co, result, temp"<< std:: endl;
            // ------------MatMul------------
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++)   {
                        for (int co = 0; co < colsize; co++) {
                            for (int hid = 0; hid < hiddensize; hid++) {
                              // index5D로 바꾸기
                                input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * hiddensize + hid;
                                weight_index = hid * colsize + co;
  //std::cout << input->GetShape() << '\n' << weight->GetShape() << '\n';
                                //weight_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * hiddensize) + hid) * colsize + co;

  std::cout << "input[" << input_index <<"] : " << (*input)[input_index] << "\tweight[" << weight_index <<"] : " << (*weight)[weight_index] << '\n';
                                temp        += (*input)[input_index] * (*weight)[weight_index];
                            }

  //std:: cout<< std:: fixed<< ba << ", "<< ch<< ", "<< ro<< ", "<< co<< ", "<< (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)]<< ", ";
                            (*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)] = temp;
  //std:: cout<< temp<< std:: endl;
  //std::cout<<(*result)[Index5D(result->GetShape(), ti, ba, ch, ro, co)]<<'\n';
                            temp = 0.f;
                        }
                    }
                }
            }
        return TRUE;
    }

//  weight gradient의 batchsize가 1인데 x, output_gradient, hidden_gradient는 batchsize n이라서 맞지 않음.
//
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

        //output_Gradient = this->GetGradient();
std::cout << "CellMatMul()\n" << '\n';
            int input_index  = 0;
            int weight_index = 0;
            int result_index = 0;

            DTYPE temp = 0.f;
std::cout << "Batchsize : " << batchsize << '\n';
std::cout << "channelsize : " << channelsize << '\n';
std::cout << "rowsize : " << rowsize << '\n';
std::cout << "colsize : " << colsize << '\n';
std::cout << "hiddensize : " << hiddensize << '\n';


// printf("result\n");
// std::cout<< result << std::endl;

  //std:: cout<< "ba, ch, ro, co, result, temp"<< std:: endl;
            // ------------MatMul------------
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++)   {
                        for (int co = 0; co < colsize; co++) {
                            for (int hid = 0; hid < hiddensize; hid++) {
                              // index5D로 바꾸기
                                input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * hiddensize + hid;
                                weight_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * hiddensize) + hid) * colsize + co;

  std::cout << "input[" << input_index <<"] : " << (*input)[input_index] << "\tweight[" << weight_index <<"] : " << (*weight)[weight_index] << '\n';
                                temp        += (*input)[input_index] * (*weight)[weight_index];
                            }

                              (*result)[Index5D(result->GetShape(), resulttimesize - 1, resultbatchsize - 1, ch, ro, co)] += temp;
std::cout << "result[" << Index5D(result->GetShape(), resulttimesize - 1, resultbatchsize - 1, ch, ro, co) << "] : " << (*result)[Index5D(result->GetShape(), resulttimesize - 1, resultbatchsize - 1, ch, ro, co)] << '\n';
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

    // std::cout << input->GetShape() << '\n' << weight->GetShape() << '\n';
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
        return TRUE;
    }

    int BiasAdd(Tensor<DTYPE> *input, Tensor<DTYPE> *bias, Tensor<DTYPE> *result, int ti){
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();
        int input_index = 0, bias_index = 0;
// Shape *shapeOfResult = result->GetShape();

std::cout<< "BiasAdd" << '\n';
std::cout<< "BiasAdd- Batchsize : " << batchsize << '\n';
//         for (int ba = 0; ba < batchsize; ba++) {
//             for (int ch = 0; ch < channelsize; ch++) {
//                 for (int ro = 0; ro < rowsize; ro++) {
//                     for (int co = 0; co < colsize; co++) {
// input_index = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * colsize + co;
// bias_index  = co;
// //bias_index = ((((channelsize) + ch) * rowsize) + ro) * colsize + co;
//
// std::cout << "input[" << input_index <<"] : " << (*input)[input_index] << "\tbias[" << bias_index <<"] : " << (*bias)[bias_index] << '\n';
//
//                             (*result)[Index5D(shapeOfResult, ti, ba, ch, ro, co)] = (*input)[Index5D(input->GetShape(), ti, ba, ch, ro, co)] + (*bias)[Index5D(bias->GetShape(), ti, ba, ch, ro, co)];
//
//                     }
//                 }
//             }
//         }


        // Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        // Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        // Tensor<DTYPE> *bias   = (*input_contatiner)[1]->GetResult();
        // Tensor<DTYPE> *result = this->GetResult();
        ///
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
        //bias_index = ((((channelsize) + ch) * rowsize) + ro) * colsize + co;
                        (*result)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                = (*input)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                  + (*bias)[Index5D(m_pBiasTenShape, *m_ti_bias, *m_ba_bias, *m_ch_bias, *m_ro_bias, *m_co_bias)];
std::cout << "input[" << input_index <<"] : " << (*input)[input_index] << "\tbias[" << bias_index <<"] : " << (*bias)[bias_index] << '\n';
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

std::cout<< "Add" << '\n';
std::cout<< "Add- Batchsize : " << batchsize << '\n';
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
input_index0 = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * colsize + co;
input_index1 = ((((((ti * batchsize) + ba) * channelsize) + ch) * rowsize) + ro) * colsize + co;

std::cout << "input0[" << input_index0 <<"] : " << (*input0)[input_index0] << "\tinput1[" << input_index1 <<"] : " << (*input1)[input_index1] << '\n';

                            (*result)[Index5D(shapeOfResult, ti, ba, ch, ro, co)] = (*input0)[Index5D(input0->GetShape(), ti, ba, ch, ro, co)] + (*input1)[Index5D(input1->GetShape(), ti, ba, ch, ro, co)];

                    }
                }
            }
        }

        return TRUE;
    }

    int CellBiasAdd(Tensor<DTYPE> *input, Tensor<DTYPE> *bias, Tensor<DTYPE> *result, int ti){
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int input_index = 0, bias_index = 0;

        Shape *shapeOfInput  = input->GetShape();
        Shape *shapeOfBias   = bias->GetShape();
        Shape *shapeOfResult = result->GetShape();

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

        if ((*shapeOfBias)[0] == 1) m_ti_bias = &m_zero;
        if ((*shapeOfBias)[1] == 1) m_ba_bias = &m_zero;
        if ((*shapeOfBias)[2] == 1) m_ch_bias = &m_zero;
        if ((*shapeOfBias)[3] == 1) m_ro_bias = &m_zero;
        if ((*shapeOfBias)[4] == 1) m_co_bias = &m_zero;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                            (*result)[Index5D(shapeOfResult, (*m_ti_bias), (*m_ba_bias), ch, ro, co)] += ((*input)[Index5D(shapeOfInput, ti, ba, ch, ro, co)] + (*bias)[Index5D(shapeOfBias, ti, ba, ch, ro, co)]);

                    }
                }
            }
        }

        return TRUE;
    }

    //
    int Transpose(Tensor<DTYPE> *input, Tensor<DTYPE>* result, int ti){
      std::cout<< "Transpose" << '\n';
      int inputRowsize     = input->GetRowSize();
      int inputChannelsize = input->GetChannelSize();

      int batchsize   = result->GetBatchSize();
      int channelsize = result->GetChannelSize();
      int rowsize     = result->GetRowSize();
      int colsize     = result->GetColSize();

      // try catch로 바꾸기
      if(inputChannelsize == 1){
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int co = 0; co < colsize; co++) {
                    for (int ro = 0; ro < rowsize; ro++) {
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
