#ifndef MSE_H_
#define MSE_H_    value

#include "..//Objective.h"

template<typename DTYPE>
class MSE : public Objective<DTYPE>{
public:
    MSE(NeuralNetwork<DTYPE> *pNeuralNetwork, std::string pName) : Objective<DTYPE>(pNeuralNetwork, pName) {
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        this->Alloc(pNeuralNetwork);
    }

    ~MSE() {
        std::cout << "MSE::~MSE()" << '\n';
    }

    virtual int Alloc(NeuralNetwork<DTYPE> *pNeuralNetwork) {
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        Operator<DTYPE> *pInput = pNeuralNetwork->GetResultOperator();

        int timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        Shape *shapeOfGradient = new Shape(pInput->GetResult()->GetShape());

        this->SetGradient(new Tensor<DTYPE>(shapeOfGradient));

        return TRUE;
    }

    virtual Tensor<DTYPE>* ForwardPropagate(Operator<DTYPE> *pLabel) {
        Tensor<DTYPE> *input = this->GetTensor();
        Tensor<DTYPE> *label = pLabel->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        Tensor<DTYPE> *gradient = this->GetGradient();
        result->Reset();
        gradient->Reset();

        int timesize = input->GetTimeSize();
        int batchsize = input->GetBatchSize();
        int count = timesize * batchsize;

        int channelsize = input->GetChannelSize();
        int rowsize = input->GetRowSize();
        int colsize = input->GetColSize();
        int capacity = channelsize * rowsize * colsize;

        int index = 0;

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < capacity; j++) {
                index = i * capacity + j;
                (*result)[i] += Error((*input)[index], (*label)[index], capacity);
                (*gradient)[index] = ((*input)[index] - (*label)[index]);
            }
        }

        return result;
    }

    virtual Tensor<DTYPE>* BackPropagate() {
        Tensor<DTYPE> *gradient = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();
        input_delta->Reset();

        int capacity = input_delta->GetData()->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*input_delta)[i] += (*gradient)[i];
        }

        return NULL;
    }

    inline DTYPE Error(DTYPE pred, DTYPE ans, int numOfOutputDim) {
        return (pred - ans) * (pred - ans) / numOfOutputDim * 0.5;
    }
};

#endif  // MSE_H_
