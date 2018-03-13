#ifndef __LINEAR_LAYER__
#define __LINEAR_LAYER__    value

#include "../Layer.h"

namespace _Layer {
template<typename DTYPE> class Linear : public Layer<DTYPE>{
private:
    Tensorholder<DTYPE> *m_pWeight;
    Tensorholder<DTYPE> *m_pBias;

    Operator<DTYPE> *m_pMatMul;
    Operator<DTYPE> *m_pAdd;
    Operator<DTYPE> *m_pActivation;

public:
    Linear(Operator<float> *pInput, int pNumInputCol, int pNumOutputCol, std::string activation = "No", int use_bias = FALSE, std::string pName = NULL) {
        Alloc(pInput, pNumInputCol, pNumOutputCol, activation, use_bias, pName);
    }

    virtual ~Linear() {}

    int Alloc(Operator<float> *pInput, int pNumInputCol, int pNumOutputCol, std::string activation, int use_bias, std::string pName) {
        Operator<float> *out = pInput;

        Tensorholder<DTYPE> *pWeight = this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Truncated_normal(1, 1, 1, pNumInputCol, pNumOutputCol, 0.0, 0.1), "Linear_Weight_" + pName));
        out = this->AddOperator(new MatMul<DTYPE>(out, pWeight, "Linear_MatMul_" + pName));

        if (use_bias) {
            Tensorholder<DTYPE> *pBias = this->AddParameter(new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumOutputCol, 0.1), "Linear_Bias_" + pName));
            out = this->AddOperator(new Add<DTYPE>(out, pBias, "Linear_Add_" + pName));
        }

        if ((activation == "Relu") || (activation == "relu")) out = this->AddOperator(new Relu<DTYPE>(out, "Linear_Relu_" + pName));
        else if ((activation == "Sigmoid") || (activation == "sigmoid")) out = this->AddOperator(new Sigmoid<DTYPE>(out, "Linear_Sigmoid_" + pName));

        return TRUE;
    }
};
}


#endif  // __LINEAR_LAYER__
