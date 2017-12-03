#ifndef SOFTMAXCROSSENTROPY_H_
#define SOFTMAXCROSSENTROPY_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class SoftmaxCrossEntropy : public Operator<DTYPE> {
private:
    Tensor<DTYPE> *m_aSoftmax_Result = NULL;
    DTYPE m_epsilon          = 0.0; // for backprop

public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (SoftmaxCrossEntropy::Alloc())
    SoftmaxCrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, DTYPE epsilon = 1e-20) : Operator<DTYPE>(pInput0, pInput1) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        Alloc(pInput0, pInput1, epsilon);
    }

    SoftmaxCrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        Alloc(pInput0, pInput1, 1e-20);
    }

    SoftmaxCrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, DTYPE epsilon, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int, std::string)" << '\n';
        Alloc(pInput0, pInput1, epsilon);
    }

    virtual ~SoftmaxCrossEntropy() {
        std::cout << "SoftmaxCrossEntropy::~SoftmaxCrossEntropy()" << '\n';

        delete m_aSoftmax_Result;
    }

    virtual bool Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, DTYPE epsilon = 1e-20) {
        std::cout << "SoftmaxCrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        // if pInput0 and pInput1의 shape가 다르면 abort

        int *shape     = pInput0->GetOutput()->GetShape();
        Tensor<DTYPE> *output = new Tensor<DTYPE>(shape[0], shape[1], 1, 1, 1);
        this->SetOutput(output);

        m_aSoftmax_Result = new Tensor<DTYPE>(shape);

        m_epsilon = epsilon;

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape             = Operator<DTYPE>::GetInputOperator()[0]->GetOutput()->GetShape();
        DTYPE *****input_data = Operator<DTYPE>::GetInputOperator()[0]->GetOutput()->GetData();
        DTYPE *****label_data = Operator<DTYPE>::GetInputOperator()[1]->GetOutput()->GetData();

        this->GetOutput()->Reset();
        DTYPE *****output         = Operator<DTYPE>::GetOutput()->GetData();
        DTYPE *****softmax_result = m_aSoftmax_Result->GetData();

        int Time    = shape[0];
        int Batch   = shape[1];
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        DTYPE sum[Time][Batch] = { 0.0 };
        DTYPE max[Time][Batch] = { 0.0 };
        int    num_of_output    = Channel * Row * Col;

        DTYPE temp = 0.0;

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                max[ti][ba] = Max(input_data[ti][ba], Channel, Row, Col);

                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            // std::cout << (exp(input_data[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon) << '\n';
                            temp += (exp(input_data[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon);
                        }
                    }
                }
                // 부동소수점 문제가 있는 듯 함 - 중요
                sum[ti][ba] = temp;
                temp        = 0.0;
            }
        }


        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            // std::cout << (exp(input_data[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon) << ", " << sum[ti][ba] << '\n';
                            softmax_result[ti][ba][ch][ro][co] = (exp(input_data[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon) / sum[ti][ba];
                            output[ti][ba][0][0][0]           += cross_entropy(label_data[ti][ba][ch][ro][co],
                                                                               softmax_result[ti][ba][ch][ro][co],
                                                                               num_of_output);
                        }
                    }
                }
            }
        }


        return true;
    }

    virtual bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape = Operator<DTYPE>::GetInputOperator()[0]->GetOutput()->GetShape();

        DTYPE *****label_data     = Operator<DTYPE>::GetInputOperator()[1]->GetOutput()->GetData();
        DTYPE *****softmax_result = m_aSoftmax_Result->GetData();

        this->GetInputOperator()[0]->GetDelta()->Reset();
        DTYPE *****delta_input_data = Operator<DTYPE>::GetInputOperator()[0]->GetDelta()->GetData();

        int Time    = shape[0];
        int Batch   = shape[1];
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        int num_of_output = Channel * Row * Col;

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            delta_input_data[ti][ba][ch][ro][co] = SoftmaxCrossEntropy_derivative(label_data[ti][ba],
                                                                                                    softmax_result[ti][ba][ch][ro][co],
                                                                                                    shape,
                                                                                                    ch,
                                                                                                    ro,
                                                                                                    co,
                                                                                                    num_of_output);
                        }
                    }
                }
            }
        }

        return true;
    }

    DTYPE Max(DTYPE ***data, int Channel, int Row, int Col) {
        // initialize
        DTYPE max = data[0][0][0];

        for (int ch = 0; ch < Channel; ch++) {
            for (int ro = 0; ro < Row; ro++) {
                for (int co = 0; co < Col; co++) {
                    if (data[ch][ro][co] > max) max = data[ch][ro][co];
                }
            }
        }

        // std::cout << max << '\n';

        return max;
    }

    DTYPE cross_entropy(DTYPE label, DTYPE prediction, int num_of_output) {
        DTYPE error_ = -label *log(prediction + m_epsilon) / num_of_output;

        return error_;
    }

    DTYPE SoftmaxCrossEntropy_derivative(DTYPE ***label_data, DTYPE prediction, int *shape, int pChannel, int pRow, int pCol, int num_of_output) {
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        DTYPE delta_ = 0.0;

        for (int ch = 0; ch < Channel; ch++) {
            for (int ro = 0; ro < Row; ro++) {
                for (int co = 0; co < Col; co++) {
                    if ((ch == pChannel) && (ro == pRow) && (co == pCol)) {
                        delta_ += -label_data[ch][ro][co] * (1 - prediction);
                    } else delta_ += -label_data[ch][ro][co] * (-prediction);
                }
            }
        }

        return delta_ / num_of_output;
    }

};

#endif  // SOFTMAXCROSSENTROPY_H_
