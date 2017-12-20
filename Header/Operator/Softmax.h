#ifndef SOFTMAX_H_
#define SOFTMAX_H_    value

#include "..//Operator.h"

template <typename DTYPE>
class Softmax : public Operator<DTYPE> {
private:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

    DTYPE m_epsilon                  = 0.0;
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Softmax::Alloc())
    Softmax(Operator<DTYPE> *pInput, DTYPE epsilon = 1e-20) :Operator<DTYPE>(pInput) {
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        Alloc(pInput, epsilon);
    }

    Softmax(Operator<DTYPE> *pInput, std::string pName) :Operator<DTYPE>(pInput, pName) {
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        Alloc(pInput);
    }

    Softmax(Operator<DTYPE> *pInput, DTYPE epsilon, std::string pName) :Operator<DTYPE>(pInput, pName) {
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        Alloc(pInput, epsilon);
    }

    ~Softmax() {
        std::cout << "Softmax::~Softmax()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput,  DTYPE epsilon = 1e-20) {
        std::cout << "Softmax::Alloc(Operator *, Operator *)" << '\n';

        Tensor<DTYPE> *output = new Tensor<DTYPE>(pInput->GetOutput()->GetShape());
        this->SetOutput(output);
        Tensor<DTYPE> *delta = new Tensor<DTYPE>(pInput->GetOutput()->GetShape());
        this->SetDelta(delta);

        m_epsilon         = epsilon;

        return 1;
    }

    virtual int ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape          = this->GetInputOperator()[0]->GetOutput()->GetShape();
        TENSOR_DTYPE input  = this->GetInputOperator()[0]->GetOutput()->GetData();
        TENSOR_DTYPE output = this->GetOutput()->GetData();

        int Time    = shape[0];
        int Batch   = shape[1];
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        DTYPE sum[Time][Batch] = { 0.0 };
        DTYPE max[Time][Batch] = { 0.0 };

        DTYPE temp = 0.0;

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                max[ti][ba] = Max(input[ti][ba], Channel, Row, Col);

                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            temp += (exp(input[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon);
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
                            output[ti][ba][ch][ro][co] = (exp(input[ti][ba][ch][ro][co] - max[ti][ba]) + m_epsilon) / sum[ti][ba];
                        }
                    }
                }
            }
        }

        return 1;
    }

    virtual int ComputeBackPropagate() {
        // std::cout << this->GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape              = this->GetOutput()->GetShape();
        TENSOR_DTYPE output      = this->GetOutput()->GetData();
        TENSOR_DTYPE delta       = this->GetDelta()->GetData();
        TENSOR_DTYPE delta_input = this->GetInputOperator()[0]->GetDelta()->GetData();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            delta_input[ti][ba][ch][ro][co] = Softmax_derivative(
                                delta[ti][ba],
                                output[ti][ba],
                                output[ti][ba][ch][ro][co],
                                shape,
                                ch,
                                ro,
                                co
                                );
                        }
                    }
                }
            }
        }

        this->GetDelta()->Reset();

        return 1;
    }

    template<typename TEMP_DTYPE>
    DTYPE Max(TEMP_DTYPE data, int Channel, int Row, int Col) {
        // initialize
        DTYPE max = data[0][0][0];

        for (int ch = 0; ch < Channel; ch++) {
            for (int ro = 0; ro < Row; ro++) {
                for (int co = 0; co < Col; co++) {
                    if (data[ch][ro][co] > max) max = data[ch][ro][co];
                }
            }
        }

        return max;
    }

    template<typename TEMP_DTYPE>
    DTYPE Softmax_derivative(TEMP_DTYPE delta_data, TEMP_DTYPE prediction_data, DTYPE prediction_target, int *shape, int pChannel, int pRow, int pCol) {
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        DTYPE delta_ = 0.0;

        for (int ch = 0; ch < Channel; ch++) {
            for (int ro = 0; ro < Row; ro++) {
                for (int co = 0; co < Col; co++) {
                    if ((ch == pChannel) && (ro == pRow) && (co == pCol)) {
                        delta_ += delta_data[ch][ro][co] * prediction_data[ch][ro][co] * (1 - prediction_target);
                    } else delta_ += delta_data[ch][ro][co] * prediction_data[ch][ro][co] * (-prediction_target);
                }
            }
        }

        return delta_;
    }

};

#endif  // SOFTMAX_H_
