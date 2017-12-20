#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Convolution : public Operator<DTYPE>{
private:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

    // jw
    int m_Stride1;
    int m_Stride2;
    int m_Stride3;
    int m_Stride4;
    int m_Padding1;
    int m_Padding2;
    // ~jw

public:
    // pInput1 : input
    // pInput1 : weight

    Convolution(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int stride1, int stride2, int stride3, int stride4, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        m_Stride1 = stride1;
        m_Stride2 = stride2;
        m_Stride3 = stride3;
        m_Stride4 = stride4;

        Alloc(pInput0, pInput1, stride1, stride2, stride3, stride4);
    }

    // Convolution(Operator *pInput, Operator *pWeight, int stride1, int stride2, int stride3, int stride4, int padding1, int padding2, std::string pName) : Operator(pInput, pWeight, pName){
    // std::cout << "Convolution::Convolution(Operator *, Operator *, std::string)" << '\n';
    // }

    virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int stride1, int stride2, int stride3, int stride4) {
        int width  = (pInput0->GetOutput()->GetCol() - pInput1->GetOutput()->GetCol() + 1) / stride1;
        int height = (pInput0->GetOutput()->GetRow() - pInput1->GetOutput()->GetCol() + 1) / stride1;

        this->SetOutput(new Tensor<DTYPE>(pInput0->GetOutput()->GetTime(), pInput0->GetOutput()->GetBatch(), pInput1->GetOutput()->GetBatch(), height, width));
        this->SetDelta(new Tensor<DTYPE>(pInput0->GetOutput()->GetTime(), pInput0->GetOutput()->GetBatch(), pInput1->GetOutput()->GetBatch(), height, width));

        return 1;
    }

    virtual ~Convolution() {
        std::cout << "Convolution::~Convolution()" << '\n';
    }

    virtual int ComputeForwardPropagate() {
        std::cout << this->GetName() << " : ComputeForwardPropagate()" << '\n';

        TENSOR_DTYPE input  = this->GetInputOperator()[0]->GetOutput()->GetData();
        TENSOR_DTYPE weight = this->GetInputOperator()[1]->GetOutput()->GetData();
        TENSOR_DTYPE output = this->GetOutput()->GetData();

        int inputTime    = this->GetInputOperator()[0]->GetOutput()->GetTime();
        int inputBatch   = this->GetInputOperator()[0]->GetOutput()->GetBatch();
        int inputChannel = this->GetInputOperator()[0]->GetOutput()->GetChannel();
        int inputRow     = this->GetInputOperator()[0]->GetOutput()->GetRow();
        int inputCol     = this->GetInputOperator()[0]->GetOutput()->GetCol();

        int maskBatch   = this->GetInputOperator()[1]->GetOutput()->GetBatch();
        int maskChannel = this->GetInputOperator()[1]->GetOutput()->GetChannel();
        int maskHeight  = this->GetInputOperator()[1]->GetOutput()->GetRow();
        int maskWidth   = this->GetInputOperator()[1]->GetOutput()->GetCol();

        float net = 0.f;

        for (int outputPlane = 0; outputPlane < maskBatch; outputPlane++) {  // p
            for (int i = 0; (i * m_Stride3 + maskHeight - 1) < inputRow; i++) {
                for (int j = 0; (j * m_Stride2 + maskWidth - 1) < inputCol; j++) {
                    for (int inTi = 0; inTi < inputTime; inTi++) {
                        for (int inBa = 0; inBa < inputBatch; inBa++) {
                            for (int inCh = 0; inCh < inputChannel; inCh++) {  // q
                                net += ComputeConvolution(input[inTi][inBa][inCh], weight, i, j, maskWidth, maskHeight, m_Stride2, m_Stride3);
                            }
                            output[inTi][inBa][outputPlane][i][j] = net;
                            net                                   = 0;
                        }
                    }

                    // CHECK LIST
                    // is output assigned?
                }
            }
        }


        return true;
    }

    DTYPE ComputeConvolution(DTYPE **x, DTYPE *****weight, int i, int j, int mw, int mh, int sx, int sy) {
        DTYPE output  = 0.f;
        int   time    = this->GetInputOperator()[1]->GetOutput()->GetTime();
        int   batch   = this->GetInputOperator()[1]->GetOutput()->GetBatch();
        int   channel = this->GetInputOperator()[1]->GetOutput()->GetChannel();

        for (int ti = 0; ti < time; ti++) {
            for (int ba = 0; ba < batch; ba++) {
                for (int ch = 0; ch < channel; ch++) {
                    // convolution computation here
                    DTYPE **w = weight[ti][ba][ch];

                    for (int u = 0; u < mh; u++) {
                        for (int v = 0; v < mw; v++) {
                            output += x[i * sy + u][j * sx + v] * w[u][v];
                        }
                    }
                }
            }
        }

        return output;
    }

    virtual int ComputeBackPropagate() {
        std::cout << this->GetName() << " : ComputeBackPropagate()" << '\n';

        return true;
    }
};

#endif  // CONVOLUTION_H_
