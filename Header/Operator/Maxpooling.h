#ifndef MAXPOOLING_H_
#define MAXPOOLING_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Maxpooling : public Operator<DTYPE>{
private:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

    int m_StrideRow = 0;
    int m_StrideCol = 0;

public:
    /*
     * Maxpooling(Operator<DTYPE> *pInput, MetaParameter<DTYPE> *pParam) : Operator<DTYPE>(pInput, pParam) {
     * std::cout << "Maxpooling::Maxpooling(Operator<DTYPE> *, MetaParameter<DTYPE> *)" << '\n';
     * this->Alloc(pInput, pParam);
     * }
     *
     * Maxpooling(Operator<DTYPE> *pInput, MetaParameter<DTYPE> *pParam, std::string pName) : Operator<DTYPE>(pInput, pParam, pName) {
     * std::cout << "Maxpooling::Maxpooling(Operator<DTYPE> *, MetaParameter<DTYPE> *, std::string)" << '\n';
     * this->Alloc(pInput, pParam);
     * }*/
    Maxpooling(Operator<DTYPE> *pInput, int strideRow, int strideCol) : Operator<DTYPE>(pInput) {
        std::cout << "Maxpooling::Maxpooling(Operator<DTYPE> *, int, int)" << '\n';
        this->Alloc(pInput, strideRow, strideCol);

        m_StrideRow = strideRow;
        m_StrideCol = strideCol;
    }

    Maxpooling(Operator<DTYPE> *pInput, int strideRow, int strideCol, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Maxpooling::Maxpooling(Operator<DTYPE> *, int, int, std::string)" << '\n';
        this->Alloc(pInput, strideRow, strideCol);

        m_StrideRow = strideRow;
        m_StrideCol = strideCol;
    }

    //
    ~Maxpooling() {
        std::cout << "Maxpooling::~Maxpooling()" << '\n';
    }

    //
    virtual int Alloc(Operator<DTYPE> *pInput, int strideRow, int strideCol) {
        std::cout << "Maxpooling::Alloc(Operator<DTYPE> *, int, int)" << '\n';

        int Time    = pInput->GetOutput()->GetTime();
        int Batch   = pInput->GetOutput()->GetBatch();
        int Channel = pInput->GetOutput()->GetChannel();
        int Row     = pInput->GetOutput()->GetRow();
        int Col     = pInput->GetOutput()->GetCol();

        Row = (Row + strideRow - 1) / strideRow;
        Col = (Col + strideCol - 1) / strideCol;

        // 결과물 shape
        this->SetOutput(new Tensor<DTYPE>(Time, Batch, Channel, Row, Col));
        // Gradient는 Trainable한 요소에서만 필요하다.
        this->SetDelta(new Tensor<DTYPE>(Time, Batch, Channel, Row, Col));

        return 1;
    }

    //
    virtual int ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int Time    = this->GetOutput()->GetTime();
        int Batch   = this->GetOutput()->GetBatch();
        int Channel = this->GetOutput()->GetChannel();

        int RowInput = this->GetInputOperator()[0]->GetOutput()->GetRow();
        int ColInput = this->GetInputOperator()[0]->GetOutput()->GetCol();

        int RowOutput = this->GetOutput()->GetRow();
        int ColOutput = this->GetOutput()->GetCol();

        TENSOR_DTYPE input  = this->GetInputOperator()[0]->GetOutput()->GetData();
        TENSOR_DTYPE output = this->GetOutput()->GetData();

        Tensor<int> valid(Time, Batch, Channel, RowOutput, ColOutput);
        int *****valid_output = valid.GetData();

        // int valid_output[Time][Batch][Channel][RowOutput][ColOutput] = { 0 };

        for (int ti = 0; ti < Time; ti++)
            for (int ba = 0; ba < Batch; ba++)
                for (int ch = 0; ch < Channel; ch++)
                    for (int ro = 0; ro < RowInput; ro++) {
                        int ro2 = ro / m_StrideRow;

                        for (int co = 0; co < ColInput; co++)
                            if (!valid_output[ti][ba][ch][ro2][co / m_StrideCol]
                                || (input[ti][ba][ch][ro][co] > output[ti][ba][ch][ro2][co / m_StrideCol])) {
                                valid_output[ti][ba][ch][ro2][co / m_StrideCol] = 1;

                                output[ti][ba][ch][ro2][co / m_StrideCol] = input[ti][ba][ch][ro][co];
                            }
                    }
        return 1;

        /*
         * DTYPE max = 0;
         *
         * int strideRow = 0;
         * int strideCol = 0;
         *
         * for (int ti = 0; ti < Time; ti++)
         *  for (int ba = 0; ba < Batch; ba++)
         *      for (int ch = 0; ch < Channel; ch++)
         *          for (int ro = 0; ro < Row; ro++)
         *          {
         *              if (RowInput < (ro + 1) * m_StrideRow)
         *                  strideRow = RowInput - (ro * m_StrideRow);
         *              else
         *                  strideRow = m_StrideRow;
         *
         *              for (int co = 0; co < Col; co++)
         *              {
         *                  if (ColInput < (co + 1) * m_StrideCol)
         *                      strideCol = ColInput - (co * m_StrideCol);
         *                  else
         *                      strideCol = m_StrideRow;
         *
         *                  max = input[ti][ba][ch][ro * m_StrideRow][co * m_StrideCol];
         *
         *                  for (int u = ro * m_StrideRow; u < (ro + 1) * strideRow; u++)
         *                      for (int v = co * m_StrideCol; v < (co + 1) * strideCol; v++)
         *                          if (max < input[ti][ba][ch][u][v])
         *                              max = input[ti][ba][ch][u][v];
         *
         *                  output[ti][ba][ch][ro][co] = max;
         *              }
         *          }
         * return 1;
         */
    }

    //
    virtual int ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int Time    = this->GetOutput()->GetTime();
        int Batch   = this->GetOutput()->GetBatch();
        int Channel = this->GetOutput()->GetChannel();

        int RowInput = this->GetInputOperator()[0]->GetOutput()->GetRow();
        int ColInput = this->GetInputOperator()[0]->GetOutput()->GetCol();

        int RowOutput = this->GetOutput()->GetRow();
        int ColOutput = this->GetOutput()->GetCol();

        TENSOR_DTYPE input  = this->GetInputOperator()[0]->GetOutput()->GetData();
        TENSOR_DTYPE output = this->GetOutput()->GetData();

        TENSOR_DTYPE delta_input = this->GetInputOperator()[0]->GetDelta()->GetData();
        TENSOR_DTYPE delta       = this->GetDelta()->GetData();

        Tensor<int> valid(Time, Batch, Channel, RowOutput, ColOutput);
        int *****valid_delta = valid.GetData();

        this->GetInputOperator()[0]->GetDelta()->Reset();

        for (int ti = 0; ti < Time; ti++)
            for (int ba = 0; ba < Batch; ba++)
                for (int ch = 0; ch < Channel; ch++)
                    for (int ro = 0; ro < RowInput; ro++) {
                        int ro2 = ro / m_StrideRow;

                        for (int co = 0; co < ColInput; co++)
                            if (!valid_delta[ti][ba][ch][ro2][co / m_StrideCol]
                                || (input[ti][ba][ch][ro][co] == output[ti][ba][ch][ro2][co / m_StrideCol])) {
                                valid_delta[ti][ba][ch][ro2][co / m_StrideCol] = 1;

                                delta_input[ti][ba][ch][ro][co] = delta[ti][ba][ch][ro2][co / m_StrideCol];
                            }
                    }
        return 1;
    }
};
//
#endif //MAXPOOLING_H_
