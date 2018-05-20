#ifndef ADD_H_
#define ADD_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Addall : public Operator<DTYPE>{
private:
    Shape *m_pLeftTenShape;
    Shape *m_pRightTenShape;

    int m_timesize;
    int m_batchsize;
    int m_channelsize;
    int m_rowsize;
    int m_colsize;

public:
    Addall(Operator<DTYPE> *pLeftInput, Operator<DTYPE> *pRightInput, std::string pName) : Operator<DTYPE>(pLeftInput, pRightInput, pName) {
        #if __DEBUG__
        std::cout << "Addall::Addall(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pLeftInput, pRightInput);
    }

    ~Addall() {
        #if __DEBUG__
        std::cout << "Addall::~Addall()" << '\n';
        #endif  // __DEBUG__
    }

    int Alloc(Operator<DTYPE> *pLeftInput, Operator<DTYPE> *pRightInput) {
        #if __DEBUG__
        std::cout << "Addall::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        m_pLeftTenShape  = pLeftInput->GetResult()->GetShape();
        m_pRightTenShape = pRightInput->GetResult()->GetShape();

        m_timesize    = (*m_pLeftTenShape)[0];
        m_batchsize   = (*m_pLeftTenShape)[1];
        m_channelsize = (*m_pLeftTenShape)[2];
        m_rowsize     = (*m_pLeftTenShape)[3];
        m_colsize     = (*m_pLeftTenShape)[4];

        this->SetResult(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        this->SetGradient(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        return TRUE;
    }

    int ForwardPropagate(int pThreadNum = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left   = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *right  = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int numOfThread = this->GetNumOfThread();

        for (int m_ba = pThreadNum; m_ba < m_batchsize; m_ba += numOfThread) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*result)[Index4D(m_pLeftTenShape, m_ba, m_ch, m_ro, m_co)]
                            = (*left)[Index4D(m_pLeftTenShape, m_ba, m_ch, m_ro, m_co)]
                              + (*right)[Index4D(m_pRightTenShape, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate(int pThreadNum = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left_grad  = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *right_grad = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        int numOfThread = this->GetNumOfThread();

        for (int m_ba = pThreadNum; m_ba < m_batchsize; m_ba += numOfThread) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*left_grad)[Index4D(m_pLeftTenShape, m_ba, m_ch, m_ro, m_co)]
                            += (*this_grad)[Index4D(m_pLeftTenShape, m_ba, m_ch, m_ro, m_co)];

                        (*right_grad)[Index4D(m_pRightTenShape, m_ba, m_ch, m_ro, m_co)]
                            += (*this_grad)[Index4D(m_pLeftTenShape, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }

        return TRUE;
    }

#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime) {
        this->ForwardPropagate();
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime) {
        this->BackPropagate();

        return TRUE;
    }

#endif  // __CUDNN__
};


template<typename DTYPE> class AddColWise : public Operator<DTYPE>{
private:
    Shape *m_pInputTenShape;
    Shape *m_pBiasTenShape;

    int m_timesize;
    int m_batchsize;
    int m_channelsize;
    int m_rowsize;
    int m_colsize;

public:
    AddColWise(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName) : Operator<DTYPE>(pInput, pBias, pName) {
        #if __DEBUG__
        std::cout << "AddColWise::AddColWise(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pBias);
    }

    ~AddColWise() {
        #if __DEBUG__
        std::cout << "AddColWise::~AddColWise()" << '\n';
        #endif  // __DEBUG__
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias) {
        #if __DEBUG__
        std::cout << "AddColWise::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        m_pInputTenShape = pInput->GetResult()->GetShape();
        m_pBiasTenShape  = pBias->GetResult()->GetShape();

        m_timesize    = (*m_pInputTenShape)[0];
        m_batchsize   = (*m_pInputTenShape)[1];
        m_channelsize = (*m_pInputTenShape)[2];
        m_rowsize     = (*m_pInputTenShape)[3];
        m_colsize     = (*m_pInputTenShape)[4];

        #if __DEBUG__

        if ((*m_pBiasTenShape)[0] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[1] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[2] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[3] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);
        #endif  // __DEBUG__

        this->SetResult(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        this->SetGradient(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        return TRUE;
    }

    int ForwardPropagate(int pThreadNum = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *bias   = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int numOfThread = this->GetNumOfThread();

        for (int m_ba = pThreadNum; m_ba < m_batchsize; m_ba += numOfThread) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*result)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)]
                            = (*input)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)]
                              + (*bias)[m_co];
                    }
                }
            }
        }


        return TRUE;
    }

    int BackPropagate(int pThreadNum = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input_grad = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *bias_grad  = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        int numOfThread = this->GetNumOfThread();

        for (int m_ba = pThreadNum; m_ba < m_batchsize; m_ba += numOfThread) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*input_grad)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)]
                            += (*this_grad)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)];

                        (*bias_grad)[m_co]
                            += (*this_grad)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }


        return TRUE;
    }

#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime) {
        this->ForwardPropagate();
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime) {
        this->BackPropagate();

        return TRUE;
    }

#endif  // __CUDNN__
};

template<typename DTYPE> class AddChannelWise : public Operator<DTYPE>{
private:
    Shape *m_pInputTenShape;
    Shape *m_pBiasTenShape;

    int m_timesize;
    int m_batchsize;
    int m_channelsize;
    int m_rowsize;
    int m_colsize;

public:
    AddChannelWise(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName) : Operator<DTYPE>(pInput, pBias, pName) {
        #if __DEBUG__
        std::cout << "AddChannelWise::AddChannelWise(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pBias);
    }

    ~AddChannelWise() {
        #if __DEBUG__
        std::cout << "AddChannelWise::~AddChannelWise()" << '\n';
        #endif  // __DEBUG__
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias) {
        #if __DEBUG__
        std::cout << "AddColWise::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        m_pInputTenShape = pInput->GetResult()->GetShape();
        m_pBiasTenShape  = pBias->GetResult()->GetShape();

        m_timesize    = (*m_pInputTenShape)[0];
        m_batchsize   = (*m_pInputTenShape)[1];
        m_channelsize = (*m_pInputTenShape)[2];
        m_rowsize     = (*m_pInputTenShape)[3];
        m_colsize     = (*m_pInputTenShape)[4];

        #if __DEBUG__

        if ((*m_pBiasTenShape)[0] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[1] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[3] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);

        if ((*m_pBiasTenShape)[4] != 1) printf("Receive invalid bias shape in %s (%s %d), cannot handling\n", __FUNCTION__, __FILE__, __LINE__);
        #endif  // __DEBUG__

        this->SetResult(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        this->SetGradient(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        return TRUE;
    }

    int ForwardPropagate(int pThreadNum = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *bias   = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int numOfThread = this->GetNumOfThread();

        for (int m_ba = pThreadNum; m_ba < m_batchsize; m_ba += numOfThread) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*result)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)]
                            = (*input)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)]
                              + (*bias)[m_ch];
                    }
                }
            }
        }


        return TRUE;
    }

    int BackPropagate(int pThreadNum = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input_grad = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *bias_grad  = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        int numOfThread = this->GetNumOfThread();

        for (int m_ba = pThreadNum; m_ba < m_batchsize; m_ba += numOfThread) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*input_grad)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)]
                            += (*this_grad)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)];

                        (*bias_grad)[m_ch]
                            += (*this_grad)[Index4D(m_pInputTenShape, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }

        return TRUE;
    }

#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime) {
        this->ForwardPropagate();
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime) {
        this->BackPropagate();

        return TRUE;
    }

#endif  // __CUDNN__
};


#endif  // ADD_H_
