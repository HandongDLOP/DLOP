#ifndef BATCHNORMALIZE_H_
#define BATCHNORMALIZE_H_    value

#include "..//Operator.h"

#include <cmath>
#include <cfloat>

template< typename DTYPE>

class BatchNormalize: public Operator< DTYPE>{

private:

    Tensor< DTYPE>* m_aMean;
    Tensor< DTYPE>* m_aDMean;
    Tensor< DTYPE>* m_aVariance;
    Tensor< DTYPE>* m_aDVariance;
    Tensor< DTYPE>* m_aNormalized;
    Tensor< DTYPE>* m_aDNormalized;

    Tensor< DTYPE>* m_aFixedMean;
    Tensor< DTYPE>* m_aFixedVariance;

    int m_mti;
    int m_isConv;
    int m_isFixed;

    int m_testTimeSize;
    int m_tti;

public:

    BatchNormalize( Operator< DTYPE>* pInput, Operator< DTYPE>* pScale, Operator< DTYPE>* pShift, int pMeanTimeSize, int pTestTimeSize, int pIsConv, std:: string pName): Operator< DTYPE>( pName)
    {
        std:: cout<< "BatchNormalize:: BatchNormalize( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int, int, std:: string)"<< '\n';

	m_mti= 0;
	m_isConv= pIsConv;
	m_isFixed= FALSE;

	m_testTimeSize= pTestTimeSize;
	m_tti= 0;
	
        this-> Alloc( pInput, pScale, pShift, pMeanTimeSize);

	m_aMean-> Reset( );
	m_aDMean-> Reset( );
	m_aVariance-> Reset( );
	m_aDVariance-> Reset( );
	m_aNormalized-> Reset( );
	m_aDNormalized-> Reset( );

	m_aFixedMean-> Reset( );
	m_aFixedVariance-> Reset( );
    }

    ~ BatchNormalize( )
    {
        std:: cout<< "BatchNormalize:: ~ BatchNormalize()"<< '\n';
	
	delete m_aMean;
	delete m_aDMean;
	delete m_aVariance;
	delete m_aDVariance;
	delete m_aNormalized;
	delete m_aDNormalized;

	delete m_aFixedMean;
	delete m_aFixedVariance;
    }

    int Alloc( Operator< DTYPE>* pInput, Operator< DTYPE>* pScale, Operator< DTYPE>* pShift, int pMeanTimeSize)
    {
        std:: cout << "BatchNormalize:: Alloc( Operator< DTYPE>*, Operator< DTYPE>*, Operator< DTYPE>*, int)"<< '\n';

        Shape* inputShape= pInput-> GetResult( )-> GetShape( );
        int channelSize= ( * inputShape)[ 2];
        int rowSize= ( * inputShape)[ 3];
        int colSize= ( * inputShape)[ 4];
	int meanTimeSize= 0;
	int meanRowSize= 0;
	int meanColSize= 0;

	Operator< DTYPE>::Alloc( 3, pInput, pScale, pShift);

        this-> SetResult( new Tensor< DTYPE>( new Shape( inputShape)));
        this-> SetDelta( new Tensor< DTYPE>( new Shape( inputShape)));

	if( pMeanTimeSize> 0)
	    meanTimeSize= pMeanTimeSize;
	else
	    meanTimeSize= 1;

	if( m_isConv)
	{
	    meanRowSize= 1;
	    meanColSize= 1;
	}
	else
	{
	    meanRowSize= rowSize;
	    meanColSize= colSize;
	}
	m_aMean= new Tensor< DTYPE>( meanTimeSize, 1, channelSize, meanRowSize, meanColSize);
	m_aDMean= new Tensor< DTYPE>( 1, 1, channelSize, meanRowSize, meanColSize);
	m_aVariance= new Tensor< DTYPE>( meanTimeSize, 1, channelSize, meanRowSize, meanColSize);
	m_aDVariance= new Tensor< DTYPE>( 1, 1, channelSize, meanRowSize, meanColSize);
	m_aNormalized= new Tensor< DTYPE>( new Shape( inputShape));
	m_aDNormalized= new Tensor< DTYPE>( new Shape( inputShape));

	m_aFixedMean= new Tensor< DTYPE>( 1, 1, channelSize, meanRowSize, meanColSize);
	m_aFixedVariance= new Tensor< DTYPE>( 1, 1, channelSize, meanRowSize, meanColSize);

        return TRUE;
    }

    void Unfix( )
    {
	m_aMean-> Reset( );
	m_aVariance-> Reset( );

	m_isFixed= FALSE;
    }

    void Fix( )
    {
        Tensor< DTYPE>* input= this-> GetInput( )[ 0]-> GetResult( );

	int batchSize= input-> GetBatchSize( );
        int rowSize= input-> GetRowSize( );
        int colSize= input-> GetColSize( );

	Shape* meanShape= m_aMean-> GetShape( );
	int meanTimeSize= ( * meanShape)[ 0];
        int meanChannelSize= ( * meanShape)[ 2];
        int meanRowSize= ( * meanShape)[ 3];
        int meanColSize= ( * meanShape)[ 4];

	int effMeanTimeSize= 0;
	int effBatchSize= 0;
	int unbiasEstimator= 0;

	int meanIndex= 0;
	int fixedMeanIndex= 0;

	if( ! m_mti)
	    return;
	else if( m_mti< meanTimeSize)
	    effMeanTimeSize= m_mti;
	else
	    effMeanTimeSize= meanTimeSize;

	m_aFixedMean-> Reset( );
	m_aFixedVariance-> Reset( );

	for( int mti= 0; mti< effMeanTimeSize; mti++)
	    for( int mch= 0; mch< meanChannelSize; mch++)
		for( int mro= 0; mro< meanRowSize; mro++)
		    for( int mco= 0; mco< meanColSize; mco++)
		    {
	                meanIndex= Index5D( meanShape, mti, 0, mch, mro, mco);
	    	        fixedMeanIndex= Index4D( meanShape, 0, mch, mro, mco);

		        ( * m_aFixedMean)[ fixedMeanIndex]+= ( * m_aMean)[ meanIndex];
		        ( * m_aFixedVariance)[ fixedMeanIndex]+= ( * m_aVariance)[ meanIndex];
		    }
	if( m_isConv)
	    effBatchSize= batchSize* rowSize* colSize;
	else
	    effBatchSize= batchSize;
	unbiasEstimator= effMeanTimeSize* ( effBatchSize- 1)/ effBatchSize;
	if( ! unbiasEstimator)
	    unbiasEstimator= effMeanTimeSize;

	for( int mch= 0; mch< meanChannelSize; mch++)
	    for( int mro= 0; mro< meanRowSize; mro++)
	        for( int mco= 0; mco< meanColSize; mco++)
		{
	            fixedMeanIndex= Index4D( meanShape, 0, mch, mro, mco);

	            ( * m_aFixedMean)[ fixedMeanIndex]/= effMeanTimeSize;
	            ( * m_aFixedVariance)[ fixedMeanIndex]/= unbiasEstimator;
		}
	m_isFixed= TRUE;
	m_tti= 0;
    }

    int ComputeForwardPropagate( )
    {
        Tensor< DTYPE>* input= this-> GetInput( )[ 0]-> GetResult( );
        Tensor< DTYPE>* scale= this-> GetInput( )[ 1]-> GetResult( );
        Tensor< DTYPE>* shift= this-> GetInput( )[ 2]-> GetResult( );

        Tensor< DTYPE>* result= this-> GetResult( );
	Tensor< DTYPE>* mean= NULL;
	Tensor< DTYPE>* variance= NULL;

        Shape* inputShape= input-> GetShape( );
        int batchSize= ( * inputShape)[ 1];
        int channelSize= ( * inputShape)[ 2];
        int rowSize= ( * inputShape)[ 3];
        int colSize= ( * inputShape)[ 4];

	Shape* meanShape= m_aMean-> GetShape( );
	int meanTimeSize= ( * meanShape)[ 0];
        int meanRowSize= ( * meanShape)[ 3];
        int meanColSize= ( * meanShape)[ 4];

	int effBatchSize= 0;

	int mti= m_mti% meanTimeSize;
	int mro= 0;
	int mco= 0;

	int index= 0;
	int meanIndex= 0;
	int fixedMeanIndex= 0;

	float value= 0;
	float meanValue= 0;
	float normalValue= 0;

	if( m_isConv)
	    effBatchSize= batchSize* rowSize* colSize;
	else
	    effBatchSize= batchSize;

	if( m_isFixed)
	{
	    m_tti++;
	    if( m_tti== m_testTimeSize)
		this-> Unfix( );

	    mean= m_aFixedMean;
	    variance= m_aFixedVariance;
	}
	else
	{
	    mro= 0;
	    mco= 0;

	    for( int ba= 0; ba< batchSize; ba++)
		for( int ch= 0; ch< channelSize; ch++)
		    for( int ro= 0; ro< rowSize; ro++)
			for( int co= 0; co< colSize; co++)
		        {
			    value= ( * input)[ Index4D(inputShape, ba, ch, ro, co)];

			    if( ! m_isConv)
			    {
				mro= ro;
				mco= co;
			    }
			    meanIndex= Index5D(meanShape, mti, 0, ch, mro, mco);

			    ( * m_aMean)[ meanIndex]+= value;
			    ( * m_aVariance)[ meanIndex]+= value* value; 
			}
	    for( int ch= 0; ch< channelSize; ch++)
	        for( mro= 0; mro< meanRowSize; mro++)
	            for( mco= 0; mco< meanColSize; mco++)
		    {
	                meanIndex= Index5D(meanShape, mti, 0, ch, mro, mco);
			meanValue= ( * m_aMean)[ meanIndex]/ effBatchSize;

			( * m_aMean)[ meanIndex]= meanValue;
	                ( * m_aVariance)[ meanIndex]= ( ( * m_aVariance)[ meanIndex]/ effBatchSize)- meanValue* meanValue;
		    }
	    if( mti+ 1== meanTimeSize)
	        this-> Fix( );

	    mean= m_aMean;
	    variance= m_aVariance;
        }
	mro= 0;
	mco= 0;

	for( int ba= 0; ba< batchSize; ba++)
	    for( int ch= 0; ch< channelSize; ch++)
		for( int ro= 0; ro< rowSize; ro++)
		    for( int co= 0; co< colSize; co++)
		    {
		        index= Index4D( inputShape, ba, ch, ro, co);

			if( ! m_isConv)
			{
			    mro= ro;
			    mco= co;
		 	}
			fixedMeanIndex= Index4D( meanShape, 0, ch, mro, mco);
			if( m_isFixed)
			    meanIndex= fixedMeanIndex;
			else
			    meanIndex= Index5D( meanShape, mti, 0, ch, mro, mco);
			    
			normalValue= ( ( * input)[ index]- ( * mean)[ meanIndex])/ std:: sqrt( ( * variance)[ meanIndex]+ FLT_EPSILON); 

			( * m_aNormalized)[ index]= normalValue;
			( * result)[ index]= ( ( * scale)[ fixedMeanIndex]* normalValue)+ ( * shift)[ fixedMeanIndex]; 
		    }
        return TRUE;
    }

    int ComputeBackPropagate( )
    {
	Tensor< DTYPE>* dResult= this-> GetDelta( );

	Tensor< DTYPE>* dInput= this-> GetInput( )[ 0]-> GetDelta( );
	Tensor< DTYPE>* dScale= this-> GetInput( )[ 1]-> GetGradient( );
	Tensor< DTYPE>* dShift= this-> GetInput( )[ 2]-> GetGradient( );

        Tensor< DTYPE>* input= this-> GetInput( )[ 0]-> GetResult( );
        Tensor< DTYPE>* scale= this-> GetInput( )[ 1]-> GetResult( );

        Shape* inputShape= input-> GetShape( );
        int batchSize= ( * inputShape)[ 1];
        int channelSize= ( * inputShape)[ 2];
        int rowSize= ( * inputShape)[ 3];
        int colSize= ( * inputShape)[ 4];

	Shape* meanShape= m_aMean-> GetShape( );
	int meanTimeSize= ( * meanShape)[ 0];

	int effBatchSize= 0;

	int mti= m_mti% meanTimeSize;
	int mro= 0;
	int mco= 0;

	int index= 0;
	int meanIndex= 0;
	int fixedMeanIndex= 0;

	float dValue= 0;
	float dNormalValue= 0;
	float varianceValue= 0;
	float stddevValue= 0;
	float dInputValue= 0;

	dInput-> Reset( );
	dScale-> Reset( );
	dShift-> Reset( );
	m_aDMean-> Reset( );
	m_aDVariance-> Reset( );

	mro= 0;
	mco= 0;

	for( int ba= 0; ba< batchSize; ba++)
	    for( int ch= 0; ch< channelSize; ch++)
		for( int ro= 0; ro< rowSize; ro++)
		    for( int co= 0; co< colSize; co++)
		    {
			index= Index4D( inputShape, ba, ch, ro, co);
			dValue= ( * dResult)[ index];

			if( ! m_isConv)
			{
			    mro= ro;
			    mco= co;
		 	}
			meanIndex= Index5D( meanShape, mti, 0, ch, mro, mco);
			fixedMeanIndex= Index4D( meanShape, 0, ch, mro, mco);

			( * dShift)[ fixedMeanIndex]+= dValue;
			( * dScale)[ fixedMeanIndex]+= dValue* ( * m_aNormalized)[ index];

			( * m_aDNormalized)[ index]= dValue* ( * scale)[ fixedMeanIndex];
		    }
	mro= 0;
	mco= 0;

	for( int ba= 0; ba< batchSize; ba++)
	    for( int ch= 0; ch< channelSize; ch++)
		for( int ro= 0; ro< rowSize; ro++)
		    for( int co= 0; co< colSize; co++)
		    {
			index= Index4D( inputShape, ba, ch, ro, co);
			dNormalValue= ( * m_aDNormalized)[ index];

			if( ! m_isConv)
			{
			    mro= ro;
			    mco= co;
		 	}
			meanIndex= Index5D( meanShape, mti, 0, ch, mro, mco);
			fixedMeanIndex= Index4D( meanShape, 0, ch, mro, mco);

			varianceValue= ( * m_aVariance)[ meanIndex]+ FLT_EPSILON;
			stddevValue= std:: sqrt( varianceValue);
			dInputValue= dNormalValue/ stddevValue;

			( * dInput)[ index]= dInputValue;

			( * m_aDMean)[ fixedMeanIndex]-= dInputValue;
			( * m_aDVariance)[ fixedMeanIndex]+= dNormalValue* ( ( * input)[ index]- ( * m_aMean)[ meanIndex])* ( - .5f)/ ( varianceValue* stddevValue);
		    }
	mro= 0;
	mco= 0;

	if( m_isConv)
	    effBatchSize= batchSize* rowSize* colSize;
	else
	    effBatchSize= batchSize;

	for( int ba= 0; ba< batchSize; ba++)
	    for( int ch= 0; ch< channelSize; ch++)
		for( int ro= 0; ro< rowSize; ro++)
		    for( int co= 0; co< colSize; co++)
		    {
			index= Index4D( inputShape, ba, ch, ro, co);

			if( ! m_isConv)
			{
			    mro= ro;
			    mco= co;
		 	}
			meanIndex= Index5D( meanShape, mti, 0, ch, mro, mco);
			fixedMeanIndex= Index4D( meanShape, 0, ch, mro, mco);

			( * dInput)[ index]+= ( ( * m_aDMean)[ fixedMeanIndex]+ ( * m_aDVariance)[ fixedMeanIndex]* 2* ( ( * input)[ index]- ( * m_aMean)[ meanIndex]))/ effBatchSize;
		    }
	m_mti++;

        return TRUE;
    }
};

#endif  // BATCHNORMALIZE_H_
