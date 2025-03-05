// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__Adam__h__
#define __PUJ_ML__Optimizer__Adam__h__

#include <PUJ_ML/Optimizer/GradientDescent.h>




#include <cmath>







namespace PUJ_ML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class Adam
      : public PUJ_ML::Optimizer::GradientDescent< _TModel >
    {
    public:
      using TModel     = _TModel;
      using Self       = Adam;
      using Superclass = PUJ_ML::Optimizer::GradientDescent< TModel >;
      using TReal      = typename Superclass::TReal;
      using TNatural   = typename Superclass::TNatural;
      using TMatrix    = typename Superclass::TMatrix;
      using TColumn    = typename Superclass::TColumn;
      using TRow       = typename Superclass::TRow;
      using TMap       = typename Superclass::TMap;
      using TConstMap  = typename Superclass::TConstMap;
      using TDebug     = typename Superclass::TDebug;

    public:
      Adam( TModel& m )
        : Superclass( m )
        {
        }
      virtual ~Adam( ) override
        {
        }

      const TReal& beta1( ) const { return( this->m_Beta1 ); }
      const TReal& beta2( ) const { return( this->m_Beta2 ); }

      void setBeta1( const TReal& b ) { this->m_Beta1 = b; }
      void setBeta2( const TReal& b ) { this->m_Beta2 = b; }

      template< class _TX_tr, class _Ty_tr, class _TX_te, class _Ty_te >
      void fit(
        const Eigen::EigenBase< _TX_tr >& bX_train,
        const Eigen::EigenBase< _Ty_tr >& by_train,
        const Eigen::EigenBase< _TX_te >& bX_test,
        const Eigen::EigenBase< _Ty_te >& by_test
        )
        {
          static const TReal _0 = TReal( 0 );
          static const TReal _1 = TReal( 1 );
          static const TReal _M = std::numeric_limits< TReal >::max( );

          auto X_tr = bX_train.derived( ).template cast< TReal >( );
          auto y_tr = by_train.derived( ).template cast< TReal >( );
          auto X_te = bX_test.derived( ).template cast< TReal >( );
          auto y_te = by_test.derived( ).template cast< TReal >( );

          TReal b1t = this->m_Beta1;
          TReal b2t = this->m_Beta2;
          TNatural t = 0;
          bool stop = false;
          TColumn G( this->m_Model->size( ) );
          TColumn mt = TColumn::Zero( G.rows( ) );
          TColumn vt = TColumn::Zero( G.rows( ) );

          while( !stop )
          {
            t++;

            TReal J_tr = this->m_Model->cost_gradient( G, X_tr, y_tr, this->m_Lambda1, this->m_Lambda2 );
            if( !std::isnan( J_tr ) && !std::isinf( J_tr ) )
            {
              TReal J_te = ( 0 < X_te.rows( ) )? this->m_Model->cost( X_te, y_te ): _M;
              mt = ( mt * this->m_Beta1 ) + ( G * ( _1 - this->m_Beta1 ) );
              vt = ( vt * this->m_Beta2 ) + ( G.array( ).pow( 2 ) * ( _1 - this->m_Beta2 ) ).matrix( );

              *( this->m_Model ) -= ( ( mt * ( _1 / ( _1 - b1t ) ) ).array( ) / ( ( ( vt * ( _1 / ( _1 - b2t ) ) ).array( ) ).sqrt( ) + this->m_Epsilon ) ).matrix( ) * this->m_Alpha;

              stop
                =
                this->m_Debug(
                  t, std::sqrt( G.array( ).pow( 2 ).sum( ) ), J_tr, J_te
                  );

              b1t *= this->m_Beta1;
              b2t *= this->m_Beta2;
            }
            else
              stop = true;
          } // end while
        }

    protected:
      TReal m_Beta1 { 0.9 };
      TReal m_Beta2 { 0.999 };
    };
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Optimizer__Adam__h__

// eof - Adam.h
