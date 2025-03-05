// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__GradientDescent__h__
#define __PUJ_ML__Optimizer__GradientDescent__h__


#include <PUJ_ML/Optimizer/Base.h>

#include <cmath>


namespace PUJ_ML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class GradientDescent
      : public PUJ_ML::Optimizer::Base< _TModel >
    {
    public:
      using TModel     = _TModel;
      using Self       = GradientDescent;
      using Superclass = PUJ_ML::Optimizer::Base< TModel >;
      using TReal      = typename Superclass::TReal;
      using TNatural   = typename Superclass::TNatural;
      using TMatrix    = typename Superclass::TMatrix;
      using TColumn    = typename Superclass::TColumn;
      using TRow       = typename Superclass::TRow;
      using TMap       = typename Superclass::TMap;
      using TConstMap  = typename Superclass::TConstMap;
      using TDebug     = typename Superclass::TDebug;

    public:
      GradientDescent( TModel& m )
        : Superclass( m )
        {
        }
      virtual ~GradientDescent( ) override
        {
        }

      const TReal& alpha( ) const     { return( this->m_Alpha ); }
      void setAlpha( const TReal& a ) { this->m_Alpha = a; }

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

          TNatural t = 0;
          bool stop = false;
          TColumn G( this->m_Model->size( ) );

          while( !stop )
          {
            t++;

            TReal J_tr = this->m_Model->cost_gradient( G, X_tr, y_tr, this->m_Lambda1, this->m_Lambda2 );
            if( !std::isnan( J_tr ) && !std::isinf( J_tr ) )
            {
              TReal J_te = ( 0 < X_te.rows( ) )? this->m_Model->cost( X_te, y_te ): _M;

              *( this->m_Model ) -= G * this->m_Alpha;

              stop
                =
                this->m_Debug(
                  t, std::sqrt( G.array( ).pow( 2 ).sum( ) ), J_tr, J_te
                  );
            }
            else
              stop = true;
          } // end while
        }

    protected:
      TReal m_Alpha { 1e-2 };
    };
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Optimizer__GradientDescent__h__

// eof - GradientDescent.h
