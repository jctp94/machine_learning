// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Regression__Linear__h__
#define __PUJ_ML__Model__Regression__Linear__h__

#include <PUJ_ML/Model/Base.h>

namespace PUJ_ML
{
  namespace Model
  {
    namespace Regression
    {
      /**
       */
      template< class _TReal, class _TNatural = unsigned long long >
      class Linear
        : public PUJ_ML::Model::Base< _TReal, _TNatural >
      {
      public:
        using TReal      = _TReal;
        using TNatural   = _TNatural;
        using Self       = Linear;
        using Superclass = PUJ_ML::Model::Base< TReal, TNatural >;
        using TMatrix    = typename Superclass::TMatrix;
        using TColumn    = typename Superclass::TColumn;
        using TRow       = typename Superclass::TRow;
        using TMap       = typename Superclass::TMap;
        using TConstMap  = typename Superclass::TConstMap;

      public:
        Linear( const TNatural& n = 1 );
        virtual ~Linear( ) override;

        template< class _TX >
        auto operator()( const Eigen::EigenBase< _TX >& X ) const;

        /**
         * TODO: Use of L1 regularization is not yet solved
         */
        template< class _TX, class _Ty >
        void fit(
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by,
          const TReal& L1 = 0, const TReal& L2 = 0
          );

        template< class _TG, class _TX, class _Ty >
        TReal cost_gradient(
          Eigen::EigenBase< _TG >& G,
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by,
          const TReal& L1, const TReal& L2
          );

        template< class _TX, class _Ty >
        TReal cost(
          const Eigen::EigenBase< _TX >& X,
          const Eigen::EigenBase< _Ty >& y
          );
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/Regression/Linear.hxx>

#endif // __PUJ_ML__Model__Regression__Linear__h__

// eof - Linear.h
