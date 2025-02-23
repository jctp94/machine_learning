// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Regression__Logistic__h__
#define __PUJ_ML__Model__Regression__Logistic__h__

#include <PUJ_ML/Model/Regression/Linear.h>

namespace PUJ_ML
{
  namespace Model
  {
    namespace Regression
    {
      /**
       */
      template< class _TReal, class _TNatural = unsigned long long >
      class Logistic
        : public PUJ_ML::Model::Regression::Linear< _TReal, _TNatural >
      {
      public:
        using TReal      = _TReal;
        using TNatural   = _TNatural;
        using Superclass
        =
          PUJ_ML::Model::Regression::Linear< TReal, TNatural >;
        using Self       = Logistic;
        using TMatrix    = typename Superclass::TMatrix;
        using TColumn    = typename Superclass::TColumn;
        using TRow       = typename Superclass::TRow;
        using TMap       = typename Superclass::TMap;
        using TConstMap  = typename Superclass::TConstMap;

      protected:
        using TIdx = Eigen::Index;

      public:
        Logistic( const TNatural& n = 1 );
        virtual ~Logistic( ) override;

        template< class _TX >
        auto operator()( const Eigen::EigenBase< _TX >& X, bool threshold = false ) const;

        /**
         * TODO: This method has no sense in a logistic regression
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

      protected:
        /**
         */
        struct SVisitor
        {
          SVisitor( const TColumn& Z );
          void init( const TReal& y, const TIdx& i, const TIdx& j );
          void operator()( const TReal& y, const TIdx& i, const TIdx& j );

          const TColumn* Z;
          TReal J;
          TReal E;
          TReal D;
        };
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/Regression/Logistic.hxx>

#endif // __PUJ_ML__Model__Regression__Logistic__h__

// eof - Logistic.h
