// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizer__Base__h__
#define __PUJ_ML__Optimizer__Base__h__

#include <limits>

namespace PUJ_ML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class Base
    {
    public:
      using TModel    = _TModel;
      using Self      = Base;
      using TReal     = typename TModel::TReal;
      using TNatural  = typename TModel::TNatural;
      using TMatrix   = typename TModel::TMatrix;
      using TColumn   = typename TModel::TColumn;
      using TRow      = typename TModel::TRow;
      using TMap      = typename TModel::TMap;
      using TConstMap = typename TModel::TConstMap;

      using TDebug
      =
        std::function< bool( const TNatural&, const TReal&, const TReal&, const TReal& ) >;

    public:
      Base( TModel& m )
        : m_Model( &m )
        {
        }
      virtual ~Base( )
        {
        }

      TModel* model( ) const        { return( this->m_Model ); }
      const TReal& lambda1( ) const { return( this->m_Lambda1 ); }
      const TReal& lambda2( ) const { return( this->m_Lambda2 ); }
      const TReal& epsilon( ) const { return( this->m_Epsilon ); }

      void setLambda1( const TReal& l ) { this->m_Lambda1 = l; }
      void setLambda2( const TReal& l ) { this->m_Lambda2 = l; }
      void setEpsilon( const TReal& e ) { this->m_Epsilon = e; }
      void setDebug( TDebug d )         { this->m_Debug = d; }

    protected:
      TModel* m_Model { nullptr };

      TReal m_Lambda1 { 0 };
      TReal m_Lambda2 { 0 };
      TReal m_Epsilon { std::numeric_limits< TReal >::epsilon( ) };

      TDebug m_Debug
        {
          [](
            const TNatural& t,
            const TReal& nG,
            const TReal& J_tr, const TReal& J_te
            ) -> bool
          {
            return( false );
          }
        };
    };
  } // end namespace
} // end namespace

#endif // __PUJ_ML__Optimizer__Base__h__

// eof - Base.h
