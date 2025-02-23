// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>

/**
 */
template< class _TModel >
class SimpleDebugger
{
public:
  using TModel   = _TModel;
  using TReal    = typename TModel::TReal;
  using TNatural = typename TModel::TNatural;

public:
  SimpleDebugger( const TNatural& max_epochs )
    : m_MaxEpochs( max_epochs )
    {
    }

  bool operator()( const TNatural& e, const TReal& nG, const TReal& J_tr, const TReal& J_te )
    {
      std::cout << e << " " << nG << " " << J_tr << " " << J_te << std::endl;
      return( !( e < this->m_MaxEpochs ) );
    }

protected:
  TNatural m_MaxEpochs;
};

// eof - SimpleDebugger.h
