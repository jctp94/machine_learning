// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <PUJ_ML/Model/Base.h>
#include <cstring>

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::Base< _TReal, _TNatural >::
Base( const TNatural& n )
{
  this->_resize( n );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
PUJ_ML::Model::Base< _TReal, _TNatural >::
~Base( )
{
  this->_resize( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
typename PUJ_ML::Model::Base< _TReal, _TNatural >::
TReal& PUJ_ML::Model::Base< _TReal, _TNatural >::
operator[]( const TNatural& i )
{
  static TReal _z;
  if( i < this->m_S )
    return( this->m_P[ i ] );
  else
  {
    _z = TReal( 0 );
    return( _z );
  } // end if
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const typename PUJ_ML::Model::Base< _TReal, _TNatural >::
TReal& PUJ_ML::Model::Base< _TReal, _TNatural >::
operator[]( const TNatural& i ) const
{
  static const TReal _z = TReal( 0 );
  if( i < this->m_S )
    return( this->m_P[ i ] );
  else
    return( _z );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
const typename PUJ_ML::Model::Base< _TReal, _TNatural >::
TNatural& PUJ_ML::Model::Base< _TReal, _TNatural >::
size( ) const
{
  return( this->m_S );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::Base< _TReal, _TNatural >::
init( )
{
  TMap( this->m_P, this->m_S, 0 ) *= TReal( 0 );
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::Base< _TReal, _TNatural >::
_resize( const TNatural& n )
{
  if( this->m_P != nullptr )
    std::free( this->m_P );
  this->m_S = n;
  if( n > 0 )
    this->m_P
      =
      reinterpret_cast< TReal* >( std::calloc( n, sizeof( TReal ) ) );
  else
    this->m_P = nullptr;
}

// -------------------------------------------------------------------------
template< class _TReal, class _TNatural >
void PUJ_ML::Model::Base< _TReal, _TNatural >::
_to_stream( std::ostream& o ) const
{
  o << this->m_S;
  for( TNatural i = 0; i < this->m_S; ++i )
    o << " " << this->m_P[ i ];
}

// -------------------------------------------------------------------------
namespace PUJ_ML
{
  namespace Model
  {
    template class PUJ_ML_EXPORT Base< float, unsigned int >;
    template class PUJ_ML_EXPORT Base< float, unsigned long >;
    template class PUJ_ML_EXPORT Base< float, unsigned long long >;

    template class PUJ_ML_EXPORT Base< double, unsigned int >;
    template class PUJ_ML_EXPORT Base< double, unsigned long >;
    template class PUJ_ML_EXPORT Base< double, unsigned long long >;

    template class PUJ_ML_EXPORT Base< long double, unsigned int >;
    template class PUJ_ML_EXPORT Base< long double, unsigned long >;
    template class PUJ_ML_EXPORT Base< long double, unsigned long long >;
  } // end namespace
} // end namespace

// eof - Base.cxx
