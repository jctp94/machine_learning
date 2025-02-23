## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Linear import Linear

"""
"""
class Logistic( Linear ):

  '''
  '''
  m_Epsilon = 0

  '''
  '''
  def __init__( self, n = 1 ):
    super( ).__init__( n )

    self.m_Epsilon = float( 1 )
    while self.m_Epsilon + 1 > 1:
      self.m_Epsilon /= 2
    # end while
    self.m_Epsilon *= 2
    
  # end def

  '''
  '''
  def __call__( self, X, threshold = False ):
    z = super( ).__call__( X )
    if not z is None:
      if threshold:
        return ( self( X, False ) >= 0.5 ).astype( float )
      else:
        return float( 1 ) / ( float( 1 ) + numpy.exp( -z ) )
      # end if
    else:
      return None
    # end if
  # end def

  '''
  '''
  def fit( self, X, y, L1 = 0, L2 = 0 ):
    raise AssertionError(
      'There is no closed solution for a logistic regression.'
      )
  # end def

  '''
  '''
  def cost_gradient( self, X, y, L1, L2 ):
    z = self( X )

    zi = numpy.where( y == 0 )[ 0 ]
    oi = numpy.where( y == 1 )[ 0 ]

    J  = numpy.log( float( 1 ) - z[ zi , : ] + self.m_Epsilon ).mean( )
    J += numpy.log( z[ oi , : ] + self.m_Epsilon ).mean( )

    G = numpy.zeros( self.m_P.shape )
    G[ 0 ] = ( z - y ).mean( )
    G[ 1 : ] = numpy.multiply( X, z - y ).mean( axis = 0 ).T

    return ( -J, numpy.asmatrix( G ) + self._regularization( L1, L2 ) )
  # end def

  '''
  '''
  def cost( self, X, y ):
    z = self( X )
    zi = numpy.where( y == 0 )[ 0 ]
    oi = numpy.where( y == 1 )[ 0 ]
    J  = numpy.log( float( 1 ) - z[ zi , : ] + self.m_Epsilon ).mean( )
    J += numpy.log( z[ oi , : ] + self.m_Epsilon ).mean( )
    return -J
  # end def

# end class

## eof - Logistic.py
