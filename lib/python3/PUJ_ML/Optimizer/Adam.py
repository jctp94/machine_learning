## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy
from .GradientDescent import GradientDescent

"""
"""
class Adam( GradientDescent ):
  '''
  '''
  m_Beta1 = 0.9
  m_Beta2 = 0.999

  '''
  '''
  def __init__( self, m ):
    super( ).__init__( m )
  # end def

  '''
  '''
  def fit( self, D_train, D_test ):

    X_tr = D_train[ 0 ]
    y_tr = D_train[ 1 ]
    X_te, y_te = None, None
    if not D_test is None:
      X_te = D_test[ 0 ]
      y_te = D_test[ 1 ]
    # end if

    b1t = self.m_Beta1
    b2t = self.m_Beta2
    mt = numpy.zeros( ( self.m_Model.size( ), 1 ) )
    vt = numpy.zeros( ( self.m_Model.size( ), 1 ) )

    self.m_Model.init( )
    t = 0
    stop = False
    while not stop:
      t += 1

      J_tr, G = self.m_Model.cost_gradient( X_tr, y_tr, self.m_Lambda1, self.m_Lambda2 )
      if not math.isnan( J_tr ) and not math.isinf( J_tr ):
        J_te = None
        if not X_te is None:
          J_te = self.m_Model.cost( X_te, y_te )
        # end if

        cb1t = float( 1 ) / ( float( 1 ) - b1t )
        cb2t = float( 1 ) / ( float( 1 ) - b2t )

        mt = ( mt * self.m_Beta1 ) + ( G * ( float( 1 ) - self.m_Beta1 ) )
        vt = ( vt * self.m_Beta2 ) + ( numpy.multiply( G, G ) * ( float( 1 ) - self.m_Beta2 ) )

        self.m_Model -= \
          numpy.multiply(
          mt * cb1t, float( 1 ) / ( numpy.sqrt( vt * cb2t ) + self.m_Epsilon )
          ) \
          * \
          self.m_Alpha

        if not self.m_Debug is None:
          stop = self.m_Debug( t, ( G.T @ G )[ 0 , 0 ] ** 0.5, J_tr, J_te )
        # end if
      else:
        stop = True
      # end if

      b1t *= self.m_Beta1
      b2t *= self.m_Beta2
    # end while
  # end def

## eof - Adam.py
