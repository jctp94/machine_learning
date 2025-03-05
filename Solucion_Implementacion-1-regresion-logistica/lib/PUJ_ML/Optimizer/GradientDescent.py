## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math
from .Base import Base

"""
"""
class GradientDescent( Base ):
  '''
  '''
  m_Alpha = 1e-2

  '''
  '''
  def __init__( self, m ):
    super( ).__init__( m )
  # end def

  '''
  '''
  def _fit( self, X_tr, y_tr, X_te, y_te ):

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

        if not self.m_Debug is None:
          stop = self.m_Debug( t, ( G.T @ G )[ 0 , 0 ] ** 0.5, J_tr, J_te )
        # end if

        self.m_Model -= G * self.m_Alpha
      else:
        stop = True
      # end if
    # end while
  # end def

## eof - GradientDescent.py
