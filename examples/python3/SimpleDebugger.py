## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import time

"""
"""
class SimpleDebugger:

  '''
  '''
  def __init__( self, max_epochs ):
    self.m_MaxEpochs = max_epochs
  # end def

  '''
  '''
  def __call__( self, t, nG, J_train, J_test ):
    print( t, nG, J_train, J_test )
    return not ( t < self.m_MaxEpochs )
  # end def
# end class

## eof - SimpleDebugger.py
