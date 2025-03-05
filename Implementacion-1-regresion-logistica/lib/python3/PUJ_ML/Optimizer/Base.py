## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

"""
"""
class Base:
    
  '''
  '''
  m_Model = None
  m_Lambda1 = 0
  m_Lambda2 = 0
  m_Epsilon = None
  m_Debug = None

  '''
  '''
  def __init__( self, m ):
    self.m_Model = m

    self.m_Epsilon = float( 1 )
    while self.m_Epsilon + 1 > 1:
      self.m_Epsilon /= 2
    # end while
    self.m_Epsilon *= 2
  # end def
# end class

## eof - Base.py
