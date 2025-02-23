## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import matplotlib.pyplot, time

"""
"""
class GraphicDebugger:

  '''
  '''
  def __init__( self, max_epochs, sleep = 1e-4 ):
    self.m_MaxEpochs = max_epochs
    self.m_Sleep = sleep
    self.m_MaxSize = 500
    self.m_RenderOffset = 100

    self.m_Fig = None
    self.m_Ax = None
    self.m_LineTr = None
    self.m_LineTe = None
    self.m_AxX = []
    self.m_AxTr = []
    self.m_AxTe = []
  # end def

  '''
  '''
  def __call__( self, t, g, J_train, J_test ):

    stop = not ( t < self.m_MaxEpochs )

    # Initialize plots
    if len( self.m_AxX ) == 0:
      self.m_Fig, self.m_Ax = matplotlib.pyplot.subplots( )
      self.m_LineTr, = self.m_Ax.plot( [], [] )
      self.m_LineTe, = self.m_Ax.plot( [], [] )
      self.m_Ax.set_xlim( 0, 1 )
      self.m_Ax.set_ylim( 0, 1 )
      self.m_Ax.set_xlabel( 'Epoch/iteration' )
      self.m_Ax.set_ylabel( 'Cost/Loss' )
      self.m_Ax.set_title( 'Cost/Loss evolution' )
      matplotlib.pyplot.ion( ) 
      matplotlib.pyplot.show( )
    # end if

    # Update plots
    self.m_AxX += [ t ]
    self.m_AxTr += [ J_train ]
    if not J_test is None:
      self.m_AxTe += [ J_test ]
    else:
      self.m_AxTe += [ 0 ]
    # end if
    if len( self.m_AxX ) > self.m_MaxSize:
      self.m_AxX = self.m_AxX[ 1 : self.m_MaxSize ]
      self.m_AxTr = self.m_AxTr[ 1 : self.m_MaxSize ]
      self.m_AxTe = self.m_AxTe[ 1 : self.m_MaxSize ]
    # end if
    if t % self.m_RenderOffset == 0:
      self.m_LineTr.set_data( self.m_AxX, self.m_AxTr )
      self.m_LineTe.set_data( self.m_AxX, self.m_AxTe )
      self.m_Ax.set_xlim( self.m_AxX[ 0 ], self.m_AxX[ -1 ] )
      self.m_Ax.set_ylim(
        min( min( self.m_AxTr ), min( self.m_AxTe ) ),
        max( max( self.m_AxTr ), max( self.m_AxTe ) ) * 2
        )
      self.m_Fig.canvas.draw( )
      self.m_Fig.canvas.flush_events( )
      time.sleep( self.m_Sleep )
    # end if

    # Finish visualization
    if stop:
      matplotlib.pyplot.ioff( ) 
      matplotlib.pyplot.show( )
      matplotlib.pyplot.close( self.m_Fig )
    # end if

    return stop
  # end def
# end class

## eof - GraphicDebugger.py
