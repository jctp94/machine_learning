## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
sys.path.append( '../../lib/python3' )
import matplotlib.pyplot, numpy
import PUJ_ML
from SimpleDebugger import SimpleDebugger

if len( sys.argv ) < 2:
  print(
    'Usage: python '
    +
    sys.argv[ 0 ]
    +
    ' data.csv [alpha=1e-2] [L1=0] [L2=0] [train_size=0.7] [max_epochs=1000]'
    )
  sys.exit( 1 )
# end if
data_fname = sys.argv[ 1 ]
alpha, L1, L2, train_size, max_epochs = 1e-2, 0, 0, 0.7, 1000
if len( sys.argv ) > 2: alpha = float( sys.argv[ 2 ] )
if len( sys.argv ) > 3: L1 = float( sys.argv[ 3 ] )
if len( sys.argv ) > 4: L2 = float( sys.argv[ 4 ] )
if len( sys.argv ) > 5: train_size = float( sys.argv[ 5 ] )
if len( sys.argv ) > 6: max_epochs = float( sys.argv[ 6 ] )

# Get train data
D = numpy.genfromtxt( data_fname, delimiter = ' ' )
X = numpy.asmatrix( D[ : , 0 : D.shape[ 1 ] - 1 ] )
y = numpy.asmatrix( D[ : , -1 ] ).T

if abs( train_size ) < 1:
  n_tr = int( float( X.shape[ 0 ] ) * abs( train_size ) )
  D_tr = ( X[ 0 : n_tr , : ], y[ 0 : n_tr , : ] )
  D_te = ( X[ n_tr : , : ], y[ n_tr : , : ] )
else:
  D_tr = ( X, y )
  D_te = ( None, None )
# end if

# Prepare a model
m = PUJ_ML.Model.Regression.Linear( X.shape[ 1 ] )
print( 'Initial model: ' + str( m ) )

# Fit model to train data
opt = PUJ_ML.Optimizer.Adam( m )
opt.m_Alpha = alpha
opt.m_Lambda1 = L1
opt.m_Lambda2 = L2
opt.m_Debug = SimpleDebugger( max_epochs )
opt.fit( D_tr, D_te )

# Show final model
print( 'Fitted model: ' + str( m ) )
print( 'Cost = ' + str( m.cost( X, y ) ) )

## eof - LinearRegressionAdamFit.py
