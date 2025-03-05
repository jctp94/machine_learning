## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, matplotlib.pyplot, numpy, random, sys
sys.path.append( '../lib' )
import PUJ_ML
from GraphicConfussionDebugger import GraphicConfussionDebugger
from ConfussionDebugger import ConfussionDebugger

# -- Parse command line arguments
parser = argparse.ArgumentParser(
    prog = sys.argv[ 0 ],
    description = 'Fit a logistic regression',
    epilog = 'Enjoy!'
    )
parser.add_argument( 'train', type = str )
parser.add_argument( '-test', '--test', type = str, default = '0' )
parser.add_argument(
  '-d', '--delimiter', type = str, default = ' ',
  help = 'Delimiter in the CSV file'
  )
parser.add_argument(
  '-o', '--optimizer', type = str, default = 'Adam',
  help = 'Adam|GradientDescent'
  )
parser.add_argument(
  '-debugger', '--debugger', type = str,
  default = 'Simple', help = 'Simple|Graphic'
  )
parser.add_argument(
  '-v', '--validation', type = str, default = 'MCE',
  help = 'MCE|LOO|KfoldK'
  )
parser.add_argument( '-a', '--alpha', type = float, default = 1e-2 )
parser.add_argument( '-L1', '--L1', type = float, default = 0 )
parser.add_argument( '-L2', '--L2', type = float, default = 0 )
parser.add_argument( '-e', '--epochs', type = int, default = 1000 )
try:
  args = parser.parse_args( )
except BaseException as error:
  sys.exit( 1 )
# end try

# -- Prepare data
D_tr = None
D_te = None
try:
  train_coeff = abs( float( 1 ) - abs( float( args.test ) ) )
  if train_coeff > 1: train_coeff = 1
  if train_coeff < 1:
    D_tr, D_te = \
      PUJ_ML.Helpers.SplitDataForBinaryLabeling(
      numpy.genfromtxt( args.train, delimiter = args.delimiter ),
      train_coeff
      )
  else:
    D_tr = numpy.genfromtxt( args.train, delimiter = args.delimiter )
  # end if
except ValueError:
  D_tr = numpy.genfromtxt( args.train, delimiter = args.delimiter )
  D_te = numpy.genfromtxt( args.test, delimiter = args.delimiter )
# end try

# -- Check sizes
if not D_te is None:
  if D_tr.shape[ 1 ] != D_te.shape[ 1 ]:
    print( 'Data sizes are not compatible.' )
    sys.exit( 1 )
  # end if
# end if

# -- Prepare a logistic regression model
m = PUJ_ML.Model.Regression.Logistic( D_tr.shape[ 1 ] - 1 )
print( 'Initial model: ' + str( m ) )

# -- Prepare optimizer
opt = None
if args.optimizer.lower( ) == 'adam':
  opt = PUJ_ML.Optimizer.Adam( m )
elif args.optimizer.lower( ) == 'gradientdescent':
  opt = PUJ_ML.Optimizer.GradientDescent( m )
# end if
if opt is None:
  print( 'Invalid optimizer "' + args.optimizer + '"' )
  sys.exit( 1 )
# end if
opt.m_Alpha = args.alpha
opt.m_Lambda1 = args.L1
opt.m_Lambda2 = args.L2
if args.debugger.lower( ) == 'simple':
  opt.m_Debug = ConfussionDebugger( args.epochs, m, D_tr, D_te )
elif args.debugger.lower( ) == 'graphic':
  opt.m_Debug = GraphicConfussionDebugger( args.epochs, m, D_tr, D_te )
# end if

# -- Fit model to train data
if args.validation.lower( ) == 'mce':
  opt.fit( D_tr, D_te, validation = 'mce' )
elif args.validation.lower( ) == 'loo':
  opt.fit( D_tr, D_te, validation = 'loo' )
elif args.validation.lower( )[ : 5 ] == 'kfold':
 K = int( args.validation.lower( )[ 5 : ] )
 opt.fit( D_tr, D_te, validation = 'kfold', K = K )
else:
  print( 'Invalid validation strategy.' )
  sys.exit( 1 )
# end if

# Show final model
X_tr = D_tr[ : , : D_tr.shape[ 1 ] - 1 ]
y_tr = numpy.asmatrix( D_tr[ : , -1 ] ).T
X_te, y_te = None, None
if not D_te is None:
  X_te = D_te[ : , : D_te.shape[ 1 ] - 1 ]
  y_te = numpy.asmatrix( D_te[ : , -1 ] ).T
# end if

print( 'Fitted model: ' + str( m ) )
print( 'Training cost = ' + str( m.cost( X_tr, y_tr ) ) )
if not D_te is None:
  print( 'Testing cost  = ' + str( m.cost( X_te, y_te ) ) )
# end if

# -- Compute confussion matrices
K_tr = PUJ_ML.Helpers.Confussion( m, X_tr, y_tr )
print( 'Training confussion =\n', K_tr )
if not D_te is None:
  K_te = PUJ_ML.Helpers.Confussion( m, X_te, y_te )
  print( 'Testing confussion  =\n', K_te )
# end if

# -- ROC curves
ROC_tr = PUJ_ML.Helpers.ROC( m, X_tr, y_tr )
if not D_te is None:
  ROC_te = PUJ_ML.Helpers.ROC( m, X_te, y_te )
# end if

fig, ax = matplotlib.pyplot.subplots( )
ax.plot( ROC_tr[ 0 ], ROC_tr[ 1 ], lw = 1 )
if not D_te is None:
  ax.plot( ROC_te[ 0 ], ROC_te[ 1 ], lw = 1 )
# end if
ax.plot( [ 0, 1 ], [ 0, 1 ], lw = 0.5, linestyle = '--' )
ax.set_aspect( 1 )
matplotlib.pyplot.show( )

## eof - LogisticRegressionFit.py
