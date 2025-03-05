## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse, matplotlib.pyplot, numpy, random, sys
sys.path.append( '../../lib/python3' )
import PUJ_ML
from SimpleDebugger import SimpleDebugger
from GraphicDebugger import GraphicDebugger

def divide_by_groups(D_train, k):
 
    group_size = D_train.shape[0] // k
    remainder = D_train.shape[0] % k
 
    groups = []
    index = 0
 
    for i in range(k):
        add_size = group_size + 1 if remainder > 0 else group_size
        groups.append([index, index + add_size -1])
        index += add_size
        remainder -= 1

    return groups
# end divide_by_groups

# -- Parse command line arguments
parser = argparse.ArgumentParser(
    prog = sys.argv[ 0 ],
    description = 'Fit a logistic regression',
    epilog = 'Enjoy!'
    )
parser.add_argument( 'train', type = str )
parser.add_argument( '-test', '--test', type = str, default = '0.3' )
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
  help = 'MCE|LOO|Kfold'
  )
parser.add_argument( '-a', '--alpha', type = float, default = 1e-2 )
parser.add_argument( '-L1', '--L1', type = float, default = 0 )
parser.add_argument( '-L2', '--L2', type = float, default = 0 )
parser.add_argument( '-e', '--epochs', type = int, default = 1000 )
parser.add_argument( '-k', '--k', type = int, default = 5 )

try:
  args = parser.parse_args( )
except BaseException as error:
  sys.exit( 1 )
# end try

# -- Prepare data
A = numpy.genfromtxt( args.train, delimiter = args.delimiter )
D_tr = ( None, None )
D_te = ( None, None )
try:
  train_coeff = abs( float( 1 ) - abs( float( args.test ) ) )
  if train_coeff > 1: train_coeff = 1
  D_tr, D_te = PUJ_ML.Helpers.SplitDataForBinaryLabeling( A, train_coeff )
except ValueError:
  B = numpy.genfromtxt( args.test, delimiter = args.delimiter )
  D_tr = (
    numpy.asmatrix( A[ : , 0 : A.shape[ 1 ] - 1 ] ),
    numpy.asmatrix( A[ : , -1 ] ).T
    )
  D_te = (
    numpy.asmatrix( B[ : , 0 : B.shape[ 1 ] - 1 ] ),
    numpy.asmatrix( B[ : , -1 ] ).T
    )
# end try

# -- Check sizes
if not D_te[ 0 ] is None:
  if \
    D_tr[ 0 ].shape[ 1 ] != D_te[ 0 ].shape[ 1 ] \
    or \
    D_tr[ 1 ].shape[ 1 ] != D_te[ 1 ].shape[ 1 ]\
    :
    print( 'Data sizes are not compatible.' )
    sys.exit( 1 )
  # end if
# end if

# -- Prepare a logistic regression model
m = PUJ_ML.Model.Regression.Logistic( D_tr[ 0 ].shape[ 1 ] )
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
  opt.m_Debug = SimpleDebugger( args.epochs )
elif args.debugger.lower( ) == 'graphic':
  opt.m_Debug = GraphicDebugger( args.epochs )
# end if

# -- Fit model to train data
if args.validation.lower( ) == 'mce':
  print("D_tr", D_tr)
  print("D_te", D_te)
  opt.fit( D_tr, D_te )
  print("type D_te", type(D_te))
  print("type D_te[0]", type(D_te[0]))
  print("type D_te[1]", type(D_te[1]))
  print("fit finished")
elif args.validation.lower( ) == 'loo':
  for i in range(D_tr[0].shape[0]):
    train_data = (numpy.delete(D_tr[0], i, axis=0),
                  numpy.delete(D_tr[1], i, axis=0))

    test_data = (D_tr[0][i],D_tr[1][i])
    opt.fit(train_data, test_data)
elif args.validation.lower( )[ : 5 ] == 'kfold':
  K = int( args.k ) 
  row_groups = divide_by_groups(D_tr[0], K)
  for idx, rows_to_remove in enumerate(row_groups):
    train_data = (numpy.delete(D_tr[0], rows_to_remove, axis=0),
                  numpy.delete(D_tr[1], rows_to_remove, axis=0))
    test_data = (D_tr[0][rows_to_remove], D_tr[1][rows_to_remove])
    opt.fit(train_data, test_data)
else:
  print( 'Invalid validation strategy.' )
  sys.exit( 1 )
# end if

# Show final model
print( 'Fitted model: ' + str( m ) )
print( 'Training cost = ' + str( m.cost( D_tr[ 0 ], D_tr[ 1 ] ) ) )
if not D_te[ 0 ] is None and not D_te[ 1 ] is None:
  print( 'Testing cost  = ' + str( m.cost( D_te[ 0 ], D_te[ 1 ] ) ) )
# end if

# -- Compute confussion matrices
K_tr = PUJ_ML.Helpers.Confussion( m, D_tr[ 0 ], D_tr[ 1 ] )
print( 'Training confussion =\n', K_tr )

if not D_te[ 0 ] is None and not D_te[ 1 ] is None:
  K_te = PUJ_ML.Helpers.Confussion( m, D_te[ 0 ], D_te[ 1 ] )
  print( 'Testing confussion  =\n', K_te )
# end if

# -- ROC curves
ROC_tr = PUJ_ML.Helpers.ROC( m, D_tr[ 0 ], D_tr[ 1 ] )
if not D_te[ 0 ] is None and not D_te[ 1 ] is None:
  ROC_te = PUJ_ML.Helpers.ROC( m, D_te[ 0 ], D_te[ 1 ] )
# end if

fig, ax = matplotlib.pyplot.subplots( )
ax.plot( ROC_tr[ 0 ], ROC_tr[ 1 ], lw = 1 )
if not D_te[ 0 ] is None and not D_te[ 1 ] is None:
  ax.plot( ROC_te[ 0 ], ROC_te[ 1 ], lw = 1 )
# end if
ax.plot( [ 0, 1 ], [ 0, 1 ], lw = 0.5, linestyle = '--' )
ax.set_aspect( 1 )
matplotlib.pyplot.show( )



## eof - LogisticRegressionFit.py
