## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random, sys

D = numpy.genfromtxt( sys.argv[ 1 ], delimiter = ',', skip_header = 1 )
X = numpy.asmatrix( D[ : , 0 : D.shape[ 1 ] - 1 ] )
y = numpy.asmatrix( D[ : , -1 ] ).T
L = numpy.unique( numpy.asarray( y ) )

print( 'Histogram' )
for l in L:
  print( l, numpy.sum( y == l ) )
# end for
print( '============' )

Xz = X[ numpy.where( y == 0 )[ 0 ] , : ]
Xo = X[ numpy.where( y == 1 )[ 0 ] , : ]
n = min( Xz.shape[ 0 ], Xo.shape[ 0 ] )
print( 'Balanced size = ', n )

idx_z = [ i for i in range( Xz.shape[ 0 ] ) ]
idx_o = [ i for i in range( Xo.shape[ 0 ] ) ]
random.shuffle( idx_z )
random.shuffle( idx_o )

E = numpy.concatenate( ( numpy.concatenate( ( Xz[ idx_z[ : n ] , : ], Xo[ idx_z[ : n ] , : ] ), axis = 0 ), numpy.concatenate( ( numpy.zeros( ( n, 1 ) ), numpy.ones( ( n, 1 ) ) ), axis = 0 ) ), axis = 1 )
idx_e = [ i for i in range( E.shape[ 0 ] ) ]
random.shuffle( idx_e )

numpy.savetxt( sys.argv[ 2 ], E[ idx_e , : ], delimiter = ',', fmt = '%.4f' )

## eof - sandbox.py
