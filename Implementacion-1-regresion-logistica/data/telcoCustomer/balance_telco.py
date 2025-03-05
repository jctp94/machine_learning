## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random, sys

D = numpy.genfromtxt( sys.argv[ 1 ], delimiter = ',', skip_header = 1, dtype=str)

mask = ~((D == "") | (D == " ") | (D == "None")).any(axis=1)


mask = numpy.asmatrix(mask)

# #remove null values
D = D[mask.A1]

# Don't include the first column because it's the customer ID



# Remove rows with null values
print("Number of rows after removing null values.",D.shape[0] )



for i in range(D.shape[1]):
    normalized_X= D
    try:
        float_column = numpy.array(normalized_X[:, i]).flatten().astype(float)
        normalized_X[:, i] = numpy.matrix(float_column).reshape(-1, 1)
    except Exception as e:
        column_unique=numpy.unique(numpy.asarray(D[:,i]))
        for j in range(len(column_unique)):
          normalized_X[numpy.where(D[:,i]==column_unique[j])[0],i]=float(j)
    
X = numpy.asmatrix(normalized_X, dtype=float)

numpy.set_printoptions(suppress=True)
print("X",X)

X = numpy.asmatrix( D[ :, 1 : D.shape[ 1 ] - 1 ], dtype=float )
print("X",X)

y = numpy.asmatrix( D[ : , -1 ], dtype=float ).T

L = numpy.unique( numpy.asarray( y ) )
    


print("Number of columns",X.shape[1])
print("Number of rows",X.shape[0])

print("Number of y columns", y.shape[1])
print("Number of y rows ",y.shape[0])



numpy.set_printoptions(suppress=True)


print("L values",L)


Xz = X[ numpy.where( y <  1)[ 0 ] , : ]
Xo = X[ numpy.where( y >= 1 )[ 0 ] , : ]


n = min( Xz.shape[ 0 ], Xo.shape[ 0 ] )

print("Xz",Xz.shape[0])
print("Xo",Xo.shape[0])

print( 'Balanced size = ', n )

idx_z = [ i for i in range( Xz.shape[ 0 ] ) ]
idx_o = [ i for i in range( Xo.shape[ 0 ] ) ]
random.shuffle( idx_z )
random.shuffle( idx_o )

idx_selected = idx_o if len(idx_z) > len(idx_o) else idx_z




E = numpy.concatenate(
     ( numpy.concatenate( ( Xz[ idx_selected[ : n ] , : ], Xo[ idx_selected[ : n ] , : ] ), axis = 0 ),
       numpy.concatenate( ( numpy.zeros( ( n, 1 ) ), numpy.ones( ( n, 1 ) ) ), axis = 0 ) ), axis = 1 )
idx_e = [ i for i in range( E.shape[ 0 ] ) ]
random.shuffle( idx_e )

print("Final size rows",E.shape[0])
print("Final size columns",E.shape[1])

numpy.savetxt( sys.argv[ 2 ], E[ idx_e , : ], delimiter = ',', fmt = '%.4f' )




