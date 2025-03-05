## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random

'''
'''
def SplitDataForBinaryLabeling( A, train_coeff ):

  idx_z = numpy.where( A[ : , -1 ] == 0 )[ 0 ].tolist( )
  idx_o = numpy.where( A[ : , -1 ] == 1 )[ 0 ].tolist( )
  random.shuffle( idx_z )
  random.shuffle( idx_o )

  n = min( len( idx_z ), len( idx_o ) )
  n_tr = int( float( n ) * train_coeff )

  idx_tr = idx_z[ : n_tr ] + idx_o[ : n_tr ]
  random.shuffle( idx_tr )

  D_tr = A[ idx_tr , : ]

  D_te = None
  if n_tr < n:
    idx_te = idx_z[ n_tr : n ] + idx_o[ n_tr : n ]
    random.shuffle( idx_te )
    D_te = A[ idx_te , : ]
  # end if

  return ( D_tr, D_te )
# end def

'''
'''
def Confussion( m, X, y ):
  z = m( X, True )
  yp = numpy.concatenate( ( float( 1 ) - z, z ), axis = 1 )
  yo = numpy.concatenate( ( float( 1 ) - y, y ), axis = 1 )
  K = yo.T @ yp

  TP = float( K[ 0 , 0 ] )
  TN = float( K[ 1 , 1 ] )
  FN = float( K[ 0 , 1 ] )
  FP = float( K[ 1 , 0 ] )

  sensibility = 0
  specificity = 0
  accuracy = 0
  F1 = 0

  if TP + FN != 0: sensibility = TP / ( TP + FN )
  if TN + FP != 0: specificity = TN / ( TN + FP )
  if TP + FP != 0: accuracy = TP / ( TP + FP )
  if TP + ( ( FP + FN ) / 2 ) != 0: F1 = TP / ( TP + ( ( FP + FN ) / 2 ) )
  return ( K, sensibility, specificity, accuracy, F1 )
# end def

'''
'''
def ROC( m, X, y ):

  y_true = y.T.tolist( )[ 0 ]
  y_scores = m( X ).T.tolist( )[ 0 ]
  D = sorted( zip( y_scores, y_true ), reverse = True )

  n_pos = sum( y_true )
  n_neg = len( y_true ) - n_pos

  fpr = [ 0 ]
  tpr = [ 0 ]

  tp = 0
  fp = 0

  for i in range( len( D ) ):
    score, true_label = D[ i ]

    if i == 0 or score != D[ i - 1 ][ 0 ]:
      fpr.append( fp / n_neg )
      tpr.append( tp / n_pos )
    # end if

    if true_label == 1:
      tp += 1
    else:
      fp += 1
    # end if
  # end for

  fpr.append( 1 )
  tpr.append( 1 )

  return ( fpr, tpr )
# end def

## eof - Helpers.py
