// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <iostream>
#include <sstream>

#include <PUJ_ML/IO/CSV.h>
#include <PUJ_ML/Model/Regression/Logistic.h>
#include <PUJ_ML/Optimizer/Adam.h>

#include "SimpleDebugger.h"

int main( int argc, char** argv )
{
  using TReal = long double;
  using TNatural = unsigned long long;
  using TModel = PUJ_ML::Model::Regression::Logistic< TReal, TNatural >;
  using TMatrix = TModel::TMatrix;
  using TColumn = TModel::TColumn;

  if( argc < 2 )
  {
    std::cerr
      << "Usage: " << argv[ 0 ]
      << " data.csv [alpha=1e-2] [beta1=0.9] [beta2=0.999] [L1=0] [L2=0]"
      << " [train_size=0.7] [max_epochs=1000]"
      << std::endl;
    return( EXIT_FAILURE );
  } // end if
  std::string data_fname = argv[ 1 ];
  TReal alpha = 1e-2;
  TReal beta1 = 0.9;
  TReal beta2 = 0.999;
  TReal L1 = 0;
  TReal L2 = 0;
  TReal train_size = 0.7;
  TNatural max_epochs = 1000;
  if( argc > 2 ) std::istringstream( argv[ 2 ] ) >> alpha;
  if( argc > 3 ) std::istringstream( argv[ 3 ] ) >> beta1;
  if( argc > 4 ) std::istringstream( argv[ 4 ] ) >> beta2;
  if( argc > 5 ) std::istringstream( argv[ 5 ] ) >> L1;
  if( argc > 6 ) std::istringstream( argv[ 6 ] ) >> L2;
  if( argc > 7 ) std::istringstream( argv[ 7 ] ) >> train_size;
  if( argc > 8 ) std::istringstream( argv[ 8 ] ) >> max_epochs;

  // Get train data
  TMatrix D;
  PUJ_ML::IO::ReadCSV( D, data_fname, 0, ' ' );
  auto X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
  auto y = D.block( 0, D.cols( ) - 1, D.rows( ), 1 );

  TNatural n_tr = X.rows( );
  if( std::fabs( train_size ) < 1 )
    n_tr = TNatural( TReal( X.rows( ) ) * std::fabs( train_size ) );
  TMatrix X_tr = X.block( 0, 0, n_tr, X.cols( ) );
  TColumn y_tr = y.block( 0, 0, n_tr, y.cols( ) );
  TMatrix X_te = X.block( n_tr, 0, X.rows( ) - n_tr, X.cols( ) );
  TColumn y_te = y.block( n_tr, 0, y.rows( ) - n_tr, y.cols( ) );

  // Prepare a model
  TModel m( X_tr.cols( ) );
  std::cout << "Initial model: " << m << std::endl;

  // Fit model to train data
  PUJ_ML::Optimizer::Adam< TModel > opt( m );
  opt.setAlpha( alpha );
  opt.setBeta1( beta1 );
  opt.setBeta2( beta2 );
  opt.setLambda1( L1 );
  opt.setLambda2( L2 );
  opt.setDebug( SimpleDebugger< TModel >( max_epochs ) );
  opt.fit( X_tr, y_tr, X_te, y_te );

  // Show final model
  std::cout << "Fitted model: " << m << std::endl;
  std::cout << "Cost = " << m.cost( X, y ) << std::endl;

  return( EXIT_SUCCESS );
}

// eof - LogisticRegressionAdamFit.cxx
