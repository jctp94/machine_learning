## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

include_directories(
  ${PROJECT_BINARY_DIR}/lib/cxx
  ${PROJECT_SOURCE_DIR}/lib/cxx
  )

set(
  _execs
  LinearRegressionClosedFit
  LinearRegressionGradientDescentFit
  LinearRegressionAdamFit
  LogisticRegressionGradientDescentFit
  LogisticRegressionAdamFit
  )

foreach(_e ${_execs})
  add_executable(PUJ_ML_${_e} ${_e}.cxx)
  target_link_libraries(PUJ_ML_${_e} PUJ_ML)
endforeach()


## eof - CMakeLists.txt
