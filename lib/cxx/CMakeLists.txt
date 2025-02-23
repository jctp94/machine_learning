## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

file(GLOB_RECURSE _c ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
file(GLOB_RECURSE _cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE _cxx ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)
file(GLOB_RECURSE _h ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE _hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE _hxx ${CMAKE_CURRENT_SOURCE_DIR}/*.hxx)

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_library(PUJ_ML SHARED "${_c};${_cpp};${_cxx}")
generate_export_header(
  PUJ_ML
  EXPORT_MACRO_NAME PUJ_ML_EXPORT
  EXPORT_FILE_NAME PUJ_ML/Export.h
  )
target_link_libraries(PUJ_ML Eigen3::Eigen)

## eof - CMakeLists.txt
