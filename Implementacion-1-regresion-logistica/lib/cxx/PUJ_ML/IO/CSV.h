// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__IO__CSV__h__
#define __PUJ_ML__IO__CSV__h__

#include <PUJ_ML/Config.h>

namespace PUJ_ML
{
  namespace IO
  {
    /**
     */
    template< class _TD >
    bool ReadCSV(
      Eigen::EigenBase< _TD >& D, const std::string& fname,
      unsigned long long ignore_first_rows = 0,
      const char& separator = ','
      );
  } // end namespace
} // end namespace

#include <PUJ_ML/IO/CSV.hxx>

#endif // __PUJ_ML__IO__CSV__h__

// eof - CSV.h
