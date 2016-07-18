#ifndef LA_LIN_ALG_HPP
#define LA_LIN_ALG_HPP

#include <vector>
#include <complex>

#include <matrix.hpp>

void ftMatrixFull(int N, double L, const Math::Matrix<double>& mat, Math::Matrix<double> *res, Math::Matrix<double> *resIm = NULL);
void ftMatrixFullWL(int N, double L, const Math::Matrix<double>& mat, Math::Matrix<double> *res, Math::Matrix<double> *resIm = NULL);

void linearAlgebraCalculation(int N, double L, const std::vector<double> &pk, bool weakLensing, const std::vector<double> &mask, const std::vector<double> &sigmaNoise, const std::vector<double> &dataX, const std::vector<double> &dataGamma1, const std::vector<double> &dataGamma2, std::vector<std::complex<double> > &wfk, std::vector<double> &wfx, std::vector<double> &b, Math::Matrix<double> &fisher, std::vector<double> &signal, std::vector<double> &theta, Math::Matrix<double> *invHessian = NULL);

#endif

