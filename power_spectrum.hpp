#ifndef LA_POWER_SPECTRUM_HPP
#define LA_POWER_SPECTRUM_HPP

#include <vector>
#include <complex>

#include <math_constants.hpp>
#include <function.hpp>
#include <matrix.hpp>
#include <macros.hpp>

class SimplePowerSpectrum : public Math::RealFunction
{
public:
    SimplePowerSpectrum(double k0 = 0.015) : k0_(k0), norm_(1) {}
    ~SimplePowerSpectrum() {}

    double evaluate(double k) const { return norm_ * unnormalized(k); }

    // normalize so that in real space the correlation between points with distance d is corr
    void normalize(double d = 4.0, double corr = 1.0);
    void normalize2(double d = 4.0, double corr = 1.0);

private:
    double unnormalized(double k) const
    {
        check(k >= 0, "");
        const double ratio = k / k0_;
        return k / ((1 + ratio * ratio) * (1 + ratio * ratio));
    }

private:
    const double k0_;
    double norm_;
};

void discretePowerSpectrum(const Math::RealFunction& p, double L, int N, std::vector<double> *res);
void discretePowerSpectrum(const Math::RealFunction& p, double L1, double L2, int N1, int N2, std::vector<double> *res);
void discretePowerSpectrum(const Math::RealFunction& p, double L1, double L2, double L3, int N1, int N2, int N3, std::vector<double> *res);

void discretePowerSpectrum(int N, double L, const std::vector<int> &bins, const std::vector<double> &p, std::vector<double> *res);
void discretePowerSpectrum(int N1, int N2, double L1, double L2, const std::vector<int> &bins, const std::vector<double> &p, std::vector<double> *res);
void discretePowerSpectrum(int N1, int N2, int N3, double L1, double L2, double L3, const std::vector<int> &bins, const std::vector<double> &p, std::vector<double> *res);

void generateDeltaK(int N, const std::vector<double>& pk, std::vector<std::complex<double> > *deltaK, int seed = 0, std::vector<double> *realBuffer = NULL, bool z2 = false, bool normalizePower = false, bool normalizePerBin = false);
void generateDeltaK(int N1, int N2, const std::vector<double>& pk, std::vector<std::complex<double> > *deltaK, int seed = 0, std::vector<double> *realBuffer = NULL, bool z2 = false, bool normalizePower = false, bool normalizePerBin = false);
void generateDeltaK(int N1, int N2, int N3, const std::vector<double>& pk, std::vector<std::complex<double> > *deltaK, int seed = 0, std::vector<double> *realBuffer = NULL, bool z2 = false, bool normalizePower = false, bool normalizePerBin = false);

void deltaK2deltaX(int N, const std::vector<std::complex<double> >& deltaK, std::vector<double> *deltaX, double L);
void deltaX2deltaK(int N, const std::vector<double>& deltaX, std::vector<std::complex<double> > *deltaK, double L);

void deltaK2deltaX(int N1, int N2, std::vector<std::complex<double> >& deltaK, std::vector<double> *deltaX, double L1, double L2, std::vector<std::complex<double> > *buffer = NULL, bool preserveInput = true);
void deltaX2deltaK(int N1, int N2, const std::vector<double>& deltaX, std::vector<std::complex<double> > *deltaK, double L1, double L2);

void deltaK2deltaX(int N1, int N2, int N3, std::vector<std::complex<double> >& deltaK, std::vector<double> *deltaX, double L1, double L2, double L3, std::vector<std::complex<double> > *buffer = NULL, bool preserveInput = true);
void deltaX2deltaK(int N1, int N2, int N3, const std::vector<double>& deltaX, std::vector<std::complex<double> > *deltaK, double L1, double L2, double L3);

double priorK(int N, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK);
double priorX(int N, const std::vector<double>& pk, const std::vector<double>& deltaX, double L, std::vector<std::complex<double> > *deltaK = NULL);

double priorK(int N1, int N2, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK);
double priorX(int N1, int N2, const std::vector<double>& pk, const std::vector<double>& deltaX, double L1, double L2, std::vector<std::complex<double> > *deltaK = NULL);

double priorK(int N1, int N2, int N3, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK);
double priorX(int N1, int N2, int N3, const std::vector<double>& pk, const std::vector<double>& deltaX, double L1, double L2, double L3, std::vector<std::complex<double> > *deltaK = NULL);

void priorKDeriv(int N, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK, std::vector<std::complex<double> > *res);
void priorXDeriv(int N, const std::vector<double>& pk, const std::vector<double>& deltaX, double L, std::vector<double>* res, std::vector<std::complex<double> > *deltaK = NULL, std::vector<std::complex<double> > *pkDeriv = NULL);

void priorKDeriv(int N1, int N2, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK, std::vector<std::complex<double> > *res);
void priorXDeriv(int N1, int N2, const std::vector<double>& pk, const std::vector<double>& deltaX, double L1, double L2, std::vector<double>* res, std::vector<std::complex<double> > *deltaK = NULL, std::vector<std::complex<double> > *pkDeriv = NULL);

void priorKDeriv(int N1, int N2, int N3, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK, std::vector<std::complex<double> > *res);
void priorXDeriv(int N1, int N2, int N3, const std::vector<double>& pk, const std::vector<double>& deltaX, double L1, double L2, double L3, std::vector<double>* res, std::vector<std::complex<double> > *deltaK = NULL, std::vector<std::complex<double> > *pkDeriv = NULL);

void covarianceMatrix(int N, const std::vector<double>& pk, double L, Math::SymmetricMatrix<double> *covMat, std::vector<std::complex<double> >* complexBuf = NULL, std::vector<double> *realBuf = NULL);
void covarianceMatrix(int N1, int N2, const std::vector<double>& pk, double L1, double L2, Math::SymmetricMatrix<double> *covMat, std::vector<std::complex<double> >* complexBuf = NULL, std::vector<double> *realBuf = NULL);
void covarianceMatrix(int N1, int N2, int N3, const std::vector<double>& pk, double L1, double L2, double L3, Math::SymmetricMatrix<double> *covMat, std::vector<std::complex<double> >* complexBuf = NULL, std::vector<double> *realBuf = NULL);

void power(int N, double L, const std::vector<std::complex<double> > deltaK, std::vector<double> *kVals, std::vector<double> *pVals);
void power(int N1, int N2, double L1, double L2, const std::vector<std::complex<double> > deltaK, std::vector<double> *kVals, std::vector<double> *pVals);
void power(int N1, int N2, int N3, double L1, double L2, double L3, const std::vector<std::complex<double> > deltaK, std::vector<double> *kVals, std::vector<double> *pVals);

int powerSpectrumBins(int N, double L, std::vector<int> *bins, std::vector<double> *kBins = NULL);
int powerSpectrumBins(int N1, int N2, double L1, double L2, std::vector<int> *bins, std::vector<double> *kBins = NULL);
int powerSpectrumBins(int N1, int N2, int N3, double L1, double L2, double L3, std::vector<int> *bins, std::vector<double> *kBins = NULL);

int powerSpectrumBinsFull(int N, double L, std::vector<int> *bins, std::vector<double> *kBins = NULL);
int powerSpectrumBinsFull(int N1, int N2, double L1, double L2, std::vector<int> *bins, std::vector<double> *kBins = NULL);
int powerSpectrumBinsFull(int N1, int N2, int N3, double L1, double L2, double L3, std::vector<int> *bins, std::vector<double> *kBins = NULL);

void generateWhiteNoise(int N, int seed, std::vector<double> &res, bool z2 = false, bool normalizePower = false, bool normalizePerBin = false);
void generateWhiteNoise(int N1, int N2, int seed, std::vector<double> &res, bool z2 = false, bool normalizePower = false, bool normalizePerBin = false);
void generateWhiteNoise(int N1, int N2, int N3, int seed, std::vector<double> &res, bool z2 = false, bool normalizePower = false, bool normalizePerBin = false);

#endif

