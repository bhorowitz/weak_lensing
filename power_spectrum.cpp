#include <vector>
#include <ctime>
#include <complex>
#include <memory>

#include <macros.hpp>
#include <math_constants.hpp>
#include <table_function.hpp>
#include <random.hpp>
#include <matrix_impl.hpp>
#include <power_spectrum.hpp>

#include <fftw3.h>

void
SimplePowerSpectrum::normalize(double d, double corr)
{
    check(d > 0, "");

    const int n = 10000;
    const double L = d * n;
    const int N = n * 2;

    fftw_complex *pk = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2  + 1));
    for(int i = 0; i < N / 2 + 1; ++i)
    {
        const double k = 2 * Math::pi / L * i;
        pk[i][0] = unnormalized(k);
        pk[i][1] = 0;
    }

    std::vector<double> px(N);

    fftw_plan backPlan = fftw_plan_dft_c2r_1d(N, pk, &(px[0]), FFTW_ESTIMATE);
    check(backPlan, "");
    fftw_execute(backPlan);
    fftw_destroy_plan(backPlan);
    fftw_free(pk);

    const int index = N / n;
    const double c = px[index] / L;

    check(c != 0, "");
    norm_ = corr / c;
    check(norm_ > 0, "");
    output_screen1("Simple power spectrum normalization is " << norm_ << std::endl);
}

void
SimplePowerSpectrum::normalize2(double d, double corr)
{
    check(d > 0, "");

    const int n = 1000;
    const double L = d * n;
    const int N = n * 2;

    fftw_complex *pk = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * (N / 2  + 1));
    for(int i = 0; i < N; ++i)
    {
        const int iShifted = (i < (N + 1) / 2 ? i : -(N - i));
        const double k1 = 2 * Math::pi / L * iShifted;
        for(int j = 0; j < N / 2 + 1; ++j)
        {
            const double k2 = 2 * Math::pi / L * j;
            const double k = std::sqrt(k1 * k1 + k2 * k2);
            pk[i * (N / 2 + 1) + j][0] = unnormalized(k);
            pk[i * (N / 2 + 1) + j][1] = 0;
        }
    }

    std::vector<double> px(N * N);

    fftw_plan backPlan = fftw_plan_dft_c2r_2d(N, N, pk, &(px[0]), FFTW_ESTIMATE);
    check(backPlan, "");
    fftw_execute(backPlan);
    fftw_destroy_plan(backPlan);
    fftw_free(pk);

    const int index = N / n;
    const double c = px[index] / (L * L);

    check(c != 0, "");
    norm_ = corr / c;
    check(norm_ > 0, "");
    output_screen1("Simple power spectrum normalization is " << norm_ << std::endl);
}

void discretePowerSpectrum(const Math::RealFunction& p, double L, int N, std::vector<double> *res)
{
    check(L > 0, "");
    check(N > 0, "");

    res->resize(N / 2 + 1);
    for(int i = 0; i < N / 2 + 1; ++i)
    {
        const double k = 2 * Math::pi / L * i;
        res->at(i) = L * p.evaluate(k);
    }
}

void discretePowerSpectrum(const Math::RealFunction& p, double L1, double L2, int N1, int N2, std::vector<double> *res)
{
    check(L1 > 0, "");
    check(N1 > 0, "");
    check(L2 > 0, "");
    check(N2 > 0, "");

    res->resize(N1 * (N2 / 2 + 1));
    for(int i = 0; i < N1; ++i)
    {
        const int iShifted = (i < (N1 + 1) / 2 ? i : -(N1 - i));
        const double k1 = 2 * Math::pi / L1 * iShifted;
        for(int j = 0; j < N2 / 2 + 1; ++j)
        {
            const double k2 = 2 * Math::pi / L2 * j;
            const double k = std::sqrt(k1 * k1 + k2 * k2);
            res->at(i * (N2 / 2 + 1) + j) = L1 * L2 * p.evaluate(k);
        }
    }
}

void discretePowerSpectrum(const Math::RealFunction& p, double L1, double L2, double L3, int N1, int N2, int N3, std::vector<double> *res)
{
    check(L1 > 0, "");
    check(N1 > 0, "");
    check(L2 > 0, "");
    check(N2 > 0, "");
    check(L3 > 0, "");
    check(N3 > 0, "");

    res->resize(N1 * N2 * (N3 / 2 + 1));
    for(int i = 0; i < N1; ++i)
    {
        const int iShifted = (i < (N1 + 1) / 2 ? i : -(N1 - i));
        const double k1 = 2 * Math::pi / L1 * iShifted;
        for(int j = 0; j < N2; ++j)
        {
            const int jShifted = (j < (N2 + 1) / 2 ? j : -(N2 - j));
            const double k2 = 2 * Math::pi / L2 * jShifted;
            for(int k = 0; k < N3 / 2 + 1; ++k)
            {
                const double k3 = 2 * Math::pi / L3 * k;
                const double kAbs = std::sqrt(k1 * k1 + k2 * k2 + k3 * k3);
                res->at((i * N2 + j) * (N3 / 2 + 1) + k) = L1 * L2 * L3 * p.evaluate(k);
            }
        }
    }
}

void discretePowerSpectrum(int N, double L, const std::vector<int> &bins, const std::vector<double> &p, std::vector<double> *res)
{
    check(L > 0, "");
    check(N > 0, "");

    res->resize(N / 2 + 1);
    const int nBins = p.size();

    for(int i = 0; i < N / 2 + 1; ++i)
    {
        int l = bins[i];
        check(l >= 0 && l < nBins, "");
        res->at(i) = L * p[l];
    }
}

void discretePowerSpectrum(int N1, int N2, double L1, double L2, const std::vector<int> &bins, const std::vector<double> &p, std::vector<double> *res)
{
    check(L1 > 0, "");
    check(N1 > 0, "");
    check(L2 > 0, "");
    check(N2 > 0, "");

    res->resize(N1 * (N2 / 2 + 1));
    const int nBins = p.size();

    for(int i = 0; i < N1 * (N2 / 2 + 1); ++i)
    {
        int l = bins[i];
        if(l == -1)
        {
            const int i1 = i / (N2 / 2 + 1);
            const int j1 = i % (N2 / 2 + 1);
            check(j1 == 0 || j1 == N2 / 2, "");
            check(i1 > N1 / 2, "");
            const int i1New = N1 - i1;
            const int iNew = i1New * (N2 / 2 + 1) + j1;
            l = bins[iNew];
        }
        check(l >= 0 && l < nBins, "");
        res->at(i) = L1 * L2 * p[l];
    }
}

void discretePowerSpectrum(int N1, int N2, int N3, double L1, double L2, double L3, const std::vector<int> &bins, const std::vector<double> &p, std::vector<double> *res)
{
    check(L1 > 0, "");
    check(N1 > 0, "");
    check(L2 > 0, "");
    check(N2 > 0, "");
    check(L3 > 0, "");
    check(N3 > 0, "");

    res->resize(N1 * N2 * (N3 / 2 + 1));
    const int nBins = p.size();

    for(int i = 0; i < N1 * N2 * (N3 / 2 + 1); ++i)
    {
        int l = bins[i];
        if(l == -1)
        {
            const int i1 = i / (N3 / 2 + 1) / N2;
            const int j1 = i / (N3 / 2 + 1) % N2;
            const int k1 = i % (N3 / 2 + 1);
            check(k1 == 0 || k1 == N3 / 2, "");
            check(j1 > N2 / 2, "");
            const int j1New = N2 - j1;
            const int i1New = (i1 == 0 ? 0 : N1 - i1);
            const int iNew = (i1New * N2 + j1New) * (N3 / 2 + 1) + k1;
            l = bins[iNew];
        }
        check(l >= 0 && l < nBins, "");
        res->at(i) = L1 * L2 * L3 * p[l];
    }
}

void generateDeltaK(int N, const std::vector<double>& pk, std::vector<std::complex<double> > *deltaK, int seed, std::vector<double> *realBuffer, bool z2, bool normalizePower, bool normalizePerBin)
{
    check(N > 0, "");
    check(pk.size() == N / 2 + 1, "");

    if(seed == 0)
        seed = std::time(0);

    deltaK->resize(N / 2 + 1);

    std::vector<double> *myRealBuffer;
    std::unique_ptr<std::vector<double> > buf;

    if(realBuffer)
        myRealBuffer = realBuffer;
    else
    {
        buf.reset(new std::vector<double>(N));
        myRealBuffer = buf.get();
    }

    myRealBuffer->resize(N);
    generateWhiteNoise(N, seed, *myRealBuffer, z2, normalizePower, normalizePerBin);

    fftw_plan fwdPlan = fftw_plan_dft_r2c_1d(N, &((*myRealBuffer)[0]), reinterpret_cast<fftw_complex*>(&((*deltaK)[0])), FFTW_ESTIMATE);
    check(fwdPlan, "");
    fftw_execute(fwdPlan);
    fftw_destroy_plan(fwdPlan);

    for(int i = 0; i < N / 2 + 1; ++i)
        (*deltaK)[i] *= std::sqrt(pk[i] / N);
}

void generateDeltaK(int N1, int N2, const std::vector<double>& pk, std::vector<std::complex<double> > *deltaK, int seed, std::vector<double> *realBuffer, bool z2, bool normalizePower, bool normalizePerBin)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(pk.size() == N1 * (N2 / 2 + 1), "");

    if(seed == 0)
        seed = std::time(0);

    deltaK->resize(N1 * (N2 / 2 + 1));

    std::vector<double> *myRealBuffer;
    std::unique_ptr<std::vector<double> > buf;

    if(realBuffer)
        myRealBuffer = realBuffer;
    else
    {
        buf.reset(new std::vector<double>(N1 * N2));
        myRealBuffer = buf.get();
    }

    myRealBuffer->resize(N1 * N2);
    generateWhiteNoise(N1, N2, seed, *myRealBuffer, z2, normalizePower, normalizePerBin);

    fftw_plan fwdPlan = fftw_plan_dft_r2c_2d(N1, N2, &((*myRealBuffer)[0]), reinterpret_cast<fftw_complex*>(&((*deltaK)[0])), FFTW_ESTIMATE);
    check(fwdPlan, "");
    fftw_execute(fwdPlan);
    fftw_destroy_plan(fwdPlan);

    for(int i = 0; i < N1; ++i)
        for(int j = 0; j < N2 / 2 + 1; ++j)
            (*deltaK)[i * (N2 / 2 + 1) + j] *= std::sqrt(pk[i * (N2 / 2 + 1) + j] / (N1 * N2));
}

void generateDeltaK(int N1, int N2, int N3, const std::vector<double>& pk, std::vector<std::complex<double> > *deltaK, int seed, std::vector<double> *realBuffer, bool z2, bool normalizePower, bool normalizePerBin)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(pk.size() == N1 * N2 * (N3 / 2 + 1), "");

    if(seed == 0)
        seed = std::time(0);

    deltaK->resize(N1 * N2 * (N3 / 2 + 1));

    std::vector<double> *myRealBuffer;
    std::unique_ptr<std::vector<double> > buf;

    if(realBuffer)
        myRealBuffer = realBuffer;
    else
    {
        buf.reset(new std::vector<double>(N1 * N2 * N3));
        myRealBuffer = buf.get();
    }

    myRealBuffer->resize(N1 * N2 * N3);
    generateWhiteNoise(N1, N2, N3, seed, *myRealBuffer, z2, normalizePower, normalizePerBin);

    fftw_plan fwdPlan = fftw_plan_dft_r2c_3d(N1, N2, N3, &((*myRealBuffer)[0]), reinterpret_cast<fftw_complex*>(&((*deltaK)[0])), FFTW_ESTIMATE);
    check(fwdPlan, "");
    fftw_execute(fwdPlan);
    fftw_destroy_plan(fwdPlan);

    for(int i = 0; i < N1; ++i)
        for(int j = 0; j < N2; ++j)
            for(int k = 0; k < N3 / 2 + 1; ++k)
                (*deltaK)[(i * N2 + j) * (N3 / 2 + 1) + k] *= std::sqrt(pk[(i * N2 + j) * (N3 / 2 + 1) + k] / (N1 * N2 * N3));
}

void deltaK2deltaX(int N, const std::vector<std::complex<double> >& deltaK, std::vector<double> *deltaX, double L)
{
    check(N > 0, "");
    check(deltaK.size() == N / 2 + 1, "");

    deltaX->resize(N);

    // by passing FFTW_PRESERVE_INPUT to fftw this const_cast should be fine
    std::complex<double> *begin = const_cast<std::complex<double>*>(&deltaK[0]);
    fftw_plan backPlan = fftw_plan_dft_c2r_1d(N, reinterpret_cast<fftw_complex*>(begin), &((*deltaX)[0]), FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    check(backPlan, "");
    fftw_execute(backPlan);
    fftw_destroy_plan(backPlan);

    for(int i = 0; i < N; ++i)
        (*deltaX)[i] /= L;
}

void deltaX2deltaK(int N, const std::vector<double>& deltaX, std::vector<std::complex<double> > *deltaK, double L)
{
    check(N > 0, "");
    check(deltaX.size() == N, "");
    deltaK->resize(N / 2 + 1);

    // by passing FFTW_PRESERVE_INPUT to fftw this const_cast should be fine
    double* begin = const_cast<double*>(&(deltaX[0]));

    fftw_plan fwdPlan = fftw_plan_dft_r2c_1d(N, begin, reinterpret_cast<fftw_complex*>(&((*deltaK)[0])), FFTW_PRESERVE_INPUT | FFTW_ESTIMATE);
    check(fwdPlan, "");
    fftw_execute(fwdPlan);
    fftw_destroy_plan(fwdPlan);

    const double factor = L / N;
    for(int i = 0; i < N / 2 + 1; ++i)
        deltaK->at(i) *= factor;
}

void deltaK2deltaX(int N1, int N2, std::vector<std::complex<double> >& deltaK, std::vector<double> *deltaX, double L1, double L2, std::vector<std::complex<double> >* buffer, bool preserveInput)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(deltaK.size() == N1 * (N2 / 2 + 1), "");

    deltaX->resize(N1 * N2);

    std::vector<std::complex<double> > *buf;
    std::unique_ptr<std::vector<std::complex<double> > > myBuf;
    if(preserveInput)
    {
        if(buffer)
            buf = buffer;
        else
        {
            myBuf.reset(new std::vector<std::complex<double> >());
            buf = myBuf.get();
        }

        *buf = deltaK;
    }
    else
        buf = &deltaK;

    fftw_plan backPlan = fftw_plan_dft_c2r_2d(N1, N2, reinterpret_cast<fftw_complex*>(&((*buf)[0])), &((*deltaX)[0]), FFTW_ESTIMATE);
    check(backPlan, "");
    fftw_execute(backPlan);
    fftw_destroy_plan(backPlan);

    for(int i = 0; i < N1; ++i)
        for(int j = 0; j < N2; ++j)
            (*deltaX)[i * N2 + j] /= (L1 * L2);
}

void deltaX2deltaK(int N1, int N2, const std::vector<double>& deltaX, std::vector<std::complex<double> > *deltaK, double L1, double L2)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(deltaX.size() == N1 * N2, "");
    deltaK->resize(N1 * (N2 / 2 + 1));

    // by passing FFTW_PRESERVE_INPUT to fftw this const_cast should be fine
    double* begin = const_cast<double*>(&(deltaX[0]));

    fftw_plan fwdPlan = fftw_plan_dft_r2c_2d(N1, N2, begin, reinterpret_cast<fftw_complex*>(&((*deltaK)[0])), FFTW_PRESERVE_INPUT | FFTW_ESTIMATE);
    check(fwdPlan, "");
    fftw_execute(fwdPlan);
    fftw_destroy_plan(fwdPlan);

    const double factor = L1 * L2 / (N1 * N2);
    for(int i = 0; i < N1; ++i)
        for(int j = 0; j < N2 / 2 + 1; ++j)
            deltaK->at(i * (N2 / 2 + 1) + j) *= factor;
}

void deltaK2deltaX(int N1, int N2, int N3, std::vector<std::complex<double> >& deltaK, std::vector<double> *deltaX, double L1, double L2, double L3, std::vector<std::complex<double> >* buffer, bool preserveInput)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(deltaK.size() == N1 * N2 * (N3 / 2 + 1), "");

    deltaX->resize(N1 * N2 * N3);

    std::vector<std::complex<double> > *buf;
    std::unique_ptr<std::vector<std::complex<double> > > myBuf;
    if(preserveInput)
    {
        if(buffer)
            buf = buffer;
        else
        {
            myBuf.reset(new std::vector<std::complex<double> >());
            buf = myBuf.get();
        }

        *buf = deltaK;
    }
    else
        buf = &deltaK;

    fftw_plan backPlan = fftw_plan_dft_c2r_3d(N1, N2, N3, reinterpret_cast<fftw_complex*>(&((*buf)[0])), &((*deltaX)[0]), FFTW_ESTIMATE);
    check(backPlan, "");
    fftw_execute(backPlan);
    fftw_destroy_plan(backPlan);

    for(int i = 0; i < N1; ++i)
        for(int j = 0; j < N2; ++j)
            for(int k = 0; k < N3; ++k)
                (*deltaX)[(i * N2 + j) * N3 + k] /= (L1 * L2 * L3);
}

void deltaX2deltaK(int N1, int N2, int N3, const std::vector<double>& deltaX, std::vector<std::complex<double> > *deltaK, double L1, double L2, double L3)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(deltaX.size() == N1 * N2 * N3, "");
    deltaK->resize(N1 * N2 * (N3 / 2 + 1));

    // by passing FFTW_PRESERVE_INPUT to fftw this const_cast should be fine
    double* begin = const_cast<double*>(&(deltaX[0]));

    fftw_plan fwdPlan = fftw_plan_dft_r2c_3d(N1, N2, N3, begin, reinterpret_cast<fftw_complex*>(&((*deltaK)[0])), FFTW_PRESERVE_INPUT | FFTW_ESTIMATE);
    check(fwdPlan, "");
    fftw_execute(fwdPlan);
    fftw_destroy_plan(fwdPlan);

    const double factor = L1 * L2 * L3 / (N1 * N2 * N3);
    for(int i = 0; i < N1; ++i)
        for(int j = 0; j < N2; ++j)
            for(int k = 0; k < N3 / 2 + 1; ++k)
                deltaK->at((i * N2 + j) * (N3 / 2 + 1) + k) *= factor;
}

double priorK(int N, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK)
{
    check(N > 0, "");
    check(pk.size() == N / 2 + 1, "");
    check(deltaK.size() == N / 2 + 1, "");

    double res = 0;

#ifdef CHECKS_ON
    const double epsilon = 1e-5;
#endif

    // the first and last element should be real, everything in between is complex
    for(int i = 1; i < N / 2; ++i)
    {
        const double a = std::abs(deltaK[i]);
        check(pk[i] > 0, "");
        res += a * a / pk[i];
    }

    // mutliply by 2 for conjugate k values
    res *= 2;

    check(std::abs(std::imag(deltaK[0])) < epsilon, "");
    double a = std::abs(deltaK[0]);
    check(pk[0] > 0, "");
    res += a * a / pk[0];

    check(std::abs(std::imag(deltaK.back())) < epsilon, "");
    a = std::abs(deltaK.back());
    check(pk.back() > 0, "");
    res += a * a / pk.back();

    return res;
}

double priorK(int N1, int N2, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(pk.size() == N1 * (N2 / 2 + 1), "");
    check(deltaK.size() == N1 * (N2 / 2 + 1), "");

    double res = 0;

    for(int j = 1; j < N2 / 2; ++j)
    {
        for(int i = 0; i < N1; ++i)
        {
            const double a = std::abs(deltaK[i * (N2 / 2 + 1) + j]);
            check(pk[i * (N2 / 2 + 1) + j] > 0, "")
            res += a * a / pk[i * (N2 / 2 + 1) + j];
        }
    }

    // mutliply by 2 for conjugate k values
    res *= 2;

    for(int i = 0; i < N1; ++i)
    {
        double a = std::abs(deltaK[i * (N2 / 2 + 1)]);
        check(pk[i * (N2 / 2 + 1)] > 0, "")
        res += a * a / pk[i * (N2 / 2 + 1)];

        a = std::abs(deltaK[i * (N2 / 2 + 1) + N2 / 2]);
        check(pk[i * (N2 / 2 + 1) + N2 / 2] > 0, "");
        res += a * a / pk[i * (N2 / 2 + 1) + N2 / 2];
    }

    return res;
}

double priorK(int N1, int N2, int N3, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(pk.size() == N1 * N2 * (N3 / 2 + 1), "");
    check(deltaK.size() == N1 * N2 * (N3 / 2 + 1), "");

    double res = 0;

    for(int k = 1; k < N3 / 2; ++k)
    {
        for(int i = 0; i < N1; ++i)
        {
            for(int j = 0; j < N2; ++j)
            {
                const int index = (i * N2 + j) * (N3 / 2 + 1) + k;
                const double a = std::abs(deltaK[index]);
                check(pk[index] > 0, "")
                res += a * a / pk[index];
            }
        }
    }

    // mutliply by 2 for conjugate k values
    res *= 2;

    for(int i = 0; i < N1; ++i)
    {
        for(int j = 0; j < N2; ++j)
        {
            const int index = (i * N2 + j) * (N3 / 2 + 1);
            double a = std::abs(deltaK[index]);
            check(pk[index] > 0, "")
            res += a * a / pk[index];

            a = std::abs(deltaK[index + N3 / 2]);
            check(pk[index + N3 / 2] > 0, "");
            res += a * a / pk[index + N3 / 2];
        }
    }

    return res;
}

void priorKDeriv(int N, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK, std::vector<std::complex<double> > *res)
{
    check(N > 0, "");
    check(pk.size() == N / 2 + 1, "");
    check(deltaK.size() == N / 2 + 1, "");

    res->resize(N / 2 + 1);

#ifdef CHECKS_ON
    const double epsilon = 1e-10;
#endif

    for(int i = 0; i < N / 2 + 1; ++i)
    {
        // the first and last element should be real, everything in between is complex
        check(std::abs(std::imag(deltaK[i])) < epsilon || (i > 0 && i < N / 2), "");
        check(pk[i] > 0, "");
        res->at(i) = (i == 0 || i == N / 2 ? 1.0 : 2.0) * 2.0 * deltaK[i] / pk[i];
    }
}

void priorKDeriv(int N1, int N2, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK, std::vector<std::complex<double> > *res)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(pk.size() == N1 * (N2 / 2 + 1), "");
    check(deltaK.size() == N1 * (N2 / 2 + 1), "");

    res->resize(N1 * (N2 / 2 + 1));

#ifdef CHECKS_ON
    const double epsilon = 1e-10;
#endif

    for(int i = 0; i < N1; ++i)
    {
        for(int j = 0; j < N2 / 2 + 1; ++j)
        {
            const int index = i * (N2 / 2 + 1) + j;
            check(pk[index] > 0, "");
            res->at(index) = 2.0 * 2.0 * deltaK[index] / pk[index]; // the first factor of two is for conjugates
        }
    }

    // elements that don't have conjugate (i.e the real ones) should only count once
    res->at(0) /= 2; // (0, 0)
    res->at(N1 / 2 * (N2 / 2 + 1)) /= 2; // (N1/2, 0)
    res->at(N2 / 2) /= 2; // (0, N2/2)
    res->at(N1 /2 * (N2 / 2 + 1) + N2 / 2) /= 2; // (N1/2, N2/2)
}

void priorKDeriv(int N1, int N2, int N3, const std::vector<double>& pk, const std::vector<std::complex<double> >& deltaK, std::vector<std::complex<double> > *res)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(pk.size() == N1 * N2 * (N3 / 2 + 1), "");
    check(deltaK.size() == N1 * N2 * (N3 / 2 + 1), "");

    res->resize(N1 * N2 * (N3 / 2 + 1));

    for(int i = 0; i < N1; ++i)
    {
        for(int j = 0; j < N2; ++j)
        {
            for(int k = 0; k < N3 / 2 + 1; ++k)
            {
                const int index = (i * N2 + j) * (N3 / 2 + 1) + k;
                check(pk[index] > 0, "");
                res->at(index) = 2.0 * 2.0 * deltaK[index] / pk[index]; // the first factor of two is for conjugates
            }
        }
    }

    // elements that don't have conjugate (i.e the real ones) should only count once
    res->at(0) /= 2; // (0, 0, 0)
    res->at(N1 / 2 * N2 * (N3 / 2 + 1)) /= 2; // (N1/2, 0, 0)
    res->at(N2 / 2 * (N3 / 2 + 1)) /= 2; // (0, N2/2, 0)
    res->at((N1 / 2 * N2 + N2 / 2) * (N3 / 2 + 1)) /= 2; // (N1/2, N2/2, 0)

    res->at(N3 / 2) /= 2; // (0, 0, N3/2)
    res->at(N1 / 2 * N2 * (N3 / 2 + 1) + N3 / 2) /= 2; // (N1/2, 0, N3/2)
    res->at(N2 / 2 * (N3 / 2 + 1) + N3 / 2) /= 2; // (0, N2/2, N3/2)
    res->at((N1 / 2 * N2 + N2 / 2) * (N3 / 2 + 1) + N3 / 2) /= 2; // (N1/2, N2/2, N3/2)
}

double priorX(int N, const std::vector<double>& pk, const std::vector<double>& deltaX, double L, std::vector<std::complex<double> > *deltaK)
{
    check(N > 0, "");
    check(deltaX.size() == N, "");
    check(pk.size() == N / 2 + 1, "");

    std::vector<std::complex<double> >* myC = deltaK;
    std::unique_ptr<std::vector<std::complex<double> > > cPtr;
    if(!myC)
    {
        cPtr.reset(new std::vector<std::complex<double> >());
        myC = cPtr.get();
    }

    deltaX2deltaK(N, deltaX, myC, L);
    return priorK(N, pk, *myC);
}

double priorX(int N1, int N2, const std::vector<double>& pk, const std::vector<double>& deltaX, double L1, double L2, std::vector<std::complex<double> > *deltaK)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(deltaX.size() == N1 * N2, "");
    check(pk.size() == N1 * (N2 / 2 + 1), "");

    std::vector<std::complex<double> >* myC = deltaK;
    std::unique_ptr<std::vector<std::complex<double> > > cPtr;
    if(!myC)
    {
        cPtr.reset(new std::vector<std::complex<double> >());
        myC = cPtr.get();
    }

    deltaX2deltaK(N1, N2, deltaX, myC, L1, L2);
    return priorK(N1, N2, pk, *myC);
}

double priorX(int N1, int N2, int N3, const std::vector<double>& pk, const std::vector<double>& deltaX, double L1, double L2, double L3, std::vector<std::complex<double> > *deltaK)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(deltaX.size() == N1 * N2 * N3, "");
    check(pk.size() == N1 * N2 * (N3 / 2 + 1), "");

    std::vector<std::complex<double> >* myC = deltaK;
    std::unique_ptr<std::vector<std::complex<double> > > cPtr;
    if(!myC)
    {
        cPtr.reset(new std::vector<std::complex<double> >());
        myC = cPtr.get();
    }

    deltaX2deltaK(N1, N2, N3, deltaX, myC, L1, L2, L3);
    return priorK(N1, N2, N3, pk, *myC);
}

void priorXDeriv(int N, const std::vector<double>& pk, const std::vector<double>& deltaX, double L, std::vector<double>* res, std::vector<std::complex<double> > *deltaK, std::vector<std::complex<double> > *pkDeriv)
{
    check(N > 0, "");
    check(pk.size() == N / 2 + 1, "");
    check(deltaX.size() == N, "");

    std::vector<std::complex<double> >* myC = deltaK;
    std::unique_ptr<std::vector<std::complex<double> > > cPtr;
    if(!myC)
    {
        cPtr.reset(new std::vector<std::complex<double> >());
        myC = cPtr.get();
    }

    std::vector<std::complex<double> >* myDerivK = pkDeriv;
    std::unique_ptr<std::vector<std::complex<double> > > derivPtr;
    if(!myDerivK)
    {
        derivPtr.reset(new std::vector<std::complex<double> >());
        myDerivK = derivPtr.get();
    }

    deltaX2deltaK(N, deltaX, myC, L);
    priorKDeriv(N, pk, *myC, myDerivK);
    // the ones that have conjugates divide by 2
    for(int i = 1; i < myDerivK->size() - 1; ++i)
        myDerivK->at(i) /= 2.0;
    deltaK2deltaX(N, *myDerivK, res, L);
    for(auto it = res->begin(); it != res->end(); ++it)
        (*it) *= (L * L / N);
}

void priorXDeriv(int N1, int N2, const std::vector<double>& pk, const std::vector<double>& deltaX, double L1, double L2, std::vector<double>* res, std::vector<std::complex<double> > *deltaK, std::vector<std::complex<double> > *pkDeriv)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(pk.size() == N1 * (N2 / 2 + 1), "");
    check(deltaX.size() == N1 * N2, "");

    std::vector<std::complex<double> >* myC = deltaK;
    std::unique_ptr<std::vector<std::complex<double> > > cPtr;
    if(!myC)
    {
        cPtr.reset(new std::vector<std::complex<double> >());
        myC = cPtr.get();
    }

    std::vector<std::complex<double> >* myDerivK = pkDeriv;
    std::unique_ptr<std::vector<std::complex<double> > > derivPtr;
    if(!myDerivK)
    {
        derivPtr.reset(new std::vector<std::complex<double> >());
        myDerivK = derivPtr.get();
    }

    deltaX2deltaK(N1, N2, deltaX, myC, L1, L2);
    priorKDeriv(N1, N2, pk, *myC, myDerivK);
    // the ones that have conjugates divide by 2
    for(int i = 0; i < N1; ++i)
    {
        for(int j = 0; j < N2 / 2 + 1; ++j)
        {
            if((j == 0 || j == N2 / 2) && (i == 0 || i == N1 / 2))
                continue;
            myDerivK->at(i * (N2 / 2 + 1) + j) /= 2.0;
        }
    }
    deltaK2deltaX(N1, N2, *myDerivK, res, L1, L2);
    for(auto it = res->begin(); it != res->end(); ++it)
        (*it) *= (L1 * L1 * L2 * L2 / (N1 * N2));
}

void priorXDeriv(int N1, int N2, int N3, const std::vector<double>& pk, const std::vector<double>& deltaX, double L1, double L2, double L3, std::vector<double>* res, std::vector<std::complex<double> > *deltaK, std::vector<std::complex<double> > *pkDeriv)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(pk.size() == N1 * N2 * (N3 / 2 + 1), "");
    check(deltaX.size() == N1 * N2 * N3, "");

    std::vector<std::complex<double> >* myC = deltaK;
    std::unique_ptr<std::vector<std::complex<double> > > cPtr;
    if(!myC)
    {
        cPtr.reset(new std::vector<std::complex<double> >());
        myC = cPtr.get();
    }

    std::vector<std::complex<double> >* myDerivK = pkDeriv;
    std::unique_ptr<std::vector<std::complex<double> > > derivPtr;
    if(!myDerivK)
    {
        derivPtr.reset(new std::vector<std::complex<double> >());
        myDerivK = derivPtr.get();
    }

    deltaX2deltaK(N1, N2, N3, deltaX, myC, L1, L2, L3);
    priorKDeriv(N1, N2, N3, pk, *myC, myDerivK);
    // the ones that have conjugates divide by 2
    for(int i = 0; i < N1; ++i)
    {
        for(int j = 0; j < N2; ++j)
        {
            for(int k = 0; k < N3 / 2 + 1; ++k)
            {
                if((k == 0 || k == N3 / 2) && (j == 0 || j == N2 / 2) && (i == 0 || i == N1 / 2))
                    continue;
                myDerivK->at((i * N2 + j) * (N3 / 2 + 1) + k) /= 2.0;
            }
        }
    }
    deltaK2deltaX(N1, N2, N3, *myDerivK, res, L1, L2, L3);
    for(auto it = res->begin(); it != res->end(); ++it)
        (*it) *= (L1 * L1 * L1 * L2 * L2 * L2 / (N1 * N2 * N3));
}

void covarianceMatrix(int N, const std::vector<double>& pk, double L, Math::SymmetricMatrix<double> *covMat, std::vector<std::complex<double> >* complexBuf, std::vector<double> *realBuf)
{
    check(N > 0, "");
    check(pk.size() == N / 2 + 1, "");
    check(L > 0, "");

    std::vector<std::complex<double> > *cBuf;
    std::vector<double> * buf;
    std::unique_ptr<std::vector<std::complex<double> > > myCBuf;
    std::unique_ptr<std::vector<double> > myBuf;

    if(complexBuf)
        cBuf = complexBuf;
    else
    {
        myCBuf.reset(new std::vector<std::complex<double> >);
        cBuf = myCBuf.get();
    }
    cBuf->resize(N / 2 + 1);

    if(realBuf)
        buf = realBuf;
    else
    {
        myBuf.reset(new std::vector<double>);
        buf = myBuf.get();
    }
    buf->resize(N);

    for(int i = 0; i < N / 2 + 1; ++i)
        cBuf->at(i) = pk[i];

    fftw_plan backPlan = fftw_plan_dft_c2r_1d(N, reinterpret_cast<fftw_complex*>(&((*cBuf)[0])), &((*buf)[0]), FFTW_ESTIMATE);
    check(backPlan, "");
    fftw_execute(backPlan);
    fftw_destroy_plan(backPlan);

    covMat->resize(N, N);
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            // since the matrix is symmetric
            if(j > i)
                continue;

            const int dx = std::abs(i - j);
            (*covMat)(i, j) = buf->at(dx) / (L * L);
        }
    }
}

void covarianceMatrix(int N1, int N2, const std::vector<double>& pk, double L1, double L2, Math::SymmetricMatrix<double> *covMat, std::vector<std::complex<double> >* complexBuf, std::vector<double> *realBuf)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(pk.size() == N1 * (N2 / 2 + 1), "");
    check(L1 > 0, "");
    check(L2 > 0, "");

    std::vector<std::complex<double> > *cBuf;
    std::vector<double> * buf;
    std::unique_ptr<std::vector<std::complex<double> > > myCBuf;
    std::unique_ptr<std::vector<double> > myBuf;

    if(complexBuf)
        cBuf = complexBuf;
    else
    {
        myCBuf.reset(new std::vector<std::complex<double> >);
        cBuf = myCBuf.get();
    }
    cBuf->resize(N1 * (N2 / 2 + 1));

    if(realBuf)
        buf = realBuf;
    else
    {
        myBuf.reset(new std::vector<double>);
        buf = myBuf.get();
    }
    buf->resize(N1 * N2);

    for(int i = 0; i < N1 * (N2 / 2 + 1); ++i)
        cBuf->at(i) = pk[i];

    fftw_plan backPlan = fftw_plan_dft_c2r_2d(N1, N2, reinterpret_cast<fftw_complex*>(&((*cBuf)[0])), &((*buf)[0]), FFTW_ESTIMATE);
    check(backPlan, "");
    fftw_execute(backPlan);
    fftw_destroy_plan(backPlan);

    covMat->resize(N1 * N2, N1 * N2);
    for(int i = 0; i < N1; ++i)
    {
        for(int j = 0; j < N2; ++j)
        {
            const int index1 = i * N2 + j;
            for(int k = 0; k < N1; ++k)
            {
                for(int l = 0; l < N2; ++l)
                {
                    const int index2 = k * N2 + l;

                    // since the matrix is symmetric
                    if(index2 > index1)
                        continue;

                    const int dx = std::abs(i - k);
                    const int dy = std::abs(j - l);
                    (*covMat)(index1, index2) = buf->at(dx * N2 + dy) / (L1 * L1 * L2 * L2);
                }
            }
        }
    }

    for(int i = 0; i < N1; ++i)
    {
        const int i1 = (i == 0 ? 0 : N1 - i);
        for(int j = 0; j < N2 / 2 + 1; ++j)
        {
            const int index = i * (N2 / 2 + 1) + j;
            const int index1 = i1 * (N2 / 2 + 1) + j;
            cBuf->at(index) = pk[index1];
        }
    }

    fftw_plan backPlan1 = fftw_plan_dft_c2r_2d(N1, N2, reinterpret_cast<fftw_complex*>(&((*cBuf)[0])), &((*buf)[0]), FFTW_ESTIMATE);
    check(backPlan1, "");
    fftw_execute(backPlan1);
    fftw_destroy_plan(backPlan1);

    for(int i = 0; i < N1; ++i)
    {
        for(int j = 0; j < N2; ++j)
        {
            const int index1 = i * N2 + j;
            for(int k = 0; k < N1; ++k)
            {
                for(int l = 0; l < N2; ++l)
                {
                    const int index2 = k * N2 + l;

                    // since the matrix is symmetric
                    if(index2 > index1)
                        continue;

                    if((i > k && j > l) || (i < k && j < l))
                        continue;

                    const int dx = std::abs(i - k);
                    const int dy = std::abs(j - l);
                    (*covMat)(index1, index2) = buf->at(dx * N2 + dy) / (L1 * L1 * L2 * L2);
                }
            }
        }
    }
}

void covarianceMatrix(int N1, int N2, int N3, const std::vector<double>& pk, double L1, double L2, double L3, Math::SymmetricMatrix<double> *covMat, std::vector<std::complex<double> >* complexBuf, std::vector<double> *realBuf)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(pk.size() == N1 * N2 * (N3 / 2 + 1), "");
    check(L1 > 0, "");
    check(L2 > 0, "");
    check(L3 > 0, "");

    std::vector<std::complex<double> > *cBuf;
    std::vector<double> * buf;
    std::unique_ptr<std::vector<std::complex<double> > > myCBuf;
    std::unique_ptr<std::vector<double> > myBuf;

    if(complexBuf)
        cBuf = complexBuf;
    else
    {
        myCBuf.reset(new std::vector<std::complex<double> >);
        cBuf = myCBuf.get();
    }
    cBuf->resize(N1 * N2 * (N3 / 2 + 1));

    if(realBuf)
        buf = realBuf;
    else
    {
        myBuf.reset(new std::vector<double>);
        buf = myBuf.get();
    }
    buf->resize(N1 * N2 * N3);

    for(int i = 0; i < N1 * N2 * (N3 / 2 + 1); ++i)
        cBuf->at(i) = pk[i];

    fftw_plan backPlan = fftw_plan_dft_c2r_3d(N1, N2, N3, reinterpret_cast<fftw_complex*>(&((*cBuf)[0])), &((*buf)[0]), FFTW_ESTIMATE);
    check(backPlan, "");
    fftw_execute(backPlan);
    fftw_destroy_plan(backPlan);

    covMat->resize(N1 * N2 * N3, N1 * N2 * N3);
    for(int i = 0; i < N1; ++i)
    {
        for(int j = 0; j < N2; ++j)
        {
            for(int k = 0; k < N3; ++k)
            {
                const int index1 = (i * N2 + j) * N3 + k;
                for(int l = 0; l < N1; ++l)
                {
                    for(int m = 0; m < N2; ++m)
                    {
                        for(int n = 0; n < N3; ++n)
                        {
                            const int index2 = (l * N2 + m) * N3 + n;

                            // since the matrix is symmetric
                            if(index2 > index1)
                                continue;

                            const int dx = std::abs(i - l);
                            const int dy = std::abs(j - m);
                            const int dz = std::abs(k - n);
                            (*covMat)(index1, index2) = buf->at((dx * N2 + dy) * N3 + dz) / (L1 * L1 * L2 * L2 * L3 * L3);
                        }
                    }
                }
            }
        }
    }

    // it is assumed that pk depends on the amplitude of k ONLY
}

void power(int N, double L, const std::vector<std::complex<double> > deltaK, std::vector<double> *kVals, std::vector<double> *pVals)
{
    check(N > 0, "");
    check(L > 0, "");
    check(deltaK.size() == N / 2 + 1, "");

    std::vector<int> bins(N / 2 + 1);
    const int nBins = powerSpectrumBins(N, L, &bins, kVals);
    check(nBins > 0, "");
    check(kVals->size() == nBins, "");

    std::vector<double> &pk = *pVals;
    pk.resize(nBins);
    for(int i = 0; i < nBins; ++i)
        pk[i] = 0;

    std::vector<int> numK(nBins, 0);
    for(int i = 0; i < N / 2 + 1; ++i)
    {
        const int bin = bins[i];
        if(bin == -1)
            continue;

        check(bin >= 0 && bin < nBins, "");

        const std::complex<double> d = deltaK[i];
        pk[bin] += std::real(d) * std::real(d) + std::imag(d) * std::imag(d);
        ++numK[bin];
    }
    
    for(int i = 0; i < nBins; ++i)
        pk[i] /= (numK[i] * L);
}

void power(int N1, int N2, double L1, double L2, const std::vector<std::complex<double> > deltaK, std::vector<double> *kVals, std::vector<double> *pVals)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(L1 > 0, "");
    check(L2 > 0, "");
    check(deltaK.size() == N1 * (N2 / 2 + 1), "");

    std::vector<int> bins(N1 * (N2 / 2 + 1));
    const int nBins = powerSpectrumBins(N1, N2, L1, L2, &bins, kVals);
    check(nBins > 0, "");
    check(kVals->size() == nBins, "");

    std::vector<double> &pk = *pVals;
    pk.resize(nBins);
    for(int i = 0; i < nBins; ++i)
        pk[i] = 0;

    std::vector<int> numK(nBins, 0);
    for(int i = 0; i < N1 * (N2 / 2 + 1); ++i)
    {
        const int bin = bins[i];
        if(bin == -1)
            continue;

        check(bin >= 0 && bin < nBins, "");

        const std::complex<double> d = deltaK[i];
        pk[bin] += std::real(d) * std::real(d) + std::imag(d) * std::imag(d);
        ++numK[bin];
    }
    
    for(int i = 0; i < nBins; ++i)
        pk[i] /= (numK[i] * L1 * L2);
}

void power(int N1, int N2, int N3, double L1, double L2, double L3, const std::vector<std::complex<double> > deltaK, std::vector<double> *kVals, std::vector<double> *pVals)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(L1 > 0, "");
    check(L2 > 0, "");
    check(L3 > 0, "");
    check(deltaK.size() == N1 * N2 * (N3 / 2 + 1), "");

    std::vector<int> bins(N1 * N2 * (N3 / 2 + 1));
    const int nBins = powerSpectrumBins(N1, N2, N3, L1, L2, L3, &bins, kVals);
    check(nBins > 0, "");
    check(kVals->size() == nBins, "");

    std::vector<double> &pk = *pVals;
    pk.resize(nBins);
    for(int i = 0; i < nBins; ++i)
        pk[i] = 0;

    std::vector<int> numK(nBins, 0);
    for(int i = 0; i < N1 * N2 * (N3 / 2 + 1); ++i)
    {
        const int bin = bins[i];
        if(bin == -1)
            continue;

        check(bin >= 0 && bin < nBins, "");

        const std::complex<double> d = deltaK[i];
        pk[bin] += std::real(d) * std::real(d) + std::imag(d) * std::imag(d);
        ++numK[bin];
    }
    
    for(int i = 0; i < nBins; ++i)
        pk[i] /= (numK[i] * L1 * L2 * L3);
}

int powerSpectrumBins(int N, double L, std::vector<int> *bins, std::vector<double> *kBins)
{
    check(N > 0, "");
    check(L > 0, "");

    bins->resize(N / 2 + 1);

    const double kMin = 2 * Math::pi / L;
    const double binWidthMax = 2 * kMin; 
    const double kMax = 2 * Math::pi / L * N / 2;
    const int nBins = int(std::ceil(kMax / binWidthMax));
    const double binWidth = kMax / nBins;
    
    for(int i = 0; i < N / 2 + 1; ++i)
    {
        const double k = 2 * Math::pi / L * i;
        check(k <= kMax * 1.00001, "");
        int bin = std::floor(k / kMax * nBins);
        if(bin == nBins)
            --bin;
        check(bin >= 0 && bin < nBins, "");
        bins->at(i) = bin;
    }

    if(kBins)
    {
        kBins->resize(nBins);
        for(int i = 0; i < nBins; ++i)
            kBins->at(i) = binWidth / 2 + i * binWidth;
    }

    return nBins;
}

int powerSpectrumBins(int N1, int N2, double L1, double L2, std::vector<int> *bins, std::vector<double> *kBins)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(L1 > 0, "");
    check(L2 > 0, "");

    bins->resize(N1 * (N2 / 2 + 1));

    const double maxL = std::max(L1, L2);
    const double kMin = 2 * Math::pi / maxL;
    const double binWidthMax = 2 * kMin; 
    const double kMax1 = 2 * Math::pi / L1 * N1 / 2;
    const double kMax2 = 2 * Math::pi / L2 * N2 / 2;
    const double kMax = std::sqrt(kMax1 * kMax1 + kMax2 * kMax2);
    const int nBins = int(std::ceil(kMax / binWidthMax));
    const double binWidth = kMax / nBins;
    
    for(int i = 0; i < N1; ++i)
    {
        const int iShifted = (i < (N1 + 1) / 2 ? i : -(N1 - i));
        const double k1 = 2 * Math::pi / L1 * iShifted;
        for(int j = 0; j < N2 / 2 + 1; ++j)
        {
            const int index = i * (N2 / 2 + 1) + j;
            if((j == 0 || j == N2 / 2) && i > N1 / 2)
            {
                bins->at(index) = -1;
                continue;
            }

            const double k2 = 2 * Math::pi / L2 * j;
            const double k = std::sqrt(k1 * k1 + k2 * k2);
            check(k <= kMax * 1.00001, "");
            int bin = std::floor(k / kMax * nBins);
            if(bin == nBins)
                --bin;
            check(bin >= 0 && bin < nBins, "");
            bins->at(index) = bin;
        }
    }

    if(kBins)
    {
        kBins->resize(nBins);
        for(int i = 0; i < nBins; ++i)
            kBins->at(i) = binWidth / 2 + i * binWidth;
    }

    return nBins;
}

int powerSpectrumBins(int N1, int N2, int N3, double L1, double L2, double L3, std::vector<int> *bins, std::vector<double> *kBins)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(L1 > 0, "");
    check(L2 > 0, "");
    check(L3 > 0, "");

    bins->resize(N1 * N2 * (N3 / 2 + 1));

    const double maxL = std::max(std::max(L1, L2), L3);
    const double kMin = 2 * Math::pi / maxL;
    const double binWidthMax = 2 * kMin; 
    const double kMax1 = 2 * Math::pi / L1 * N1 / 2;
    const double kMax2 = 2 * Math::pi / L2 * N2 / 2;
    const double kMax3 = 2 * Math::pi / L3 * N3 / 2;
    const double kMax = std::sqrt(kMax1 * kMax1 + kMax2 * kMax2 + kMax3 * kMax3);
    const int nBins = int(std::ceil(kMax / binWidthMax));
    const double binWidth = kMax / nBins;
    
    for(int i = 0; i < N1; ++i)
    {
        const int iShifted = (i < (N1 + 1) / 2 ? i : -(N1 - i));
        const double k1 = 2 * Math::pi / L1 * iShifted;
        for(int j = 0; j < N2; ++j)
        {
            const int jShifted = (j < (N2 + 1) / 2 ? j : -(N2 - j));
            const double k2 = 2 * Math::pi / L2 * jShifted;
            for(int k = 0; k < N3 / 2 + 1; ++k)
            {
                const int index = (i * N2 + j) * (N3 / 2 + 1) + k;
                if((k == 0 || k == N3 / 2) && j > N2 / 2)
                {
                    bins->at(index) = -1;
                    continue;
                }

                const double k3 = 2 * Math::pi / L3 * k;
                const double kAbs = std::sqrt(k1 * k1 + k2 * k2 + k3 * k3);
                check(kAbs <= kMax * 1.00001, "");
                int bin = std::floor(kAbs / kMax * nBins);
                if(bin == nBins)
                    --bin;
                check(bin >= 0 && bin < nBins, "");
                bins->at(index) = bin;
            }
        }
    }

    if(kBins)
    {
        kBins->resize(nBins);
        for(int i = 0; i < nBins; ++i)
            kBins->at(i) = binWidth / 2 + i * binWidth;
    }

    return nBins;
}

int powerSpectrumBinsFull(int N, double L, std::vector<int> *bins, std::vector<double> *kBins)
{
    check(N > 0, "");
    check(L > 0, "");

    bins->resize(N);

    const double kMin = 2 * Math::pi / L;
    const double binWidthMax = 2 * kMin; 
    const double kMax = 2 * Math::pi / L * N / 2;
    const int nBins = int(std::ceil(kMax / binWidthMax));
    const double binWidth = kMax / nBins;
    
    for(int i = 0; i < N; ++i)
    {
        const int iShifted = (i < (N + 1) / 2 ? i : -(N - i));
        const double k = 2 * Math::pi / L * iShifted;
        check(k <= kMax * 1.00001, "");
        int bin = std::floor(k / kMax * nBins);
        if(bin == nBins)
            --bin;
        check(bin >= 0 && bin < nBins, "");
        bins->at(i) = bin;
    }

    if(kBins)
    {
        kBins->resize(nBins);
        for(int i = 0; i < nBins; ++i)
            kBins->at(i) = binWidth / 2 + i * binWidth;
    }

    return nBins;
}

int powerSpectrumBinsFull(int N1, int N2, double L1, double L2, std::vector<int> *bins, std::vector<double> *kBins)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(L1 > 0, "");
    check(L2 > 0, "");

    bins->resize(N1 * N2);

    const double maxL = std::max(L1, L2);
    const double kMin = 2 * Math::pi / maxL;
    const double binWidthMax = 2 * kMin; 
    const double kMax1 = 2 * Math::pi / L1 * N1 / 2;
    const double kMax2 = 2 * Math::pi / L2 * N2 / 2;
    const double kMax = std::sqrt(kMax1 * kMax1 + kMax2 * kMax2);
    const int nBins = int(std::ceil(kMax / binWidthMax));
    const double binWidth = kMax / nBins;
    
    for(int i = 0; i < N1; ++i)
    {
        const int iShifted = (i < (N1 + 1) / 2 ? i : -(N1 - i));
        const double k1 = 2 * Math::pi / L1 * iShifted;
        for(int j = 0; j < N2; ++j)
        {
            const int jShifted = (j < (N2 + 1) / 2 ? j : -(N2 - j));
            const int index = i * N2 + j;
            const double k2 = 2 * Math::pi / L2 * jShifted;
            const double k = std::sqrt(k1 * k1 + k2 * k2);
            check(k <= kMax * 1.00001, "");
            int bin = std::floor(k / kMax * nBins);
            if(bin == nBins)
                --bin;
            check(bin >= 0 && bin < nBins, "");
            bins->at(index) = bin;
        }
    }

    if(kBins)
    {
        kBins->resize(nBins);
        for(int i = 0; i < nBins; ++i)
            kBins->at(i) = binWidth / 2 + i * binWidth;
    }

    return nBins;
}

int powerSpectrumBinsFull(int N1, int N2, int N3, double L1, double L2, double L3, std::vector<int> *bins, std::vector<double> *kBins)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");
    check(L1 > 0, "");
    check(L2 > 0, "");
    check(L3 > 0, "");

    bins->resize(N1 * N2 * N3);

    const double maxL = std::max(std::max(L1, L2), L3);
    const double kMin = 2 * Math::pi / maxL;
    const double binWidthMax = 2 * kMin; 
    const double kMax1 = 2 * Math::pi / L1 * N1 / 2;
    const double kMax2 = 2 * Math::pi / L2 * N2 / 2;
    const double kMax3 = 2 * Math::pi / L3 * N3 / 2;
    const double kMax = std::sqrt(kMax1 * kMax1 + kMax2 * kMax2 + kMax3 * kMax3);
    const int nBins = int(std::ceil(kMax / binWidthMax));
    const double binWidth = kMax / nBins;
    
    for(int i = 0; i < N1; ++i)
    {
        const int iShifted = (i < (N1 + 1) / 2 ? i : -(N1 - i));
        const double k1 = 2 * Math::pi / L1 * iShifted;
        for(int j = 0; j < N2; ++j)
        {
            const int jShifted = (j < (N2 + 1) / 2 ? j : -(N2 - j));
            const double k2 = 2 * Math::pi / L2 * jShifted;
            for(int k = 0; k < N3; ++k)
            {
                const int kShifted = (k < (N3 + 1) / 2 ? k : -(N3 - k));
                const double k3 = 2 * Math::pi / L3 * kShifted;
                const int index = (i * N2 + j) * N3 + k;
                const double kAbs = std::sqrt(k1 * k1 + k2 * k2 + k3 * k3);
                check(k <= kMax * 1.00001,"");
                int bin = std::floor(k / kMax * nBins);
                if(bin == nBins)
                    --bin;
                check(bin >= 0 && bin < nBins, "");
                bins->at(index) = bin;
            }
        }
    }

    if(kBins)
    {
        kBins->resize(nBins);
        for(int i = 0; i < nBins; ++i)
            kBins->at(i) = binWidth / 2 + i * binWidth;
    }

    return nBins;
}

void generateWhiteNoise(int N, int seed, std::vector<double> &res, bool z2, bool normalizePower, bool normalizePerBin)
{
    check(N > 0, "");

    res.resize(N);
    Math::GaussianGenerator noiseGenNew(seed, 0, 1);
    for(int i = 0; i < N; ++i)
    {
        if(z2)
            res[i] = (noiseGenNew.generate() > 0 ? 1.0 : -1.0);
        else
            res[i] = noiseGenNew.generate();
    }

    if(!normalizePower)
        return;

    std::vector<std::complex<double> > resK(N / 2 + 1);
    deltaX2deltaK(N, res, &resK, 1.0);

    std::vector<int> bins;
    std::vector<double> kBinVals;
    const int nBins = powerSpectrumBins(N, 1.0, &bins, &kBinVals);
    check(bins.size() == N / 2 + 1, "");
    check(kBinVals.size() == nBins, "");

    std::vector<int> binsFull;
    const int nBinsFull = powerSpectrumBinsFull(N, 1.0, &binsFull);
    check(nBinsFull == nBins, "");

    std::vector<double> binPowers(nBins, 0);
    std::vector<int> count(nBins, 0);
    for(int i = 0; i < N / 2 + 1; ++i)
    {
        const int l = bins[i];
        if(l == -1)
            continue;
        binPowers[l] += std::abs(resK[i]) * std::abs(resK[i]);
        ++count[l];
    }

    for(int l = 0; l < nBins; ++l)
        binPowers[l] /= count[l];

    for(int i = 0; i < N / 2 + 1; ++i)
    {
        int thisBin = bins[i];
        if(thisBin == -1)
        {
            check(i > N / 2, "");
            const int iNew = N - i;
            thisBin = bins[iNew];
        }
        check(thisBin >= 0 && thisBin <= nBins, "");

        if(normalizePerBin)
        {
            resK[i] /= std::sqrt(binPowers[thisBin]);
            resK[i] /= std::sqrt(double(N));
        }
        else
        {
            const double thisPower = std::abs(resK[i]) * std::abs(resK[i]);
            if(thisPower != 0)
            {
                resK[i] /= std::sqrt(thisPower);
                resK[i] /= std::sqrt(double(N));
            }
        }
    }

    deltaK2deltaX(N, resK, &res, 1.0);
}

void generateWhiteNoise(int N1, int N2, int seed, std::vector<double> &res, bool z2, bool normalizePower, bool normalizePerBin)
{
    check(N1 > 0, "");
    check(N2 > 0, "");

    res.resize(N1 * N2);
    Math::GaussianGenerator noiseGenNew(seed, 0, 1);
    for(int i = 0; i < N1 * N2; ++i)
    {
        if(z2)
            res[i] = (noiseGenNew.generate() > 0 ? 1.0 : -1.0);
        else
            res[i] = noiseGenNew.generate();
    }

    if(!normalizePower)
        return;

    std::vector<std::complex<double> > resK(N1 * (N2 / 2 + 1));
    deltaX2deltaK(N1, N2, res, &resK, 1.0, 1.0);

    std::vector<int> bins;
    std::vector<double> kBinVals;
    const int nBins = powerSpectrumBins(N1, N2, 1.0, 1.0, &bins, &kBinVals);
    check(bins.size() == N1 * (N2 / 2 + 1), "");
    check(kBinVals.size() == nBins, "");

    std::vector<int> binsFull;
    const int nBinsFull = powerSpectrumBinsFull(N1, N2, 1.0, 1.0, &binsFull);
    check(nBinsFull == nBins, "");

    std::vector<double> binPowers(nBins, 0);
    std::vector<int> count(nBins, 0);
    for(int i = 0; i < N1 * (N2 / 2 + 1); ++i)
    {
        const int l = bins[i];
        if(l == -1)
            continue;
        binPowers[l] += std::abs(resK[i]) * std::abs(resK[i]);
        ++count[l];
    }

    for(int l = 0; l < nBins; ++l)
        binPowers[l] /= count[l];

    for(int i = 0; i < N1 * (N2 / 2 + 1); ++i)
    {
        int thisBin = bins[i];
        if(thisBin == -1)
        {
            const int i1 = i / (N2 / 2 + 1);
            const int j1 = i % (N2 / 2 + 1);
            check(j1 == 0 || j1 == N2 / 2, "");
            check(i1 > N1 / 2, "");
            const int i1New = N1 - i1;
            const int iNew = i1New * (N2 / 2 + 1) + j1;
            thisBin = bins[iNew];
        }
        check(thisBin >= 0 && thisBin <= nBins, "");

        if(normalizePerBin)
        {
            resK[i] /= std::sqrt(binPowers[thisBin]);
            resK[i] /= std::sqrt(double(N1 * N2));
        }
        else
        {
            const double thisPower = std::abs(resK[i]) * std::abs(resK[i]);
            if(thisPower != 0)
            {
                resK[i] /= std::sqrt(thisPower);
                resK[i] /= std::sqrt(double(N1 * N2));
            }
        }
    }

    deltaK2deltaX(N1, N2, resK, &res, 1.0, 1.0, NULL, true);
}

void generateWhiteNoise(int N1, int N2, int N3, int seed, std::vector<double> &res, bool z2, bool normalizePower, bool normalizePerBin)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(N3 > 0, "");

    res.resize(N1 * N2 * N3);
    Math::GaussianGenerator noiseGenNew(seed, 0, 1);
    for(int i = 0; i < N1 * N2 * N3; ++i)
    {
        if(z2)
            res[i] = (noiseGenNew.generate() > 0 ? 1.0 : -1.0);
        else
            res[i] = noiseGenNew.generate();
    }

    if(!normalizePower)
        return;

    std::vector<std::complex<double> > resK(N1 * N2 * (N3 / 2 + 1));
    deltaX2deltaK(N1, N2, N3, res, &resK, 1.0, 1.0, 1.0);

    std::vector<int> bins;
    std::vector<double> kBinVals;
    const int nBins = powerSpectrumBins(N1, N2, N3, 1.0, 1.0, 1.0, &bins, &kBinVals);
    check(bins.size() == N1 * N2 * (N3 / 2 + 1), "");
    check(kBinVals.size() == nBins, "");

    std::vector<int> binsFull;
    const int nBinsFull = powerSpectrumBinsFull(N1, N2, N3, 1.0, 1.0, 1.0, &binsFull);
    check(nBinsFull == nBins, "");

    std::vector<double> binPowers(nBins, 0);
    std::vector<int> count(nBins, 0);
    for(int i = 0; i < N1 * N2 * (N3 / 2 + 1); ++i)
    {
        const int l = bins[i];
        if(l == -1)
            continue;
        binPowers[l] += std::abs(resK[i]) * std::abs(resK[i]);
        ++count[l];
    }

    for(int l = 0; l < nBins; ++l)
        binPowers[l] /= count[l];

    for(int i = 0; i < N1 * N2 * (N3 / 2 + 1); ++i)
    {
        int thisBin = bins[i];
        if(thisBin == -1)
        {
            const int i1 = i / (N3 / 2 + 1) / N2;
            const int j1 = i / (N3 / 2 + 1) % N2;
            const int k1 = i % (N3 / 2 + 1);
            check(k1 == 0 || k1 == N3 / 2, "");
            check(j1 > N2 / 2, "");
            const int j1New = N2 - j1;
            const int i1New = (i1 == 0 ? 0 : N1 - i1);
            const int iNew = (i1New * N2 + j1New) * (N3 / 2 + 1) + k1;
            thisBin = bins[iNew];
        }
        check(thisBin >= 0 && thisBin <= nBins, "");

        if(normalizePerBin)
        {
            resK[i] /= std::sqrt(binPowers[thisBin]);
            resK[i] /= std::sqrt(double(N1 * N2 * N3));
        }
        else
        {
            const double thisPower = std::abs(resK[i]) * std::abs(resK[i]);
            if(thisPower != 0)
            {
                resK[i] /= std::sqrt(thisPower);
                resK[i] /= std::sqrt(double(N1 * N2 * N3));
            }
        }
    }

    deltaK2deltaX(N1, N2, N3, resK, &res, 1.0, 1.0, 1.0, NULL, true);
}

