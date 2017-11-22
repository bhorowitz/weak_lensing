#include <vector>
#include <ctime>
#include <cmath>
#include <complex>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <sstream>
#include <iomanip>

#include <macros.hpp>
#include <exception_handler.hpp>
#include <table_function.hpp>
#include <lbfgs_general.hpp>
#include <conjugate_gradient_general.hpp>
#include <hmc_general.hpp>
#include <random.hpp>
#include <timer.hpp>
#include <math_constants.hpp>
#include <numerics.hpp>
#include <matrix_impl.hpp>
#include <cosmo_mpi.hpp>
#include <parser.hpp>

#include "power_spectrum.hpp"
#include "utils.hpp"
#include "lin_alg.hpp"

#include <fftw3.h>

namespace
{

class DeltaKVector2
{
public:
    DeltaKVector2(int N1, int N2, int nExtra = 0) : N1_(N1), N2_(N2), c_(N1 * (N2 / 2 + 1)), q_(nExtra)
    {
        check(N1 > 0, "");
        check(N2 > 0, "");
        check(nExtra >= 0, "");
    }

    DeltaKVector2(const DeltaKVector2& other) : N1_(other.N1_), N2_(other.N2_), c_(other.c_), q_(other.q_)
    {
    }

    std::vector<std::complex<double> >& get() { return c_; }
    const std::vector<std::complex<double> >& get() const { return c_; }

    std::vector<double>& getExtra() { return q_; }
    const std::vector<double>& getExtra() const { return q_; }

    int getN1() const { return N1_; }
    int getN2() const { return N2_; }

    // copy from other, multiplying with coefficient (for MPI, the correct coefficient should be passed for EVERY process)
    void copy(const DeltaKVector2& other, double c = 1.)
    {
        check(N1_ == other.N1_, "");
        check(N2_ == other.N2_, "");
        for(int i = 0; i < c_.size(); ++i)
            c_[i] = c * other.c_[i];

        q_ = other.q_;
        for(auto it = q_.begin(); it != q_.end(); ++it)
            (*it) *= c;
    }

    // set all the elements to 0
    void setToZero()
    {
        for(auto it = c_.begin(); it != c_.end(); ++it)
            *it = std::complex<double>(0, 0);

        for(auto it = q_.begin(); it != q_.end(); ++it)
            *it = 0;
    }

    // get the norm (for MPI, ALL the processes should get the norm)
    double norm() const
    {
        return std::sqrt(dotProduct(*this));
    }

    // dot product with another vector (for MPI, ALL the processes should get the dot product)
    double dotProduct(const DeltaKVector2& other) const
    {
        check(N1_ == other.N1_, "");
        check(N2_ == other.N2_, "");
        check(q_.size() == other.q_.size(), "");
        double res = 0;
        for(int j = 0; j < N2_ / 2 + 1; ++j)
        {
            const int iMax = (j > 0 && j < N2_ / 2 ? N1_ : N1_ / 2 + 1);
            for(int i = 0; i < iMax; ++i)
            {
                const int index = i * (N2_ / 2 + 1) + j;
                res += (std::real(c_[index]) * std::real(other.c_[index]) + std::imag(c_[index]) * std::imag(other.c_[index]));
            }
        }

        for(int i = 0; i < q_.size(); ++i)
            res += q_[i] * other.q_[i];
        return res;
    }

    // add another vector with a given coefficient (for MPI, the correct coefficient should be passed for EVERY process)
    void add(const DeltaKVector2& other, double c = 1.)
    {
        check(N1_ == other.N1_, "");
        check(N2_ == other.N2_, "");
        check(q_.size() == other.q_.size(), "");
        for(int i = 0; i < c_.size(); ++i)
            c_[i] += c * other.c_[i];

        for(int i = 0; i < q_.size(); ++i)
            q_[i] += c * other.q_[i];
    }

    // multiply with another vector TERM BY TERM
    void multiply(const DeltaKVector2& other)
    {
        check(N1_ == other.N1_, "");
        check(N2_ == other.N2_, "");
        check(q_.size() == other.q_.size(), "");
        for(int i = 0; i < c_.size(); ++i)
            c_[i] = std::complex<double>(std::real(c_[i]) * std::real(other.c_[i]), std::imag(c_[i]) * std::imag(other.c_[i]));

        for(int i = 0; i < q_.size(); ++i)
            q_[i] *= other.q_[i];
    }

    // divide by another vector TERM BY TERM
    void divide(const DeltaKVector2& other)
    {
        check(N1_ == other.N1_, "");
        check(N2_ == other.N2_, "");
        check(q_.size() == other.q_.size(), "");
        for(int i = 0; i < c_.size(); ++i)
        {
            check(std::real(other.c_[i]) != 0, "");
            const double re = std::real(c_[i]) / std::real(other.c_[i]);
            double im = 0;
            if(std::imag(c_[i]) != 0)
            {
                check(std::imag(other.c_[i]) != 0, "");
                im = std::imag(c_[i]) / std::imag(other.c_[i]);
            }
            c_[i] = std::complex<double>(re, im);
        }

        for(int i = 0; i < q_.size(); ++i)
            q_[i] *= other.q_[i];
    }
    // take power of elements TERM BY TERM
    void pow(double p)
    {
        for(int i = 0; i < c_.size(); ++i)
        {
            const double re = std::pow(std::real(c_[i]), p);
            const double im = std::pow(std::imag(c_[i]), p);
            c_[i] = std::complex<double>(re, im);
        }

        for(int i = 0; i < q_.size(); ++i)
            q_[i] = std::pow(q_[i], p);
    }

    // swap
    void swap(DeltaKVector2& other)
    {
        check(N1_ == other.N1_, "");
        check(N2_ == other.N2_, "");
        check(q_.size() == other.q_.size(), "");
        c_.swap(other.c_);
        q_.swap(other.q_);
    }

private:
    const int N1_, N2_;
    std::vector<std::complex<double> > c_;
    std::vector<double> q_;
};

class DeltaKVector2Factory
{
public:
    DeltaKVector2Factory(int N1, int N2, int nExtra = 0) : N1_(N1), N2_(N2), nExtra_(nExtra)
    {
        check(N1_ > 0, "");
        check(N2_ > 0, "");
        check(nExtra_ >= 0, "");
    }

    // create a new LargeVector with 0 elements
    // the user is in charge of deleting it
    DeltaKVector2* giveMeOne()
    {
        return new DeltaKVector2(N1_, N2_, nExtra_);
    }

private:
    const int N1_;
    const int N2_;
    const int nExtra_;
};

class DeltaK2Func
{
public:
    DeltaK2Func(int N1, int N2, const std::vector<double>& data, const std::vector<double>& sigma, const std::vector<double>& mask, const std::vector<double>& pk, double L1, double L2, bool weakLensing = false, const std::vector<double> *sigma2 = NULL, const std::vector<double> *dataGamma1 = NULL, const std::vector<double> *dataGamma2 = NULL) : N1_(N1), N2_(N2), sigma_(sigma), mask_(mask), pk_(pk), deltaK_(N1 * (N2 / 2 + 1)), derivK_(N1 * (N2 / 2 + 1)), buf_(N1 * (N2 / 2 + 1)), deltaX_(N1 * N2), derivX_(N1 * N2), L1_(L1), L2_(L2), gammaK1_(N1 * (N2 / 2 + 1)), gammaK2_(N1 * (N2 / 2 + 1)), weakLensing_(weakLensing), dataK_(N1 * (N2 / 2 + 1)), dataGamma1_(N1 * N2), dataGamma2_(N1 * N2), sigma2_(sigma2 ? *sigma2 : sigma)
    {
        check(N1_ > 0, "");
        check(N2_ > 0, "");
        check(L1_ > 0, "");
        check(L2_ > 0, "");

        setData(data, dataGamma1, dataGamma2);

        check(data_.size() == N1_ * N2_, "");
        check(sigma_.size() == N1_ * N2_, "");
        check(mask_.size() == N1_ * N2_, "");
        check(pk_.size() == N1_ * (N2_ / 2 + 1), "");
    }

    // generate white noise with given amplitude
    void whitenoise(int seed, DeltaKVector2* x, double amplitude)
    {
        std::vector<std::complex<double> >& v = x->get();
        check(v.size() == N1_ * (N2_ / 2 + 1), "");
        Math::GaussianGenerator g(seed, 0, amplitude);
        for(int i = 0; i < N1_; ++i)
        {
            for(int j = 1; j < N2_ / 2; ++j)
            {
                const double re = g.generate();
                const double im = g.generate();
                v[i * (N2_ / 2 + 1) + j] = std::complex<double>(re, im);
            }
        }

        for(int i = 1; i < N1_ / 2; ++i)
        {
            // j = 0
            double re = g.generate();
            double im = g.generate();
            v[i *(N2_ / 2 + 1)] = std::complex<double>(re, im);
            v[(N1_ - i) * (N2_ / 2 + 1)] = std::complex<double>(re, -im);

            // j = N2_ / 2
            re = g.generate();
            im = g.generate();
            v[i *(N2_ / 2 + 1) + N2_ / 2] = std::complex<double>(re, im);
            v[(N1_ - i) * (N2_ / 2 + 1) + N2_ / 2] = std::complex<double>(re, -im);
        }

        v[0] = g.generate(); // (0, 0)
        v[N1_ / 2 * (N2_ / 2 + 1)] = g.generate(); // (N1_ / 2, 0)
        v[N2_ / 2] = g.generate(); // (0, N2_ / 2)
        v[N1_ / 2 * (N2_ / 2 + 1) + N2_ / 2] = g.generate(); // (N1_ / 2, N2_ / 2)
    }

    void setData(const std::vector<double>& data, const std::vector<double> *dataGamma1 = NULL, const std::vector<double> *dataGamma2 = NULL)
    {
        check(data.size() == N1_ * N2_, "");
        data_ = data;
        if(weakLensing_)
        {
            check(dataGamma1, "");
            check(dataGamma2, "");
            check(dataGamma1->size() == N1_ * N2_, "");
            check(dataGamma2->size() == N1_ * N2_, "");

            dataGamma1_ = *dataGamma1;
            dataGamma2_ = *dataGamma2;
        }
    }

    void set(const DeltaKVector2& x)
    {
        check(x.get().size() == N1_ * (N2_ / 2 + 1), "");
        deltaK_ = x.get();
        deltaK2deltaX(N1_, N2_, deltaK_, &deltaX_, L1_, L2_, &buf_, true);
        if(weakLensing_)
            calculateWeakLensingData(deltaK_, &deltaGamma1_, &deltaGamma2_);

        q_ = x.getExtra();

        check(q_.size() == extra_.size(), "");
    }

    void setExtra(const std::vector<std::vector<double> >& extra, const std::vector<double>& qSigma)
    {
        extra_ = extra;
        qSigma_ = qSigma;
        check(qSigma_.size() == extra_.size(), "");
    }

    // for MPI, ALL the processes should get the function value
    double value()
    {
        double r = 0;
        check(data_.size() == N1_ * N2_, "");
        check(deltaX_.size() == N1_ * N2_, "");
        check(sigma_.size() == N1_ * N2_, "");
        check(mask_.size() == N1_ * N2_, "");
        for(int i = 0; i < N1_ * N2_; ++i)
        {
            const double s = sigma_[i];
            check(s > 0, "");
            if(weakLensing_)
            {
                double e1 = 0, e2 = 0;
                for(int j = 0; j < q_.size(); ++j)
                {
                    check(extra_[j].size() == 2 * N1_ * N2_, "");
                    e1 += mask_[i] * q_[j] * extra_[j][i];
                    e2 += mask_[i] * q_[j] * extra_[j][i + N1_ * N2_];
                }
                const double d1 = mask_[i] * (deltaGamma1_[i] + e1 - dataGamma1_[i]);
                const double d2 = mask_[i] * (deltaGamma2_[i] + e2 - dataGamma2_[i]);
                const double s2 = sigma2_[i];
                check(s2 > 0, "");
                r += d1 * d1 / (s * s);
                r += d2 * d2 / (s2 * s2);
            }
            else
            {
                double e = 0;
                for(int j = 0; j < q_.size(); ++j)
                {
                    check(extra_[j].size() == N1_ * N2_, "");
                    e += mask_[i] * q_[j] * extra_[j][i];
                }
                const double d = mask_[i] * (deltaX_[i] + e - data_[i]);
                r += d * d / (s * s);
            }
        }

        for(int i = 0; i < q_.size(); ++i)
        {
            const double s = qSigma_[i];
            check(s > 0, "");
            r += q_[i] * q_[i] / (s * s);
        }

        double total = r;
#ifdef COSMO_MPI
        CosmoMPI::create().reduce(&r, &total, 1, CosmoMPI::DOUBLE, CosmoMPI::SUM);
#endif
        return r + priorK(N1_, N2_, pk_, deltaK_);
    }

    void derivative(DeltaKVector2 *res)
    {
        check(res->getN1() == N1_, "");
        check(res->getN2() == N2_, "");

        priorKDeriv(N1_, N2_, pk_, deltaK_, &(res->get()));

        std::vector<double>& qDeriv = res->getExtra();
        check(qDeriv.size() == extra_.size(), "");

        for(int i = 0; i < qDeriv.size(); ++i)
            qDeriv[i] = 0;

        check(derivX_.size() == N1_ * N2_, "");
        check(data_.size() == N1_ * N2_, "");
        check(deltaX_.size() == N1_ * N2_, "");
        check(sigma_.size() == N1_ * N2_, "");
        check(sigma2_.size() == N1_ * N2_, "");
        check(mask_.size() == N1_ * N2_, "");
        if(weakLensing_)
        {
            // gamma1
            for(int i = 0; i < N1_ * N2_; ++i)
            {
                const double s = sigma_[i];
                check(s > 0, "");
                double e = 0;
                for(int j = 0; j < q_.size(); ++j)
                {
                    check(extra_[j].size() == 2 * N1_ * N2_, "");
                    e += mask_[i] * q_[j] * extra_[j][i];
                }
                const double d = mask_[i] * (deltaGamma1_[i] + e - dataGamma1_[i]);
                derivX_[i] = 2 * d / (s * s);

                for(int j = 0; j < qDeriv.size(); ++j)
                    qDeriv[j] += mask_[i] * derivX_[i] * extra_[j][i];
            }

            check(derivK_.size() == N1_ * (N2_ / 2 + 1), "");
            fftw_plan fwdPlan = fftw_plan_dft_r2c_2d(N1_, N2_, &(derivX_[0]), reinterpret_cast<fftw_complex*>(&(derivK_[0])), FFTW_ESTIMATE);
            check(fwdPlan, "");
            fftw_execute(fwdPlan);
            fftw_destroy_plan(fwdPlan);

            for(int i = 0; i < N1_ * (N2_ / 2 + 1); ++i)
                derivK_[i] *= (2.0 / (L1_ * L2_));

            // elements that don't have conjugate (i.e the real ones) should only count once
            derivK_[0] /= 2; // (0, 0)
            derivK_[N1_ / 2 * (N2_ / 2 + 1)] /= 2; // (N1/2, 0)
            derivK_[N2_ / 2] /= 2; // (0, N2/2)
            derivK_[N1_ /2 * (N2_ / 2 + 1) + N2_ / 2] /= 2; // (N1/2, N2/2)

            check(derivK_.size() == res->get().size(), "");
            for(int i = 0; i < N1_; ++i)
            {
                for(int j = 0; j < N2_ / 2 + 1; ++j)
                {
                    const int index = i * (N2_ / 2 + 1) + j;
                    if(index == 0)
                        continue;

                    double c2 = 0, s2 = 0;
                    getC2S2(N1_, N2_, L1_, L2_, i, j, &c2, &s2);

                    res->get()[index] += derivK_[index] *  c2;
                }
            }

            // gamma2
            for(int i = 0; i < N1_ * N2_; ++i)
            {
                const double s = sigma2_[i];
                check(s > 0, "");
                double e = 0;
                for(int j = 0; j < q_.size(); ++j)
                {
                    check(extra_[j].size() == 2 * N1_ * N2_, "");
                    e += mask_[i] * q_[j] * extra_[j][i + N1_ * N2_];
                }
                const double d = mask_[i] * (deltaGamma2_[i] + e - dataGamma2_[i]);
                derivX_[i] = 2 * d / (s * s);

                for(int j = 0; j < qDeriv.size(); ++j)
                    qDeriv[j] += mask_[i] * derivX_[i] * extra_[j][i + N1_ * N2_];
            }

            check(derivK_.size() == N1_ * (N2_ / 2 + 1), "");
            fftw_plan fwdPlan2 = fftw_plan_dft_r2c_2d(N1_, N2_, &(derivX_[0]), reinterpret_cast<fftw_complex*>(&(derivK_[0])), FFTW_ESTIMATE);
            check(fwdPlan2, "");
            fftw_execute(fwdPlan2);
            fftw_destroy_plan(fwdPlan2);

            for(int i = 0; i < N1_ * (N2_ / 2 + 1); ++i)
                derivK_[i] *= (2.0 / (L1_ * L2_));

            // elements that don't have conjugate (i.e the real ones) should only count once
            derivK_[0] /= 2; // (0, 0)
            derivK_[N1_ / 2 * (N2_ / 2 + 1)] /= 2; // (N1/2, 0)
            derivK_[N2_ / 2] /= 2; // (0, N2/2)
            derivK_[N1_ /2 * (N2_ / 2 + 1) + N2_ / 2] /= 2; // (N1/2, N2/2)

            check(derivK_.size() == res->get().size(), "");
            for(int i = 0; i < N1_; ++i)
            {
                for(int j = 0; j < N2_ / 2 + 1; ++j)
                {
                    const int index = i * (N2_ / 2 + 1) + j;
                    if(index == 0)
                        continue;
                    
                    double c2 = 0, s2 = 0;
                    getC2S2(N1_, N2_, L1_, L2_, i, j, &c2, &s2);

                    res->get()[index] += derivK_[index] * s2;
                }
            }
        }
        else
        {
            for(int i = 0; i < N1_ * N2_; ++i)
            {
                const double s = sigma_[i];
                check(s > 0, "");
                double e = 0;
                for(int j = 0; j < q_.size(); ++j)
                {
                    check(extra_[j].size() == N1_ * N2_, "");
                    e += mask_[i] * q_[j] * extra_[j][i];
                }
                const double d = mask_[i] * (deltaX_[i] + e - data_[i]);
                derivX_[i] = 2 * d / (s * s);

                for(int j = 0; j < qDeriv.size(); ++j)
                    qDeriv[j] += mask_[i] * derivX_[i] * extra_[j][i];
            }

            check(derivK_.size() == N1_ * (N2_ / 2 + 1), "");
            fftw_plan fwdPlan = fftw_plan_dft_r2c_2d(N1_, N2_, &(derivX_[0]), reinterpret_cast<fftw_complex*>(&(derivK_[0])), FFTW_ESTIMATE);
            check(fwdPlan, "");
            fftw_execute(fwdPlan);
            fftw_destroy_plan(fwdPlan);

            for(int i = 0; i < N1_ * (N2_ / 2 + 1); ++i)
                derivK_[i] *= (2.0 / (L1_ * L2_));

            // elements that don't have conjugate (i.e the real ones) should only count once
            derivK_[0] /= 2; // (0, 0)
            derivK_[N1_ / 2 * (N2_ / 2 + 1)] /= 2; // (N1/2, 0)
            derivK_[N2_ / 2] /= 2; // (0, N2/2)
            derivK_[N1_ /2 * (N2_ / 2 + 1) + N2_ / 2] /= 2; // (N1/2, N2/2)

            check(derivK_.size() == res->get().size(), "");
            for(int i = 0; i < derivK_.size(); ++i)
            {
                if(pk_[i] != 0)
                    res->get()[i] += derivK_[i];
            }
        }

        // q prior derivs
        for(int i = 0; i < qDeriv.size(); ++i)
        {
            const int s = qSigma_[i];
            check(s > 0, "");
            qDeriv[i] += 2 * q_[i] / (s * s);
        }
    }

    void calculateWeakLensingData(const std::vector<std::complex<double> > &dK, std::vector<double> *gamma1, std::vector<double> *gamma2)
    {
        check(dK.size() == N1_ * (N2_ / 2 + 1), "");
        for(int i = 0; i < N1_; ++i)
        {
            for(int j = 0; j < N2_ / 2 + 1; ++j)
            {
                const int index = i * (N2_ / 2 + 1) + j;
                if(index == 0)
                {
                    gammaK1_[index] = std::complex<double>(0, 0);
                    gammaK2_[index] = std::complex<double>(0, 0);
                    continue;
                }
                double c2 = 0, s2 = 0;
                getC2S2(N1_, N2_, L1_, L2_, i, j, &c2, &s2);
                gammaK1_[index] = dK[index] * c2;
                gammaK2_[index] = dK[index] * s2;
            }
        }

        deltaK2deltaX(N1_, N2_, gammaK1_, gamma1, L1_, L2_);
        deltaK2deltaX(N1_, N2_, gammaK2_, gamma2, L1_, L2_);
    }

private:
    const int N1_;
    const int N2_;
    const double L1_;
    const double L2_;
    std::vector<double> data_;
    const std::vector<double> sigma_;
    const std::vector<double> sigma2_;
    const std::vector<double> mask_;
    const std::vector<double> pk_;
    std::vector<std::complex<double> > deltaK_;
    std::vector<std::complex<double> > derivK_;
    std::vector<std::complex<double> > buf_;
    std::vector<double> deltaX_;
    std::vector<double> derivX_;

    std::vector<std::complex<double> > gammaK1_, gammaK2_;
    std::vector<std::complex<double> > dataK_;
    std::vector<double> dataGamma1_, dataGamma2_;
    std::vector<double> deltaGamma1_, deltaGamma2_;
    const bool weakLensing_;

    std::vector<std::vector<double> > extra_;
    std::vector<double> q_;
    std::vector<double> qSigma_;
};

bool testFunctionDerivs(DeltaK2Func &func, int N, int nExtra, const std::vector<double>& pk, double epsilon, int testSeed)
{
    output_screen("Testing function derivatives..." << std::endl);
    DeltaKVector2 testK(N, N, nExtra);
    DeltaKVector2 testKPert(N, N, nExtra);

    std::vector<double> realBuffer(N * N);

    generateDeltaK(N, N, pk, &(testK.get()), testSeed, &realBuffer);
    func.set(testK);
    const double val = func.value();
    DeltaKVector2 derivsK(N, N, nExtra);
    func.derivative(&derivsK);

    bool result = true;
    for(int i = 0; i < N; ++i)
    {
        //output_screen("SKIPPING THIS TEST!!!" << std::endl);
        //break;

        for(int j = 0; j < N / 2 + 1; ++j)
        {
            const int index = i * (N / 2 + 1) + j;
            check(pk[index] > 0, "");
            testKPert.copy(testK);
            testKPert.get()[index] += epsilon * std::complex<double>(1.0, 0.0);
            // certain elements have redundancy, i.e. their conjugates are in the array
            if((j == 0 || j == N / 2) && (i != 0 && i != N / 2))
            {
                const int index1 = (N - i) * (N / 2 + 1) + j;
                testKPert.get()[index1] += epsilon * std::complex<double>(1.0, 0.0);
            }
            func.set(testKPert);
            double val2 = func.value();
            double numDeriv = (val2 - val) / epsilon;
            if(!Math::areEqual(numDeriv, std::real(derivsK.get()[index]), 1e-1))
            {
                output_screen("FAIL: index (" << i << ", " << j << ") real part. Numerical derivative = " << numDeriv << ", analytic = " << std::real(derivsK.get()[index]) << "." << std::endl);
                result = false;
            }

            // if it's real
            if((i == 0 && (j == 0 || j == N / 2)) || (i == N / 2 && (j == 0 || j == N / 2)))
                continue;

            testKPert.copy(testK);
            testKPert.get()[index] += epsilon * std::complex<double>(0.0, 1.0);
            if(j == 0 || j == N / 2)
            {
                const int index1 = (N - i) * (N / 2 + 1) + j;
                testKPert.get()[index1] -= epsilon * std::complex<double>(0.0, 1.0);
            }
            func.set(testKPert);
            val2 = func.value();
            numDeriv = (val2 - val) / epsilon;
            if(!Math::areEqual(numDeriv, std::imag(derivsK.get()[index]), 1e-1))
            {
                output_screen("FAIL: index (" << i << ", " << j << ") imaginary part. Numerical derivative = " << numDeriv << ", analytic = " << std::imag(derivsK.get()[index]) << "." << std::endl);
                result = false;
            }
        }
    }

    const double epsilonQ = 1e-4;

    for(int i = 0; i < nExtra; ++i)
    {
        testKPert.copy(testK);
        testKPert.getExtra()[i] += epsilonQ;
        func.set(testKPert);
        const double val2 = func.value();
        const double numDeriv = (val2 - val) / epsilonQ;
        if(!Math::areEqual(numDeriv, derivsK.getExtra()[i], 1e-2))
        {
            output_screen("FAIL: Extra parameter index " << i << ": Numerical derivative = " << numDeriv << ", analytic = " << derivsK.getExtra()[i] << "." << std::endl);
            result = false;
        }
    }
}

class LBFGSCallback
{
public:
    LBFGSCallback(const char *fileName, int N, double L, bool cg = false) : out_(fileName), N_(N), L_(L), cg_(cg)
    {
        if(!out_)
        {
            output_screen("WARNING: the callback function is not able to write into file " << fileName << "!!!" << std::endl);
        }
    }

    ~LBFGSCallback()
    {
        if(out_)
            out_.close();
    }

    void operator()(int iter, double f, double gradNorm, const DeltaKVector2& x, const DeltaKVector2& grad)
    {
        if(!out_)
            return;

        out_ << std::setprecision(10) << iter << "\t" << f << '\t' << gradNorm << std::endl;

        std::vector<double> deltaX;
        std::vector<std::complex<double> > deltaK = x.get();
        deltaK2deltaX(N_, N_, deltaK, &deltaX, L_, L_, NULL, true);
        std::stringstream fileNameStr;
        if(cg_)
            fileNameStr << "cg";
        else
            fileNameStr << "lbfgs";
        fileNameStr << "_lin_iter_" << iter << ".dat";

        std::ofstream out;
        out.open(fileNameStr.str().c_str(), std::ios::out | std::ios::binary);
        StandardException exc;
        if(!out)
        {
            std::string exceptionStr = "Cannot write into file lbfgs_lin_iter_ITERATION.dat";
            exc.set(exceptionStr);
            throw exc;
        }
        out.write(reinterpret_cast<char*>(&(deltaX[0])), N_ * N_ * sizeof(double));
        out.close();
        // Writing out PS
        std::stringstream fileNameStr_ps;

        std::vector<double> ps, psk;
        power(N_, N_, L_, L_, deltaK, &psk, &ps);
        std::stringstream psFileNameStr;
        psFileNameStr << "lbfgs_ps_" << iter << ".txt";
        vector2file(psFileNameStr.str().c_str(), ps);

    }

    void operator()(int iter, double f, double gradNorm, const DeltaKVector2& x, const DeltaKVector2& grad, const DeltaKVector2& z)
    {
        (*this)(iter, f, gradNorm, x, grad);
    }

private:
    const int N_;
    const double L_;
    std::ofstream out_;
    const bool cg_;
};

class HMCCallback
{
public:
    HMCCallback(const char *fileName, int N, double L) : out_(fileName), N_(N), L_(L), iter_(0)
    {
        if(!out_)
        {
            output_screen("WARNING: the callback function is not able to write into file " << fileName << "!!!" << std::endl);
        }
    }

    ~HMCCallback()
    {
        if(out_)
            out_.close();
    }

    void operator()(const DeltaKVector2& x, double like)
    {
        if(!out_)
            return;

        out_ << std::setprecision(10) << iter_ << "\t" << like << '\t' << std::endl;

        std::vector<std::complex<double> > deltaK = x.get();

        if(iter_ <= 100 || iter_ % 100 == 0)
        {
            std::vector<double> deltaX;
            deltaK2deltaX(N_, N_, deltaK, &deltaX, L_, L_, NULL, true);
            std::stringstream fileNameStr;
            fileNameStr << "hmc_lin_iter_" << iter_ << ".dat";

            std::ofstream out;
            out.open(fileNameStr.str().c_str(), std::ios::out | std::ios::binary);
            StandardException exc;
            if(!out)
            {
                std::string exceptionStr = "Cannot write into file hmc_lin_iter_ITERATION.dat";
                exc.set(exceptionStr);
                throw exc;
            }
            out.write(reinterpret_cast<char*>(&(deltaX[0])), N_ * N_ * sizeof(double));
            out.close();
        }

        std::vector<double> ps, psk;
        power(N_, N_, L_, L_, deltaK, &psk, &ps);
        std::stringstream psFileNameStr;
        psFileNameStr << "hmc_ps_" << iter_ << ".txt";
        vector2file(psFileNameStr.str().c_str(), ps);

        ++iter_;
    }

private:
    const int N_;
    const double L_;
    std::ofstream out_;
    int iter_;
};

void downgrade(int N1, int N2, const std::vector<double> &map, int N1New, int N2New, std::vector<double> *mapNew)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(map.size() == N1 * N2, "");
    check(N1New > 0, "");
    check(N2New > 0, "");
    check(N1 >= N1New, "");
    check(N2 >= N2New, "");
    check(N1 % N1New == 0, "");
    check(N2 % N2New == 0, "");

    const int r1 = N1 / N1New;
    const int r2 = N2 / N2New;
    check(r1 >= 1, "");
    check(r2 >= 1, "");

    mapNew->resize(N1New * N2New);
    for(int i = 0; i < N1New; ++i)
    {
        for(int j = 0; j < N2New; ++j)
        {
            double res = 0;
            for(int k = 0; k < r1; ++k)
            {
                for(int l = 0; l < r2; ++l)
                {
                    const int i1 = i * r1 + k;
                    const int j1 = j * r2 + l;
                    res += map[i1 * N2 + j1];
                }
            }

            (*mapNew)[i * N2New + j] = res / (r1 * r2);
        }
    }
}

void fisher2window(Math::Matrix<double>& mat)
{
    for(int l = 0; l < mat.rows(); ++l)
    {
        double s = 0;
        for(int l1 = 0; l1 < mat.cols(); ++l1)
            s += mat(l, l1);

        check(s > 0 || l > 0, "");
        if(s == 0)
        {
            // fill in from previous line
            mat(l, 0) = 0;
            for(int l1 = 1; l1 < mat.cols(); ++l1)
                mat(l, l1) = mat(l - 1, l1 - 1);

            continue;
        }

        for(int l1 = 0; l1 < mat.cols(); ++l1)
            mat(l, l1) /= s;
    }
}

class SimplePowerSpectrum2 : public Math::RealFunction
{
public:
    SimplePowerSpectrum2(double k0 = 0.015, double norm = 1) : k0_(k0), norm_(norm) {}
    ~SimplePowerSpectrum2() {}

    double evaluate(double k) const { return norm_ * unnormalized(k); }

private:
    double unnormalized(double k) const
    {
        check(k >= 0, "");
        const double ratio = k / k0_;
        return k / (1 + ratio * ratio * ratio);
    }

private:
    const double k0_;
    double norm_;
};

} // namespace

int main(int argc, char *argv[])
{
    try{
        StandardException exc;

        if(argc < 2)
        {
            std::string exceptionStr = "Need to specify the parameters file";
            exc.set(exceptionStr);
            throw exc;
        }

        Parser parser(argv[1]);

        // let's first read the data file
        const int NData = 512;
        const double LData = 1380;
        std::vector<float> rhoActualFloat(NData * NData);
        std::ifstream inData("/global/homes/b/bhorowit/lyman_alpha/2dproj0-512x512.f4", std::ios::in | std::ios::binary);
        if(!inData)
        {
            std::string exceptionStr = "Cannot read the data file 2dproj0-512x512.f4";
            exc.set(exceptionStr);
            throw exc;
        }

        inData.read(reinterpret_cast<char*>(&(rhoActualFloat[0])), NData * NData * sizeof(float));
        inData.close();

        std::vector<double> rhoActual(rhoActualFloat.begin(), rhoActualFloat.end());
        check(rhoActual.size() == NData * NData, "");

        const int sumBits = NData;
        std::vector<double> rhoActualMeanVector(sumBits);
        int sumBit = 0;
        double rhoActualMean = 0;
        for(int i = 0; i < NData * NData; ++i)
            rhoActualMeanVector[(sumBit++) % sumBits] += rhoActual[i];

        for(int i = 0; i < sumBits; ++i)
            rhoActualMean += rhoActualMeanVector[i];
        rhoActualMean /= (NData * NData);

        std::vector<double> deltaActual(NData * NData);
        for(int i = 0; i < NData * NData; ++i)
            deltaActual[i] = rhoActual[i] / rhoActualMean - 1;

        vector2binFile("data_actual.dat", deltaActual);

        std::vector<std::complex<double> > deltaKActual(NData * (NData / 2 + 1));
        deltaX2deltaK(NData, NData, deltaActual, &deltaKActual, LData, LData);

        std::vector<double> kValsDataP, pValsDataP;
        power(NData, NData, LData, LData, deltaKActual, &kValsDataP, &pValsDataP);

        Math::TableFunction<double, double> dataPS;
        dataPS[0] = 0;
        const double kMaxActual = 2 * Math::pi / LData * (NData / std::sqrt(2));
        dataPS[kMaxActual] = 0;
        check(kValsDataP.size() == pValsDataP.size(), "");
        for(int i = 0; i < kValsDataP.size(); ++i)
            dataPS[kValsDataP[i]] = pValsDataP[i];
        writePSToFile("data_ps.txt", dataPS, 2 * Math::pi / LData, kMaxActual * 0.99);

        const int N = parser.getInt("N", 64);
        const double L = parser.getDouble("L", LData);
        const int lin_lim = parser.getInt("lin_lim", 64);
        const bool dataIsOriginal = parser.getBool("data_original", false);
        const bool weakLensing = parser.getBool("weak_lensing", false);


        const int simCount = parser.getInt("sim_count", 1);
        const bool z2 = parser.getBool("z2", true);
        const bool normalizePower = parser.getBool("normalize_power", true);
        const bool normalizePerBin = parser.getBool("normalize_per_bin", true);

        const bool gibbs = parser.getBool("gibbs", false);
        const bool varyingNoise = parser.getBool("vary_noise", true);
        const bool maskStuff = parser.getBool("mask", true);
        const bool maskInNoise = parser.getBool("mask_in_noise", false);
        const bool maskFromFile = parser.getBool("mask_from_file", false);

        const bool useSimplePS = parser.getBool("simple_ps", true);

        const bool sparseWindow = parser.getBool("sparse_window", true);
        const bool includeHigherPower = parser.getBool("higher_power", false);
        const double fiducialFactor = parser.getDouble("fiducial_factor", 1.0);
        const bool simulateLargerBox = parser.getBool("larger_box", false);
        double noiseVal = parser.getDouble("noise_val", 0.1);
        bool estimatePowerSpectrum = parser.getBool("estimate_power_spectrum", false);
        const int mLBFGS = parser.getInt("lbfgs_m", 10);
        const bool moreThuenteSearch = parser.getBool("more_thuente", true);
        const bool fisherNew = parser.getBool("fisher_new", true);
        const bool doHMC = parser.getBool("hmc", false);
        const int hmcMaxIters = parser.getInt("hmc_max_iterations", 1000);
        const double hmcMassToPkRatio = parser.getDouble("hmc_mass_pk_ratio", 10.0);
        const double hmcTauMax = parser.getDouble("hmc_tau_max", 1.0);
        const int hmcNMax = parser.getInt("hmc_n_max", 10);
        const double epsilonFactor = parser.getDouble("epsilon_factor", 0.1);

        const bool conjugateGrad = parser.getBool("conjugate_gradient", false);

        const std::string loc = parser.getStr("file_prefix", "");

        parser.dump();

        if(doHMC)
            estimatePowerSpectrum = false;

        SimplePowerSpectrum2 simplePS(0.05, 100);

        const Math::RealFunction *psToUse = &dataPS;
        if(useSimplePS)
            psToUse = &simplePS;

        int deltaLSparse = 2;
        deltaLSparse *= (N < 32 ? 1 : N / 32);

        if(!sparseWindow)
            deltaLSparse = 1;

        const double epsilon = epsilonFactor / (N * N);


        if(weakLensing)
            noiseVal *= 2;

        std::vector<int> bins;
        std::vector<double> kBinVals;
        const int nBins = powerSpectrumBins(N, N, L, L, &bins, &kBinVals);
        check(bins.size() == N * (N / 2 + 1), "");
        check(kBinVals.size() == nBins, "");
        vector2file("k_bins.txt", kBinVals);

        std::vector<int> binsFull;
        const int nBinsFull = powerSpectrumBinsFull(N, N, L, L, &binsFull);
        check(nBinsFull == nBins, "");

        std::vector<double> pBinVals(nBins);
        for(int i = 0; i < nBins; ++i)
            pBinVals[i] = psToUse->evaluate(kBinVals[i]);
        vector2file("actual_ps.txt", pBinVals);

        std::vector<double> pk;
        discretePowerSpectrum(N, N, L, L, bins, pBinVals, &pk);

        std::vector<double> fiducialBinVals = pBinVals;

        output_screen("fiducialFactor = " << fiducialFactor << std::endl);

        for(auto it = fiducialBinVals.begin(); it != fiducialBinVals.end(); ++it)
            *it *= fiducialFactor;

        vector2file("fiducial_ps.txt", fiducialBinVals);

        std::vector<double> pkFiducial;
        discretePowerSpectrum(N, N, L, L, bins, fiducialBinVals, &pkFiducial);

        std::vector<double> realBuffer(N * N);
        std::vector<std::complex<double> > complexBuffer(N * (N / 2 + 1));

        std::vector<std::complex<double> > deltaK;
        int seed = 100;
        generateDeltaK(N, N, pk, &deltaK, seed++, &realBuffer);

        // including power beyond k_max for aliasing
        const int NHigher = N * 2;
        std::vector<int> binsHigher;
        std::vector<double> kBinValsHigher;
        int nBinsHigher;
        std::vector<double> pBinValsHigher;
        std::vector<double> pkHigher;
        std::vector<std::complex<double> > deltaKHigher;
        std::vector<double> deltaXHigher;
        std::vector<double> deltaXLower;
        std::vector<std::complex<double> > deltaKLower;

        if(includeHigherPower)
        {
            nBinsHigher = powerSpectrumBins(NHigher, NHigher, L, L, &binsHigher, &kBinValsHigher);
            pBinValsHigher.resize(nBinsHigher);
            for(int i = 0; i < nBinsHigher; ++i)
            {
                if(kBinValsHigher[i] > kBinVals.back())
                    pBinValsHigher[i] = psToUse->evaluate(kBinValsHigher[i]);
            }
            vector2file("ps_higher.txt", pBinValsHigher);
            discretePowerSpectrum(NHigher, NHigher, L, L, binsHigher, pBinValsHigher, &pkHigher);
            generateDeltaK(NHigher, NHigher, pkHigher, &deltaKHigher, seed++);
            deltaK2deltaX(NHigher, NHigher, deltaKHigher, &deltaXHigher, L, L);
            downgrade(NHigher, NHigher, deltaXHigher, N, N, &deltaXLower);
            vector2binFile("deltax2_higher.dat", deltaXLower);
            deltaX2deltaK(N, N, deltaXLower, &deltaKLower, L, L);
            check(deltaKLower.size() == deltaK.size(), "");

            for(int i = 0; i < deltaK.size(); ++i)
                deltaK[i] += deltaKLower[i];
        }
        std::vector<double> deltaX;
        deltaK2deltaX(N, N, deltaK, &deltaX, L, L, &complexBuffer, true);
        check(deltaX.size() == N * N, "");

        if(N == NData && dataIsOriginal)
        {
            deltaX = deltaActual;
            deltaK = deltaKActual;
        }

        vector2binFile("deltax2.dat", deltaX);
        std::vector<double> deltaPS, deltaPSK;
        power(N, N, L, L, deltaK, &deltaPSK, &deltaPS);
        vector2file("delta_ps.txt", deltaPS);

        std::vector<double> sigmaNoise(N * N, noiseVal);
        // make the noise different in each pixel
        if(varyingNoise)
        {
            for(int i = 0; i < N; ++i)
            {
                for(int j = 0; j < N; ++j)
                {
                    const double f1 = std::sin(2 * Math::pi * double(i) / N);
                    const double f2 = std::cos(4 * Math::pi * double(j) / N);
                    sigmaNoise[i * N + j] *= (f1 * f1 * f2 * f2 + 0.1);
                }
            }
        }

        vector2binFile("sigmax2.dat", sigmaNoise);

        int noiseSeed = 200;
        Math::GaussianGenerator noiseGen(noiseSeed, 0, 1);
        std::vector<double> noiseX(N * N);
        //double noiseXMean = 0;
        for(int i = 0; i < N * N; ++i)
        {
            noiseX[i] = noiseGen.generate() * sigmaNoise[i];
            //noiseXMean += noiseX[i];
        }
        //noiseXMean /= (N * N);
        //output_screen("Noise mean = " << noiseXMean << std::endl);
        /*
        for(int i = 0; i < N * N; ++i)
            noiseX[i] -= noiseXMean;
        */

        std::vector<double> dataX(N * N);
        for(int i = 0; i < N * N; ++i)
            dataX[i] = deltaX[i] + noiseX[i];

        vector2binFile("datax2.dat", dataX);

        std::vector<double> mask(N * N, 1);
        // mask out some stuff
        if(maskStuff)
        {
            for(int i = 0; i < N; ++i)
            {
                for(int j = 0; j < N; ++j)
                {
                    const int di = i - N / 2;
                    const int dj = j - N / 2;
                    if(di * di + dj * dj < (N / 8) * (N / 8))
                        mask[i * N + j] = 0;

                    if(i < N / 10 || i > N - N / 10 || j < N / 10 || j > N - N / 10)
                        if(di * di + dj * dj > 0.9 * (N / 2) * (N / 2))
                            mask[i * N + j] = 0;

                }
            }

            if(maskFromFile)
            {
                std::ifstream in("mask.txt");
                if(!in)
                {
                    StandardException exc;
                    exc.set("Cannot open input file mask.txt");
                    throw exc;
                }
                for(int i = 0; i < N; ++i)
                {
                    std::string s;
                    std::getline(in, s);
                    std::stringstream str(s);
                    for(int j = 0; j < N; ++j)
                    {
                        str >> mask[i * N + j];
                    }
                }
                in.close();
            }
        }

        if(maskInNoise)
        {
            for(int i = 0; i < N * N; ++i)
            {
                if(mask[i] == 0)
                {
                    mask[i] = 1;
                    sigmaNoise[i] = 1.0;
                }
            }
        }


        vector2binFile("mask2.dat", mask);

        std::vector<double> dataGamma1(N * N), dataGamma2(N * N);
        DeltaK2Func func(N, N, dataX, sigmaNoise, mask, pkFiducial, L, L, weakLensing, &sigmaNoise, &dataGamma1, &dataGamma2);
        if(weakLensing)
        {
            func.calculateWeakLensingData(deltaK, &dataGamma1, &dataGamma2);
            vector2binFile("data_gamma1.dat", dataGamma1);
            vector2binFile("data_gamma2.dat", dataGamma2);
            for(int i = 0; i < N * N; ++i)
            {
                dataGamma1[i] += noiseGen.generate() * sigmaNoise[i];
                dataGamma2[i] += noiseGen.generate() * sigmaNoise[i];
            }
            vector2binFile("data_gamma1_noisy.dat", dataGamma1);
            vector2binFile("data_gamma2_noisy.dat", dataGamma2);
            func.setData(dataX, &dataGamma1, &dataGamma2);
        }

        const int nExtra = 0;
        std::vector<std::vector<double> > extraData(nExtra);
        for(int i = 0; i < nExtra; ++i)
            extraData[i].resize((weakLensing ? 2 : 1) * N * N, 1e-5);


        /*
        for(int i = 0; i < N / 2; ++i)
        {
            for(int j = 0; j < N / 2; ++j)
                extraData[0][i * N + j] = 1e-3;
        }
        */

        /*
        int nExtra = 0;
        std::vector<std::vector<double> > extraData;
        for(int i = 0; i < N / 4; ++i)
        {
            for(int j = 0; j < N / 4; ++j)
            {
                ++nExtra;
                std::vector<double> e(N * N);
                e[i * N + j] = 1e-3;
                extraData.push_back(e);
            }
        }
        */

        /*
        const int binToRemove = 2;
        int nExtra = 0;
        std::vector<std::vector<double> > extraData;
        for(int i = 0; i < N * (N / 2 + 1); ++i)
        {
            if(bins[i] == binToRemove)
            {
                ++nExtra;
                std::vector<std::complex<double> > myDeltaK(N * (N / 2 + 1));
                myDeltaK[i] = std::complex<double>(1.0, 0.0) * 1e2;
                std::vector<double> myDeltaX;
                deltaK2deltaX(N, N, myDeltaK, &myDeltaX, L, L, &complexBuffer, true);
                extraData.push_back(myDeltaX);

                //output_screen("VALUE: " << myDeltaX[0] << ' ' << myDeltaX[1] << ' ' << myDeltaX[100] << std::endl);

                ++nExtra;
                myDeltaK[i] = std::complex<double>(0.0, 1.0) * 1e2;
                deltaK2deltaX(N, N, myDeltaK, &myDeltaX, L, L, &complexBuffer, true);
                extraData.push_back(myDeltaX);
            }
        }
        */

        std::vector<double> qSigma(nExtra, 1e6);

        DeltaKVector2Factory factory(N, N, nExtra);

        func.setExtra(extraData, qSigma);

        if(N <= lin_lim)
            testFunctionDerivs(func, N, nExtra, pkFiducial, 1e-1, 300);

        // let's do lbfgs!
        DeltaKVector2 starting(N, N, nExtra);
        starting.setToZero();
        if(doHMC)
        {
            const double tauMax = hmcTauMax;
            const int nMax = hmcNMax;
            DeltaKVector2 masses(N, N, nExtra);
            for(int i = 0; i < N; ++i)
            {
                for(int j = 0; j < N / 2 + 1; ++j)
                {
                    const int index = i * (N / 2 + 1) + j;
                    bool hasImag = true;
                    if((j == 0 || j == N / 2) && (i == 0 || i == N / 2))
                        hasImag = false;

                    const double re = hmcMassToPkRatio / pkFiducial[index];
                    const double im = (hasImag ? re : 0.0);

                    masses.get()[index] = std::complex<double>(re, im);
                }
            }
            for(int i = 0; i < nExtra; ++i)
                masses.getExtra()[i] = 1.0; // TBD

            Math::HMCGeneral<DeltaKVector2, DeltaKVector2Factory, DeltaK2Func> hmc(&factory, &func, starting, masses, tauMax, nMax);
            HMCCallback cb("hmc_chi2.txt", N, L);
            hmc.run(hmcMaxIters, &cb);

            return 0;
        }

        Math::LBFGS_General<DeltaKVector2, DeltaKVector2Factory, DeltaK2Func> lbfgsOrig(&factory, &func, starting, mLBFGS, moreThuenteSearch);
        Math::LBFGS_General<DeltaKVector2, DeltaKVector2Factory, DeltaK2Func> lbfgs(&factory, &func, starting, mLBFGS, moreThuenteSearch);
        DeltaKVector2 deltaKMin(N, N, nExtra);

        const double gTol = 1e-20;
        if(conjugateGrad)
        {
            Math::CG_General<DeltaKVector2, DeltaKVector2Factory, DeltaK2Func> cg(&factory, &func, starting);
            LBFGSCallback cb("cg_chi2.txt", N, L);
            cg.minimize(&deltaKMin, epsilon, gTol, 10000000, Math::CG_General<DeltaKVector2, DeltaKVector2Factory, DeltaK2Func>::FLETCHER_REEVES, &cb);
            
        }
        else
        {
            LBFGSCallback cb("chi2.txt", N, L);
            lbfgsOrig.minimize(&deltaKMin, epsilon, gTol, 10000000, &cb);
        }

        std::vector<double> deltaXMin;
        deltaK2deltaX(N, N, deltaKMin.get(), &deltaXMin, L, L, &complexBuffer, true);

        vector2binFile("deltax2min.dat", deltaXMin);

        std::vector<double> deltaMinPS, deltaMinPSK;
        power(N, N, L, L, deltaKMin.get(), &deltaMinPSK, &deltaMinPS);
        vector2file("delta_min_ps.txt", deltaMinPS);

        if(deltaKMin.getExtra().size() > 0)
            vector2file("q.txt", deltaKMin.getExtra());
        
        if(!estimatePowerSpectrum)
            return 0;

        output_screen("Now trying to estimate the noise bias!" << std::endl);
        std::vector<int> binCount(nBins, 0);
        for(int i = 0; i < N * N; ++i)
        {
            const int l = binsFull[i];
            check(l >= 0 && l < nBins, "");
            ++binCount[l];
        }
        vector2file("bin_count.txt", binCount);

        std::vector<int> binCountHalf(nBins, 0);
        for(int i = 0; i < N * (N / 2 + 1); ++i)
        {
            const int l = bins[i];
            if(l < 0)
                continue;

            check(l >= 0 && l < nBins, "");
            ++binCountHalf[l];
        }

        noiseSeed = 20000;

        std::vector<double> noiseXNew(N * N), noiseXNew1(N * N);
        DeltaKVector2 gradNoise(N, N, nExtra);
        std::vector<double> bEstimated(nBins, 0);
        std::vector<std::complex<double> > noiseKNew(N * (N / 2 + 1));
        std::vector<double> noiseAvg(N * N);

        std::vector<std::vector<std::complex<double> > > noiseRealizations;

        std::vector<std::complex<double> > deltaKSum(N * (N / 2 + 1));
        std::vector<std::complex<double> > deltaKDiff(N * (N / 2 + 1));
        std::vector<std::complex<double> > deltaKFisherNew(N * (N / 2 + 1));
        std::vector<double> deltaXSum(N * N), deltaXDiff(N * N), deltaGamma1Diff(N * N), deltaGamma2Diff(N * N);

        Math::Matrix<double> fisherEst(nBins, nBins, 0);
        std::vector<std::complex<double> > deltaKTotal(N * (N / 2 + 1));
        std::vector<double> deltaXTotal(N * N);
        std::vector<double> deltaSumGamma1(N * N), deltaSumGamma2(N * N);
        std::vector<double> deltaTotalGamma1(N * N), deltaTotalGamma2(N * N);
        std::vector<double> deltaXFisherNew(N * N);
        std::vector<double> deltaGamma1FisherNew(N * N);
        std::vector<double> deltaGamma2FisherNew(N * N);

        std::vector<double> pkInv(pkFiducial);
        for(int i = 0; i < pkInv.size(); ++i)
        {
            check(pkInv[i] != 0, "");
            pkInv[i] = 1.0 / pkInv[i];
        }

        DeltaKVector2 deltaKDiffLV(N, N, nExtra);

        std::vector<int> fisherBins;
        int nonSparseEdges = 2;
        if(N > 32)
            nonSparseEdges *= (N / 32);
        check(nonSparseEdges * 2 < nBins, "");
        for(int l = 0; l < nonSparseEdges; ++l)
            fisherBins.push_back(l);
        for(int l = nonSparseEdges; l < nBins - nonSparseEdges; l += deltaLSparse)
            fisherBins.push_back(l);
        for(int l = nBins - nonSparseEdges; l < nBins; ++l)
            fisherBins.push_back(l);

        std::vector<double> pSimAvg(nBins, 0);
        std::vector<double> pRatioAvg(nBins, 0);
        std::vector<double> thetaEst(nBins);

        std::vector<double> pkChi2;

        std::vector<double> gibbsPSSum(nBins, 0);


        std::vector<std::vector<std::complex<double> > > fisherNewSignals, fisherNewMins;
        for(int c = 0; c < simCount; ++c)
        {
            generateWhiteNoise(N, N, noiseSeed++, noiseXNew, z2, normalizePower, normalizePerBin);
            if(weakLensing)
                generateWhiteNoise(N, N, noiseSeed++, noiseXNew1, z2, normalizePower, normalizePerBin);
            
            for(int i = 0; i < N * N; ++i)
            {
                noiseXNew[i] *= sigmaNoise[i];
                if(weakLensing)
                    noiseXNew1[i] *= sigmaNoise[i];
            }

            if(includeHigherPower)
            {
                generateDeltaK(NHigher, NHigher, pkHigher, &deltaKHigher, noiseSeed++);
                std::vector<double> deltaXHigher;
                deltaK2deltaX(NHigher, NHigher, deltaKHigher, &deltaXHigher, L, L);
                std::vector<double> deltaXLower;
                downgrade(NHigher, NHigher, deltaXHigher, N, N, &deltaXLower);
                std::vector<std::complex<double> > deltaKLower;
                deltaX2deltaK(N, N, deltaXLower, &deltaKLower, L, L);
                check(deltaKLower.size() == deltaK.size(), "");

                if(weakLensing)
                {
                    std::vector<double> deltaGamma1Lower(N * N), deltaGamma2Lower(N * N);
                    func.calculateWeakLensingData(deltaKLower, &deltaGamma1Lower, &deltaGamma2Lower);

                    for(int i = 0; i < N * N; ++i)
                    {
                        noiseXNew[i] += deltaGamma1Lower[i];
                        noiseXNew1[i] += deltaGamma2Lower[i];
                    }
                }
                else
                {
                    for(int i = 0; i < N * N; ++i)
                        noiseXNew[i] += deltaXLower[i];
                }
            }

            if(weakLensing)
                func.setData(dataX, &noiseXNew, &noiseXNew1);
            else
                func.setData(noiseXNew);
            DeltaKVector2 deltaKNoiseNew(starting);
            lbfgs.setStarting(starting);
            lbfgs.minimize(&deltaKNoiseNew, epsilon, gTol, 10000000);

            noiseRealizations.push_back(deltaKNoiseNew.get());

            for(int i = 0; i < N * N; ++i)
            {
                const int l = binsFull[i];
                check(l >= 0 && l < nBins, "");
                int ix = i / N, iy = i % N;
                if(iy > N / 2)
                {
                    if(ix != 0 && ix != N / 2)
                        ix = N - ix;
                    iy = N - iy;
                }
                const int indexNew = ix * (N / 2 + 1) + iy;
                check(indexNew >= 0 && indexNew < N * (N / 2 + 1), "");
                std::complex<double> s(0, 0);
                if(pkFiducial[indexNew] != 0)
                    s = deltaKNoiseNew.get()[indexNew] / pkFiducial[indexNew];
                bEstimated[l] += std::abs(s) * std::abs(s);

                noiseAvg[i] += std::abs(deltaKNoiseNew.get()[indexNew]) * std::abs(deltaKNoiseNew.get()[indexNew]);
            }

            // estimate it at EACH iteration
            std::vector<double> bEstimatedThisIter = bEstimated;
            for(int i = 0; i < nBins; ++i)
                bEstimatedThisIter[i] /= (c + 1);

            std::stringstream thisFileName;
            thisFileName << "b_est_" << c << ".txt";
            vector2file(thisFileName.str().c_str(), bEstimatedThisIter);

            std::vector<double> noiseAvgThisIter = noiseAvg;

            for(int i = 0; i < N * N; ++i)
                noiseAvgThisIter[i] /= (c + 1);

            // fisher
            DeltaKVector2 deltaKSumMin(starting);
            DeltaKVector2 deltaKFisherNewMin(starting);
            DeltaKVector2 deltaKFisherNewNoiseMin(starting);

            if(fisherNew)
            {
                generateDeltaK(N, N, pkFiducial, &deltaKFisherNew, seed++, &realBuffer);
                deltaK2deltaX(N, N, deltaKFisherNew, &deltaXFisherNew, L, L, &complexBuffer, true);

                if(weakLensing)
                {
                    func.calculateWeakLensingData(deltaKFisherNew, &deltaGamma1FisherNew, &deltaGamma2FisherNew);
                    func.setData(deltaXFisherNew, &deltaGamma1FisherNew, &deltaGamma2FisherNew);
                }
                else
                    func.setData(deltaXFisherNew);

                lbfgs.setStarting(starting);
                lbfgs.minimize(&deltaKFisherNewMin, epsilon, gTol, 10000000);


                // now add noise
                for(int i = 0; i < N * N; ++i)
                {
                    deltaXFisherNew[i] += noiseXNew[i];
                    if(weakLensing)
                    {
                        deltaGamma1FisherNew[i] += noiseXNew[i];
                        deltaGamma2FisherNew[i] += noiseXNew1[i];
                    }
                }

                if(weakLensing)
                {
                    func.setData(deltaXFisherNew, &deltaGamma1FisherNew, &deltaGamma2FisherNew);
                }
                else
                    func.setData(deltaXFisherNew);

                lbfgs.setStarting(starting);
                lbfgs.minimize(&deltaKFisherNewNoiseMin, epsilon, gTol, 10000000);

                fisherNewSignals.push_back(deltaKFisherNew);
                fisherNewMins.push_back(deltaKFisherNewNoiseMin.get());

                if(c % 100 == 0)
                {
                    Math::Matrix<double> fisherMatrixNew(nBins, nBins, 0);
                    for(int i = 0; i < N * N; ++i)
                    {
                        const int l = binsFull[i];
                        check(l >= 0 && l < nBins, "");
                        int ix = i / N, iy = i % N;
                        bool iConj = false;
                        if(iy > N / 2)
                        {
                            if(ix != 0 && ix != N / 2)
                                ix = N - ix;
                            iy = N - iy;
                            iConj = true;
                        }
                        const int iNew = ix * (N / 2 + 1) + iy;
                        const double p = fiducialBinVals[l] * L * L;
                        for(int j = 0; j < N * N; ++j)
                        {
                            const int l1 = binsFull[j];
                            check(l1 >= 0 && l1 < nBins, "");
                            int jx = j / N, jy = j % N;
                            bool jConj = false;
                            if(jy > N / 2)
                            {
                                if(jx != 0 && jx != N / 2)
                                    jx = N - jx;
                                jy = N - jy;
                                jConj = true;
                            }
                            const int jNew = jx * (N / 2 + 1) + jy;
                            std::complex<double> sum1 = 0.0;
                            double sum2 = 0;
                            for(int k = 0; k < c + 1; ++k)
                            {
                                std::complex<double> ss = fisherNewSignals[k][jNew];
                                std::complex<double> smin = fisherNewMins[k][iNew];

                                if(iConj)
                                    smin = std::conj(smin);
                                if(jConj)
                                    ss = std::conj(ss);

                                sum1 += smin * std::conj(ss);
                                sum2 += std::abs(ss) * std::abs(ss);
                            }

                            sum1 /= (c + 1);
                            sum2 /= (c + 1);
                            const std::complex<double> T = sum1 / sum2;

                            fisherMatrixNew(l, l1) += std::abs(T) * std::abs(T) / (2 * p * p);
                        }
                    }

                    std::stringstream thisIterFileName;
                    thisIterFileName << "fisher_new_est_" << c << ".txt";
                    matrix2file(thisIterFileName.str().c_str(), fisherMatrixNew);
                }
            }

            /*
            generateDeltaK(N, N, pkFiducial, &deltaKSum, seed++, &realBuffer);

            deltaK2deltaX(N, N, deltaKSum, &deltaXSum, L, L, &complexBuffer, true);

            if(weakLensing)
            {
                func.calculateWeakLensingData(deltaKSum, &deltaSumGamma1, &deltaSumGamma2);
                func.setData(deltaXSum, &deltaSumGamma1, &deltaSumGamma2);
            }
            else
                func.setData(deltaXSum);

            lbfgs.setStarting(starting);
            lbfgs.minimize(&deltaKSumMin, epsilon, gTol, 10000000);
            */

            for(int b1 = 0; b1 < fisherBins.size(); ++b1)
            {
                const int l1 = fisherBins[b1];

                const double p = fiducialBinVals[l1] * L * L;
                const double pExtra = p / 100;
                std::vector<double> pkExtra(N * (N / 2 + 1), 0);
                for(int i = 1; i < N * (N / 2 + 1); ++i)
                {
                    int thisBin = bins[i];
                    if(thisBin == -1)
                    {
                        const int i1 = i / (N / 2 + 1);
                        const int j1 = i % (N / 2 + 1);
                        check(j1 == 0 || j1 == N / 2, "");
                        check(i1 > N / 2, "");
                        const int i1New = N - i1;
                        const int iNew = i1New * (N / 2 + 1) + j1;
                        thisBin = bins[iNew];
                    }
                    check(thisBin >= 0 && thisBin <= nBins, "");

                    if(thisBin == l1)
                        pkExtra[i] = pExtra;
                }

                generateDeltaK(N, N, pkExtra, &deltaKDiff, seed++, &realBuffer, z2, normalizePower, normalizePerBin);
                for(int i = 0; i < N * (N / 2 + 1); ++i)
                {
                    //deltaKTotal[i] = deltaKSum[i] + deltaKDiff[i];
                    deltaKTotal[i] = deltaKDiff[i];
                }

                deltaK2deltaX(N, N, deltaKTotal, &deltaXTotal, L, L, &complexBuffer, true);
                if(weakLensing)
                {
                    func.calculateWeakLensingData(deltaKTotal, &deltaTotalGamma1, &deltaTotalGamma2);
                    func.setData(deltaXTotal, &deltaTotalGamma1, &deltaTotalGamma2);
                }
                else
                    func.setData(deltaXTotal);

                DeltaKVector2 deltaKTotalMin(starting);
                lbfgs.setStarting(starting);
                lbfgs.minimize(&deltaKTotalMin, epsilon, gTol, 10000000);

                double sum1 = 0;
                for(int i = 0; i < N * N; ++i)
                {
                    int ix = i / N, iy = i % N;
                    if(iy > N / 2)
                    {
                        if(ix != 0 && ix != N / 2)
                            ix = N - ix;
                        iy = N - iy;
                    }
                    const int indexNew = ix * (N / 2 + 1) + iy;
                    check(indexNew >= 0 && indexNew < N * (N / 2 + 1), "");
                    if(binsFull[i] == l1)
                        sum1 += std::abs(deltaKDiff[indexNew]) * std::abs(deltaKDiff[indexNew]);
                }

                for(int i = 0; i < N * N; ++i)
                {
                    const int l = binsFull[i];
                    check(l >= 0 && l < nBins, "");
                    int ix = i / N, iy = i % N;
                    if(iy > N / 2)
                    {
                        if(ix != 0 && ix != N / 2)
                            ix = N - ix;
                        iy = N - iy;
                    }
                    const int indexNew = ix * (N / 2 + 1) + iy;
                    check(indexNew >= 0 && indexNew < N * (N / 2 + 1), "");

                    //const std::complex<double> diff = deltaKTotalMin.get()[indexNew] - deltaKSumMin.get()[indexNew];
                    const std::complex<double> diff = deltaKTotalMin.get()[indexNew];
                    fisherEst(l, l1) += std::abs(diff) * std::abs(diff) / sum1;
                }
            }

            // estimating the fisher matrix at EACH iteration
            Math::Matrix<double> fisherEstThisIter = fisherEst;

            for(int l = 0; l < nBins; ++l)
            {
                const double p = fiducialBinVals[l] * L * L;

                for(int l1 = 0; l1 < nBins; ++l1)
                {
                    fisherEstThisIter(l, l1) /= (2 * p * p);
                    fisherEstThisIter(l, l1) /= (c + 1);
                    fisherEstThisIter(l, l1) *= binCount[l1];
                }

                // clean up possible noise
                for(int l1 = 0; l1 < nBins; ++l1)
                {
                    if(std::abs(l1 - l) > 40)
                        fisherEstThisIter(l, l1) = 0;
                }
            }

            std::stringstream thisIterFileName;
            thisIterFileName << "fisher_est_" << c << ".txt";
            matrix2file(thisIterFileName.str().c_str(), fisherEstThisIter);

            Math::Matrix<double> windowEstThisIter(fisherEstThisIter);
            windowEstThisIter.transpose();
            fisher2window(windowEstThisIter);

            std::stringstream windowThisIterFileName;
            windowThisIterFileName << "window_est_" << c << ".txt";
            matrix2file(windowThisIterFileName.str().c_str(), windowEstThisIter);

            // pk
            if(gibbs)
            {
                generateDeltaK(N, N, pkInv, &deltaKDiff, seed++, &realBuffer);

                deltaK2deltaX(N, N, deltaKDiff, &deltaXDiff, L, L, &complexBuffer, true);

                const double factor = (L * L * L * L / (N * N));

                if(weakLensing)
                {
                    func.calculateWeakLensingData(deltaKDiff, &deltaGamma1Diff, &deltaGamma2Diff);
                    for(int i = 0; i < N * N; ++i)
                    {
                        deltaGamma1Diff[i] *= sigmaNoise[i] * sigmaNoise[i] * factor;
                        deltaGamma2Diff[i] *= sigmaNoise[i] * sigmaNoise[i] * factor;
                    }
                    func.setData(deltaXDiff, &deltaGamma1Diff, &deltaGamma2Diff);
                }
                else
                {
                    for(int i = 0; i < N * N; ++i)
                        deltaXDiff[i] *= sigmaNoise[i] * sigmaNoise[i] * factor;

                    func.setData(deltaXDiff);
                }

                DeltaKVector2 deltaKDiffMin(starting);
                lbfgs.setStarting(starting);
                lbfgs.minimize(&deltaKDiffMin, epsilon, gTol, 10000000);

                DeltaKVector2 deltaKDiffMin1(starting);
                deltaKDiffMin1.get() = deltaKDiff;
                lbfgsOrig.applyInverseHessian(&deltaKDiffMin1);

                if(c == 0)
                {
                    std::vector<double> diff;
                    deltaK2deltaX(N, N, deltaKDiffMin.get(), &diff, L, L, &complexBuffer, true);
                    vector2file("gibbs_diff.txt", diff);
                    deltaK2deltaX(N, N, deltaKDiffMin1.get(), &diff, L, L, &complexBuffer, true);
                    vector2file("gibbs_diff1.txt", diff);
                }

                for(int i = 0; i < N * (N / 2 + 1); ++i)
                    deltaKSum[i] = deltaKMin.get()[i] + noiseRealizations[c][i] + 1.0 * deltaKDiffMin.get()[i];

                // renormalize power
                /*
                for(int i = 1; i < N * (N / 2 + 1); ++i)
                {
                    const double a = std::abs(deltaKSum[i]);
                    check(a > 0, "");
                    const double r = std::sqrt(pk[i]) / a;
                    deltaKSum[i] *= r;
                }
                */

                std::vector<double> deltaSumPS, deltaSumPSK;
                power(N, N, L, L, deltaKSum, &deltaSumPSK, &deltaSumPS);

                for(int i = 0; i < nBins; ++i)
                    gibbsPSSum[i] += deltaSumPS[i];

                std::vector<double> gibbsPSAvg = gibbsPSSum;
                for(int i = 0; i < nBins; ++i)
                    gibbsPSAvg[i] /= (c + 1);

                std::stringstream gibbsPSFileName;
                gibbsPSFileName << "gibbs_ps_" << c << ".txt";
                vector2file(gibbsPSFileName.str().c_str(), deltaSumPS);

                std::stringstream gibbsPSAvgFileName;
                gibbsPSAvgFileName << "gibbs_ps_avg_" << c << ".txt";
                vector2file(gibbsPSAvgFileName.str().c_str(), gibbsPSAvg);
            }
            else
            {
                generateDeltaK(N, N, pkFiducial, &deltaKSum, seed++, &realBuffer);
            }

            deltaK2deltaX(N, N, deltaKSum, &deltaXSum, L, L, &complexBuffer, true);

            if(weakLensing)
            {
                func.calculateWeakLensingData(deltaKSum, &deltaSumGamma1, &deltaSumGamma2);
                func.setData(deltaXSum, &deltaSumGamma1, &deltaSumGamma2);
            }
            else
                func.setData(deltaXSum);

            deltaKSumMin.copy(starting);
            lbfgs.setStarting(starting);
            lbfgs.minimize(&deltaKSumMin, epsilon, gTol, 10000000);

            std::vector<double> pNum(nBins, 0), pDenom(nBins, 0);

            for(int i = 0; i < N * N; ++i)
            {
                const int l = binsFull[i];
                check(l >= 0 && l < nBins, "");
                int ix = i / N, iy = i % N;
                if(iy > N / 2)
                {
                    if(ix != 0 && ix != N / 2)
                        ix = N - ix;
                    iy = N - iy;
                }
                const int indexNew = ix * (N / 2 + 1) + iy;
                check(indexNew >= 0 && indexNew < N * (N / 2 + 1), "");

                std::complex<double> s = deltaKSumMin.get()[indexNew];
                pSimAvg[l] += std::abs(s) * std::abs(s);

                pDenom[l] += std::abs(s) * std::abs(s);
                const std::complex<double> sNum = deltaKSum[indexNew];
                pNum[l] += std::abs(sNum) * std::abs(sNum);
            }

            std::vector<double> ratio(nBins);
            for(int l = 0; l < nBins; ++l)
            {
                double s = 0;
                for(int l1 = 0; l1 < nBins; ++l1)
                    s += windowEstThisIter(l, l1) * pNum[l1] / binCount[l1];
                ratio[l] = s / pDenom[l];

                pRatioAvg[l] += ratio[l];
            }

            std::vector<double> pRatioAvgThisIter = pRatioAvg;
            for(int l = 0; l < nBins; ++l)
                pRatioAvgThisIter[l] /= (c + 1);


            std::stringstream pRatioAvgFileName;
            pRatioAvgFileName << "theta_factor_" << c << ".txt";
            vector2file(pRatioAvgFileName.str().c_str(), pRatioAvgThisIter);

            std::vector<double> pSimAvgThisIter = pSimAvg;

            for(int i = 0; i < nBins; ++i)
                pSimAvgThisIter[i] /= (c + 1);

            std::stringstream pSimAvgFileName;
            pSimAvgFileName << "theta_norm_" << c << ".txt";
            vector2file(pSimAvgFileName.str().c_str(), pSimAvgThisIter);

            std::vector<double> thetaEstThisIter(nBins);
            for(int i = 0; i < N * N; ++i)
            {
                const int l = binsFull[i];
                check(l >= 0 && l < nBins, "");
                int ix = i / N, iy = i % N;
                if(iy > N / 2)
                {
                    if(ix != 0 && ix != N / 2)
                        ix = N - ix;
                    iy = N - iy;
                }
                const int indexNew = ix * (N / 2 + 1) + iy;
                check(indexNew >= 0 && indexNew < N * (N / 2 + 1), "");

                std::complex<double> s = deltaKMin.get()[indexNew];
                thetaEstThisIter[l] += (std::abs(s) * std::abs(s) - noiseAvgThisIter[i]);
            }

            for(int l = 0; l < nBins; ++l)
            {
                double s = 0;
                for(int l1 = 0; l1 < nBins; ++l1)
                    s += windowEstThisIter(l, l1) * fiducialBinVals[l1];

                //thetaEstThisIter[l] *= (s / pSimAvgThisIter[l]);
                thetaEstThisIter[l] *= (pRatioAvgThisIter[l] / (L * L));
            }

            thetaEst = thetaEstThisIter;

            std::stringstream thisThetaFileName;
            thisThetaFileName << "theta_est_" << c << ".txt";
            vector2file(thisThetaFileName.str().c_str(), thetaEstThisIter);

            std::vector<double> thetaError(nBins);
            for(int l = 0; l < nBins; ++l)
            {
                thetaError[l] = windowEstThisIter(l, l);
                thetaError[l] *= pRatioAvgThisIter[l] * 2 * fiducialBinVals[l] * fiducialBinVals[l];
                thetaError[l] = std::sqrt(thetaError[l]);
            }

            std::stringstream thisThetaErrorFileName;
            thisThetaErrorFileName << "theta_error_" << c << ".txt";
            vector2file(thisThetaErrorFileName.str().c_str(), thetaError);

            std::vector<double> fiducialConvolved(nBins);
            for(int l = 0; l < nBins; ++l)
            {
                for(int l1 = 0; l1 < nBins; ++l1)
                    fiducialConvolved[l] += windowEstThisIter(l, l1) * fiducialBinVals[l1];
            }

            double chi2 = 0;
            for(int l = 0; l < nBins; ++l)
            {
                const double diff = thetaEstThisIter[l] - fiducialConvolved[l];
                chi2 += diff * diff / (thetaError[l] * thetaError[l]);
            }
            pkChi2.push_back(chi2);
        }

        vector2file("pk_chi2.txt", pkChi2);


        if(N <= lin_lim)
        {
            std::vector<std::complex<double> > wfk(N * (N / 2 + 1));
            std::vector<double> wfx(N * N);
            std::vector<double> b(nBins);
            Math::Matrix<double> fisher(nBins, nBins);
            std::vector<double> signal(nBins, 0);
            std::vector<double> theta(nBins);

            Math::Matrix<double> invHessian;

            linearAlgebraCalculation(N, L, pkFiducial, weakLensing, mask, sigmaNoise, dataX, dataGamma1, dataGamma2, wfk, wfx, b, fisher, signal, theta, &invHessian);

            vector2binFile("wfx2.dat", wfx);
            std::vector<double> wfPS, wfPSK;
            power(N, N, L, L, wfk, &wfPSK, &wfPS);
            vector2file("wf_ps.txt", wfPS);
            vector2file("bl.txt", b);
            matrix2file("fisher.txt", fisher);
            vector2file("signal.txt", signal);
            vector2file("theta.txt", theta);
            matrix2file("inv_hessian.txt", invHessian);

            Math::Matrix<double> window(fisher);
            fisher2window(window);
            matrix2file("window.txt", window);

            seed = 10000;
            std::vector<double> testVec(N * (N / 2 + 1));
            Math::GaussianGenerator testGen(seed++, 0, 1);
            for(int i = 0; i < testVec.size(); ++i)
                testVec[i] = testGen.generate();

            Math::Matrix<double> testVecMat(N * N, 1);
            for(int i = 0; i < N * N; ++i)
            {
                int ix = i / N, iy = i % N;
                if(iy > N / 2)
                {
                    if(ix != 0 && ix != N / 2)
                        ix = N - ix;
                    iy = N - iy;
                }
                const int indexNew = ix * (N / 2 + 1) + iy;
                check(indexNew >= 0 && indexNew < N * (N / 2 + 1), "");
                testVecMat(i, 0) = testVec[indexNew];
            }
            Math::Matrix<double> invHTestVec = invHessian * testVecMat;

            DeltaKVector2 testVecV(starting);
            for(int i = 0; i < testVec.size(); ++i)
                testVecV.get()[i] = testVec[i];
            lbfgsOrig.applyInverseHessian(&testVecV);

            std::vector<double> t1(N * (N / 2 + 1)), t2(N * (N / 2 + 1));
            for(int i = 0; i < testVec.size(); ++i)
            {
                t1[i] = invHTestVec(i, 0);
                t2[i] = std::real(testVecV.get()[i]);
            }

            vector2file("inv_hess_test_vec.txt", t1);
            vector2file("inv_hess_test_vec_est.txt", t2);

            // skip the last part (for now)
            return 0;

            // calculate ps_quad for many realizations
            const int dataSimCount = 8;
            seed = 1000;
            std::vector<std::complex<double> > deltaKData;
            std::vector<double> deltaXData;
            std::vector<double> deltaXGamma1, deltaXGamma2;
            std::vector<double> thetaAvg(nBins, 0);
            std::vector<double> deltaPKAvg(nBins, 0);
            for(int c = 0; c < dataSimCount; ++c)
            {
                generateDeltaK(N, N, pk, &deltaKData, seed++, &realBuffer);

                std::vector<double> kValsDeltaP, pValsDeltaP;
                power(N, N, L, L, deltaKData, &kValsDeltaP, &pValsDeltaP);
                for(int i = 0; i < nBins; ++i)
                    deltaPKAvg[i] += pValsDeltaP[i] / dataSimCount;

                deltaK2deltaX(N, N, deltaKData, &deltaXData, L, L, &complexBuffer, true);
                check(deltaXData.size() == N * N, "");
                Math::GaussianGenerator noiseGen(seed++, 0, 1);
                
                if(weakLensing)
                {
                    func.calculateWeakLensingData(deltaKData, &deltaXGamma1, &deltaXGamma2);
                    for(int i = 0; i < N * N; ++i)
                    {
                        deltaXGamma1[i] += noiseGen.generate() * sigmaNoise[i];
                        deltaXGamma2[i] += noiseGen.generate() * sigmaNoise[i];
                    }
                }
                else
                {
                    for(int i = 0; i < N * N; ++i)
                        deltaXData[i] += noiseGen.generate() * sigmaNoise[i];
                }

                linearAlgebraCalculation(N, L, pkFiducial, weakLensing, mask, sigmaNoise, deltaXData, deltaXGamma1, deltaXGamma2, wfk, wfx, b, fisher, signal, theta);

                for(int i = 0; i < nBins; ++i)
                    thetaAvg[i] += theta[i];
            }

            for(int i = 0; i < nBins; ++i)
                thetaAvg[i] /= dataSimCount;

            vector2file("theta_avg.txt", thetaAvg);
            vector2file("delta_pk_avg.txt", deltaPKAvg);
        }
    } catch (std::exception& e)
    {
        output_screen("EXCEPTION CAUGHT!!! " << std::endl << e.what() << std::endl);
        output_screen("Terminating!" << std::endl);
        return 1;
    }
    return 0;
}
