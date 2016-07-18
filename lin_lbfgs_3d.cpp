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
#include "delta_vector.hpp"

#include <fftw3.h>

namespace
{

class WL3DFunc
{
public:
    WL3DFunc(int N1, int N2, int N3, double L1, double L2, double L3, const std::vector<double>& gamma1, const std::vector<double>& gamma2, const std::vector<double>& sigma1, const std::vector<double>& sigma2, const std::vector<double>& mask, const std::vector<double>& pk), N1_(N1), N2_(N2), N3_(N3), L1_(L1), L2_(L2), L3_(L3), gamma1_(gamma1), gamma2_(gamma2), sigma1_(sigma1), sigma2_(sigma2), mask_(mask), pk_(pk), deltaK_(N1 * N2 * (N3 / 2 + 1)), deltaX_(N1 * N2 * N3), kappa_(N1 * N2), kappaK_(N1 * (N2 / 2 + 1)), buf_(N1 * N2 * (N3 / 2 + 1)), deltaGamma1_(N1 * N2), deltaGamma2(N1 * N2), gammaK1_(N1 * (N2 / 2 + 1)), gammaK2_(N1 * (N2 / 2 + 1))
    {
        check(N1_ > 0, "");
        check(N2_ > 0, "");
        check(N3_ > 0, "");
        check(L1_ > 0, "");
        check(L2_ > 0, "");
        check(L3_ > 0, "");
        check(gamma1_.size() == N1_ * N2_, "");
        check(gamma2_.size() == N1_ * N2_, "");
        check(sigma1_.size() == N1_ * N2_, "");
        check(sigma2_.size() == N1_ * N2_, "");
        check(mask_.size() == N1_ * N2_, "");
        check(pk_.size() == N1_ * N2_ * (N3_ / 2 + 1), "");
    }

    void setData(const std::vector<double>& gamma1, const std::vector<double>& gamma2)
    {
        check(gamma1.size() == N1_ * N2_, "");
        check(gamma2.size() == N1_ * N2_, "");

        gamma1_ = gamma1;
        gamma2_ = gamma2;
    }

    void set(const DeltaVector3& x)
    {
        check(x.isComplex(), "");
        check(x.getComplex().size() == N1_ * N2_ * (N3_ / 2 + 1), "");
        deltaK_ = x.getComplex();
        deltaK2deltaX(N1_, N2_, N3_, deltaK_, &deltaX_, L1_, L2_, L3_, &buf_, true);
        project(deltaX_, &kappa_);
        deltaX2deltaK(N1_, N2_, kappa_, &kappaK_, L1_, L2_);
        calculateWeakLensingData(kappaK_, &deltaGamma1_, &deltaGamma2_);
    }

    // for MPI, ALL the processes should get the function value
    double value()
    {
        double res = 0;
        for(int i = 0; i < N1_ * N2_; ++i)
        {
            double delta = mask_[i] * (deltaGamma1_[i] - gamma1_[i]);
            double s = sigma1_[i];
            check(s > 0, "");
            res += delta * delta / (s * s);
            delta = mask_[i] * (deltaGamma2_[i] - gamma2_[i]);
            s = sigma2_[i];
            check(s > 0, "");
            res += delta * delta / (s * s);
        }
    }

    void derivative(DeltaVector3 *res)
    {
        check(res->isComplex(), "");
        std::vector<std::complex<double> >& r = res->getComplex();
        check(r.size() == N1_ * N2_ * (N3_ / 2 + 1);

        priorKDeriv(N1_, N2_, N3_, pk_, deltaK_, &r);
    }

private:
    void project(const std::vector<double>& x, std::vector<double> *proj) const
    {
        check(x.size() == N1_ * N2_ * N3_, "");
        proj->resize(N1_ * N2_);

        for(int i = 0; i < N1_; ++i)
        {
            for(int j = 0; j < N2_; ++j)
            {
                double s = 0;
                for(int k = 0; k < N3_; ++k)
                    s += x[(i * N2_ + j) * N3_ + k];

                (*proj)[i * N2_ + j] = s;
            }
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
    const int N1_, N2_, N3_;
    const double L1_, L2_, L3_;
    std::vector<double> gamma1_, gamma2_;
    const std::vector<double> sigma1_, sigma2_, mask_;
    const std::vector<double> pk_;

    std::vector<std::complex<double> > deltaK_;
    std::vector<double> deltaX_;
    std::vector<double> kappa_;
    std::vector<std::complex<double> > kappaK_;
    std::vector<std::complex<double> > buf_;
    std::vector<double> deltaGamma1_, deltaGamma2_;
    std::vector<std::complex<double> > gammaK1_, gammaK2_;
};

} // namespace
