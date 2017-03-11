#include <cmath>

#include <macros.hpp>
#include <math_constants.hpp>
#include <numerics.hpp>

#include <power_spectrum.hpp>
#include <lyman_alpha.hpp>

#include <fftw3.h>

LymanAlpha::LymanAlpha(const std::vector<double>& delta, double L, double b) : delta_(delta), N_(delta.size()), L_(L), b_(b)
{
    check(L_ > 0, "");
    check(b_ > 0, "");
    check(N_ > 0, "");
    deltaLN_.resize(N_);
    v_.resize(N_);
    vDeriv_.resize(N_);
    tau_.resize(N_);
    flux_.resize(N_);

    complexBuffer_.resize(N_ / 2  + 1);

    reset(delta);
}

LymanAlpha::~LymanAlpha()
{
}

void
LymanAlpha::reset(const std::vector<double>& delta)
{
    check(delta.size() == N_, "");
    check(delta_.size() == N_, "");

    delta_ = delta;

    for(int i = 0; i < N_; ++i)
        deltaLN_[i] = std::exp(delta_[i]) - 1;

    vCalculated_ = false;
    tauCalculated_ = false;
    fluxCalculated_ = false;
    tauDerivsCalculated_ = false;
}

double
LymanAlpha::deltaDeriv(int n) const
{
    check(n >= 0 && n < N_, "");
    return std::exp(delta_[n]);
}

void
LymanAlpha::calculateV()
{
    check(v_.size() == N_, "");

    deltaX2deltaK(N_, deltaLN_, &complexBuffer_, L_);
    for(int i = 0; i < N_ / 2 + 1; ++i)
    {
        const double k = 2 * Math::pi * i / L_;
        const double k2 = (i == 0 ? 1.0 : k * k);
        complexBuffer_[i] *= std::complex<double>(0, k / k2);
    }

    deltaK2deltaX(N_, complexBuffer_, &v_, L_);

    for(int i = 0; i < N_ / 2 + 1; ++i)
    {
        const double k = 2 * Math::pi * i / L_;
        const double k2 = (i == 0 ? 1.0 : k * k);
        complexBuffer_[i] = std::complex<double>(0, k / k2);
    }

    check(vDeriv_.size() == N_, "");
    deltaK2deltaX(N_, complexBuffer_, &vDeriv_, 1.0);
    check(Math::areEqual(vDeriv_[0], 0.0, 1e-7), vDeriv_[0] << " should have been 0");

    for(int i = 0; i < N_; ++i)
        vDeriv_[i] /= N_;

    vCalculated_ = true;
}

double
LymanAlpha::vDeriv(int m, int n)
{
    check(m >= 0 && m < N_, "");
    check(n >= 0 && n < N_, "");

    if(!vCalculated_)
        calculateV();

    const double d1 = (m > n ? vDeriv_[m - n] : -vDeriv_[n - m]);
    const double d2 = std::exp(delta_[n]);

    return d1 * d2;
}

void
LymanAlpha::calculateTau()
{
    check(tau_.size() == N_, "");

    if(!vCalculated_)
        calculateV();

    for(int i = 0; i < N_; ++i)
        tau_[i] = 0;

    for(int i = 0; i < N_; ++i)
    {
        const double x = double(i) * L_ / N_;
        const double v = v_[i];
        const double xv = x + v;
        const double rho = deltaLN_[i] + 1;
        const double xMin = xv - 5 * b_;
        const double xMax = xv + 5 * b_;
        const int jMin = (int) std::floor(xMin / L_ * N_) - 1;
        const int jMax = (int) std::ceil(xMax / L_ * N_) + 1;
        for(int j = jMin; j <= jMax; ++j)
        {
            const int m = periodicIndex(j);
            const double u = j * L_ / N_;
            const double d = u - xv;
            tau_[m] += L_ / N_ * rho * rho * std::exp(-d * d / (b_ * b_));
        }
    }

    tauCalculated_ = true;
}

void
LymanAlpha::calculateTauDerivs()
{
    if(!vCalculated_)
        calculateV();

    tauDerivs_.resize(N_);
    for(int i = 0; i < N_; ++i)
    {
        tauDerivs_[i].resize(N_);
        for(int j = 0; j < N_; ++j)
            tauDerivs_[i][j] = 0;
    }

    for(int n = 0; n < N_; ++n)
    {
        const double x = double(n) * L_ / N_;
        const double v = v_[n];
        const double xv = x + v;
        const double xMin = xv - 5 * b_;
        const double xMax = xv + 5 * b_;
        const int iMin = (int) std::floor(xMin / L_ * N_) - 1;
        const int iMax = (int) std::ceil(xMax / L_ * N_) + 1;
        for(int i = iMin; i <= iMax; ++i)
        {
            const int iShifted = periodicIndex(i);
            const double u = i * L_ / N_;
            const double d = u - xv;
            tauDerivs_[iShifted][n] += 2 * L_ / N_ * (deltaLN_[n] + 1) * deltaDeriv(n) * std::exp(-d * d / (b_ * b_));
        }

        for(int i = 0; i < N_; ++i)
        {
            const double x = double(i) * L_ / N_;
            const double v = v_[i];
            const double xv = x + v;
            const double xMin = xv - 5 * b_;
            const double xMax = xv + 5 * b_;
            const int mMin = (int) std::floor(xMin / L_ * N_) - 1;
            const int mMax = (int) std::ceil(xMax / L_ * N_) + 1;
            for(int m = mMin; m <= mMax; ++m)
            {
                const int mShifted = periodicIndex(m);
                const double u = m * L_ / N_;
                const double d = u - xv;
                tauDerivs_[mShifted][n] += 2 * L_ / N_ * (deltaLN_[i] + 1) * (deltaLN_[i] + 1) * std::exp(-d * d / (b_ * b_)) * d / (b_ * b_) * vDeriv(i, n);
            }
        }
    }

    tauDerivsCalculated_ = true;
}


void
LymanAlpha::calculateFlux()
{
    check(flux_.size() == N_, "");
    
    if(!tauCalculated_)
        calculateTau();

    for(int i = 0; i < N_; ++i)
        flux_[i] = std::exp(-tau_[i]);

    fluxCalculated_ = true;
}

double
LymanAlpha::tauDeriv(int m, int n)
{
    check(m >= 0 && m < N_, "");
    check(n >= 0 && n < N_, "");

    if(!tauDerivsCalculated_)
        calculateTauDerivs();

    return tauDerivs_[m][n];

    /*
    double u = double(m) * L_ / N_;
    double x = double(n) * L_ / N_;
    double v = v_[n];
    double xv = x + v;
    bringToBox(&xv);
    double d1 = u - xv;
    double d2 = d1 - L_;
    double d3 = d1 + L_;

    double rho = deltaLN_[n] + 1;
    double res = rho * std::exp(delta_[n]) * (std::exp(-d1 * d1 / (b_ * b_)) + std::exp(-d2 * d2 / (b_ * b_)) + std::exp(-d3 * d3 / (b_ * b_)));

    for(int k = 0; k < N_; ++k)
    {
        x = double(k) * L_ / N_;
        v = v_[k];
        xv = x + v;
        bringToBox(&xv);
        d1 = u - xv;
        d2 = d1 - L_;
        d3 = d1 + L_;
        rho = deltaLN_[k] + 1;
        res += rho * rho * vDeriv(k, n) * (std::exp(-d1 * d1 / (b_ * b_)) * d1 / (b_ * b_) + std::exp(-d2 * d2 / (b_ * b_)) * d2 / (b_ * b_) + std::exp(-d3 * d3 / (b_ * b_)) * d3 / (b_ * b_));
    }

    res *= (2 * L_ / N_);

    return res;
    */
}

double LymanAlpha::fluxDeriv(int m, int n)
{
    if(!fluxCalculated_)
        calculateFlux();

    return -flux_[m] * tauDeriv(m, n);
}

// LymanAlpha2
LymanAlpha2::LymanAlpha2(int N1, int N2, const std::vector<double>& delta, double L1, double L2, double b) : N1_(N1), N2_(N2), delta_(delta), L1_(L1), L2_(L2), b_(b)
{
    check(N1_ > 0, "");
    check(N2_ > 0, "");
    check(L1_ > 0, "");
    check(L2_ > 0, "");
    check(b_ > 0, "");
    check(delta_.size() == N1_ * N2_, "");

    deltaLN_.resize(N1_ * N2_);
    v_.resize(N1_ * N2_);
    vDeriv_.resize(N1_ * N2_);
    tau_.resize(N1_ * N2_);
    flux_.resize(N1_ * N2_);
    complexBuffer_.resize(N1_ * (N2_ / 2  + 1));

    reset(delta);
}

LymanAlpha2::~LymanAlpha2()
{
}

void
LymanAlpha2::reset(const std::vector<double>& delta)
{
    check(delta.size() == N1_ * N2_, "");
    check(delta_.size() == N1_ * N2_, "");

    delta_ = delta;

    for(int i = 0; i < N1_ * N2_; ++i)
        deltaLN_[i] = std::exp(delta_[i]) - 1;

    vCalculated_ = false;
    tauCalculated_ = false;
    fluxCalculated_ = false;
    tauDerivsCalculated_ = false;
}

double
LymanAlpha2::deltaDeriv(int n) const
{
    check(n >= 0 && n < N1_ * N2_, "");
    return std::exp(delta_[n]);
}

void
LymanAlpha2::calculateV()
{
    check(v_.size() == N1_ * N2_, "");

    deltaX2deltaK(N1_, N2_, deltaLN_, &complexBuffer_, L1_, L2_);
    for(int i = 0; i < N1_; ++i)
    {
        const int iShifted = (i < (N1_ + 1) / 2 ? i : -(N1_ - i));
        double k1 = 2 * Math::pi / L1_ * iShifted;
        for(int j = 0; j < N2_ / 2 + 1; ++j)
        {
            int jShifted = (j < (N2_ + 1) / 2 ? j : -(N2_ - j));
            double k2 = 2 * Math::pi / L2_ * jShifted;
            double kSq = k1 * k1 + k2 * k2;

            // just to avoid annoying division by 0 (doesn't matter)
            if(kSq == 0)
                kSq = 1.0;

            complexBuffer_[i * (N2_ / 2 + 1) + j] *= std::complex<double>(0, k2 / kSq);
        }

    }

    deltaK2deltaX(N1_, N2_, complexBuffer_, &v_, L1_, L2_);

    for(int i = 0; i < N1_; ++i)
    {
        const int iShifted = (i < (N1_ + 1) / 2 ? i : -(N1_ - i));
        double k1 = 2 * Math::pi / L1_ * iShifted;
        for(int j = 0; j < N2_ / 2 + 1; ++j)
        {
            int jShifted = (j < (N2_ + 1) / 2 ? j : -(N2_ - j));
            /*
            if(j == N2_ / 2 && i > N1_ / 2)
                jShifted = -jShifted;
            */
            double k2 = 2 * Math::pi / L2_ * jShifted;
            double kSq = k1 * k1 + k2 * k2;

            // just to avoid annoying division by 0 (doesn't matter)
            if(kSq == 0)
                kSq = 1.0;

            complexBuffer_[i * (N2_ / 2 + 1) + j] = std::complex<double>(0, k2 / kSq);
        }

    }

    check(vDeriv_.size() == N1_ * N2_, "");
    deltaK2deltaX(N1_, N2_, complexBuffer_, &vDeriv_, 1.0, 1.0);

    check(Math::areEqual(vDeriv_[0], 0.0, 1e-7), vDeriv_[0] << " should have been 0");

    for(int i = 0; i < N1_ * N2_; ++i)
        vDeriv_[i] /= (N1_ * N2_);

    const double factor = 1.0;
    for(int i = 0; i < N1_ * N2_; ++i)
    {
        v_[i] *= factor;
        vDeriv_[i] *= factor;
    }

    vCalculated_ = true;
}

double
LymanAlpha2::vDeriv(int m1, int m2, int n1, int n2)
{
    check(m1 >= 0 && m1 < N1_, "");
    check(m2 >= 0 && m2 < N2_, "");
    check(n1 >= 0 && n1 < N1_, "");
    check(n2 >= 0 && n2 < N2_, "");

    if(!vCalculated_)
        calculateV();

    const int delta1 = std::abs(m1 - n1);
    const double d1 = (m2 > n2 ? vDeriv_[delta1 * N2_ + (m2 - n2)] : -vDeriv_[delta1 * N2_ + (n2 - m2)]);
    const double d2 = std::exp(delta_[n1 * N2_ + n2]);

    return d1 * d2;
}

void
LymanAlpha2::calculateTau()
{
    check(tau_.size() == N1_ * N2_, "");

    if(!vCalculated_)
        calculateV();

    for(int i = 0; i < N1_; ++i)
    {
        for(int j = 0; j < N2_; ++j)
            tau_[i * N2_ + j] = 0;

        for(int j = 0; j < N2_; ++j)
        {
            const double x = double(j) * L2_ / N2_;
            const double v = v_[i * N2_ + j];
            const double xv = x + v;
            const double rho = deltaLN_[i * N2_ + j] + 1;
            const double xMin = xv - 5 * b_;
            const double xMax = xv + 5 * b_;
            const int kMin = (int) std::floor(xMin / L2_ * N2_) - 1;
            const int kMax = (int) std::ceil(xMax / L2_ * N2_) + 1;
            for(int k = kMin; k <= kMax; ++k)
            {
                const int m = periodicIndex(k);
                const double u = k * L2_ / N2_;
                const double d = u - xv;
                tau_[i * N2_ + m] += L2_ / N2_ * rho * rho * std::exp(-d * d / (b_ * b_));
            }
        }
    }

    const double factor = 0.1;
    for(int i = 0; i < N1_ * N2_; ++i)
        tau_[i] *= factor;

    tauCalculated_ = true;
}

void
LymanAlpha2::calculateTauDerivs()
{
    if(!vCalculated_)
        calculateV();

    tauDerivsDeltaX_.resize(N1_);
    tauDerivsV_.resize(N1_);

    for(int m1 = 0; m1 < N1_; ++m1)
    {
        tauDerivsDeltaX_[m1].resize(N2_);
        tauDerivsV_[m1].resize(N2_);

        for(int i = 0; i < N2_; ++i)
        {
            tauDerivsDeltaX_[m1][i].resize(N2_, 0);
            tauDerivsV_[m1][i].resize(N2_, 0);
        }

        for(int n = 0; n < N2_; ++n)
        {
            const double x = double(n) * L2_ / N2_;
            const double v = v_[m1 * N2_ + n];
            const double xv = x + v;
            const double xMin = xv - 5 * b_;
            const double xMax = xv + 5 * b_;
            const int iMin = (int) std::floor(xMin / L2_ * N2_) - 1;
            const int iMax = (int) std::ceil(xMax / L2_ * N2_) + 1;

            for(int i = iMin; i <= iMax; ++i)
            {
                const int iShifted = periodicIndex(i);
                const double u = i * L2_ / N2_;
                const double d = u - xv;
                tauDerivsDeltaX_[m1][iShifted][n] += 2 * L2_ / N2_ * (deltaLN_[m1 * N2_ + n] + 1) * deltaDeriv(m1 * N2_ + n) * std::exp(-d * d / (b_ * b_));
                tauDerivsV_[m1][iShifted][n] += 2 * L2_ / N2_ * (deltaLN_[m1 * N2_ + n] + 1) * (deltaLN_[m1 * N2_ + n] + 1) * std::exp(-d * d / (b_ * b_)) * d / (b_ * b_);
            }
        }
    }

    tauDerivsCalculated_ = true;
}


void
LymanAlpha2::calculateFlux()
{
    check(flux_.size() == N1_ * N2_, "");
    
    if(!tauCalculated_)
        calculateTau();

    for(int i = 0; i < N1_ * N2_; ++i)
        flux_[i] = std::exp(-tau_[i]);

    fluxCalculated_ = true;
}

double
LymanAlpha2::tauDeriv(int m1, int m2, int n1, int n2)
{
    check(m1 >= 0 && m1 < N1_, "");
    check(m2 >= 0 && m2 < N2_, "");
    check(n1 >= 0 && n1 < N1_, "");
    check(n2 >= 0 && n2 < N2_, "");

    if(!tauDerivsCalculated_)
        calculateTauDerivs();

    double res = 0;
    if(m1 == n1)
        res += tauDerivsDeltaX_[m1][m2][n2];

    for(int i = 0; i < N2_; ++i)
        res += tauDerivsV_[m1][m2][i] * vDeriv(m1, i, n1, n2);

    const double factor = 0.1;

    return res * factor;
}

double LymanAlpha2::fluxDeriv(int m1, int m2, int n1, int n2)
{
    if(!fluxCalculated_)
        calculateFlux();

    return -flux_[m1 * N2_ + m2] * tauDeriv(m1, m2, n1, n2);
}

// LymanAlpha3
LymanAlpha3::LymanAlpha3(int N1, int N2, int N3, const std::vector<double>& delta, double L1, double L2, double L3, double b) : N1_(N1), N2_(N2), N3_(N3), delta_(delta), L1_(L1), L2_(L2), L3_(L3), b_(b), tauFactor_(1.0)
{
    check(N1_ > 0, "");
    check(N2_ > 0, "");
    check(N3_ > 0, "");
    check(L1_ > 0, "");
    check(L2_ > 0, "");
    check(L3_ > 0, "");
    check(b_ > 0, "");
    check(delta_.size() == N1_ * N2_ * N3_, "");

    deltaLN_.resize(N1_ * N2_ * N3_);
    v_.resize(N1_ * N2_ * N3_);
    vDeriv_.resize(N1_ * N2_ * N3_);
    tau_.resize(N1_ * N2_ * N3_);
    flux_.resize(N1_ * N2_ * N3_);
    complexBuffer_.resize(N1_ * N2_ * (N3_ / 2  + 1));

    reset(delta);
}

LymanAlpha3::~LymanAlpha3()
{
}

void
LymanAlpha3::reset(const std::vector<double>& delta)
{
    check(delta.size() == N1_ * N2_ * N3_, "");
    check(delta_.size() == N1_ * N2_ * N3_, "");

    delta_ = delta;

    for(int i = 0; i < N1_ * N2_ * N3_; ++i)
        deltaLN_[i] = std::exp(delta_[i]) - 1;

    vCalculated_ = false;
    tauCalculated_ = false;
    fluxCalculated_ = false;
    tauDerivsCalculated_ = false;
}

double
LymanAlpha3::deltaDeriv(int n) const
{
    check(n >= 0 && n < N1_ * N2_ * N3_, "");
    return std::exp(delta_[n]);
}

void
LymanAlpha3::calculateV()
{
    check(v_.size() == N1_ * N2_ * N3_, "");

    deltaX2deltaK(N1_, N2_, N3_, deltaLN_, &complexBuffer_, L1_, L2_, L3_);
    for(int i = 0; i < N1_; ++i)
    {
        const int iShifted = (i < (N1_ + 1) / 2 ? i : -(N1_ - i));
        const double k1 = 2 * Math::pi / L1_ * iShifted;
        for(int j = 0; j < N2_; ++j)
        {
            const int jShifted = (j < (N2_ + 1) / 2 ? j : -(N2_ - j));
            const double k2 = 2 * Math::pi / L2_ * jShifted;
            for(int k = 0; k < N3_ / 2 + 1; ++k)
            {
                int kShifted = (k < (N3_ + 1) / 2 ? k : -(N3_ - k));
                const double k3 = 2 * Math::pi / L3_ * kShifted;
                double kSq = k1 * k1 + k2 * k2 + k3 * k3;

                // just to avoid annoying division by 0 (doesn't matter)
                if(kSq == 0)
                    kSq = 1.0;

                double factor = 1;
                if(k == N3_ / 2)
                {
                    /*
                    if(j > N2_ / 2)
                        factor = -1;
                    else if(j == N2_ / 2 && i > N1_ / 2)
                        factor = -1;
                    if((i == 0 || i == N1_ / 2) && (j == 0 || j == N2_ / 2))
                        factor = 0;
                    */

                    factor = 0;
                }

                complexBuffer_[(i * N2_ + j) * (N3_ / 2 + 1) + k] *= std::complex<double>(0, factor * k3 / kSq);
            }
        }

    }

    deltaK2deltaX(N1_, N2_, N3_, complexBuffer_, &v_, L1_, L2_, L3_);

    for(int i = 0; i < N1_; ++i)
    {
        const int iShifted = (i < (N1_ + 1) / 2 ? i : -(N1_ - i));
        const double k1 = 2 * Math::pi / L1_ * iShifted;
        for(int j = 0; j < N2_; ++j)
        {
            const int jShifted = (j < (N2_ + 1) / 2 ? j : -(N2_ - j));
            const double k2 = 2 * Math::pi / L2_ * jShifted;
            for(int k = 0; k < N3_ / 2 + 1; ++k)
            {
                int kShifted = (k < (N3_ + 1) / 2 ? k : -(N3_ - j));
                const double k3 = 2 * Math::pi / L3_ * kShifted;
                double kSq = k1 * k1 + k2 * k2 + k3 * k3;

                // just to avoid annoying division by 0 (doesn't matter)
                if(kSq == 0)
                    kSq = 1.0;

                double factor = 1;
                if(k == N3_ / 2)
                {
                    /*
                    if(j > N2_ / 2)
                        factor = -1;
                    else if(j == N2_ / 2 && i > N1_ / 2)
                        factor = -1;
                    if((i == 0 || i == N1_ / 2) && (j == 0 || j == N2_ / 2))
                        factor = 0;
                    */

                    factor = 0;
                }

                complexBuffer_[(i * N2_ + j) * (N3_ / 2 + 1) + k] = std::complex<double>(0, factor * k3 / kSq);
            }
        }
    }

    check(vDeriv_.size() == N1_ * N2_ * N3_, "");
    deltaK2deltaX(N1_, N2_, N3_, complexBuffer_, &vDeriv_, 1.0, 1.0, 1.0);

    check(Math::areEqual(vDeriv_[0], 0.0, 1e-7), vDeriv_[0] << " should have been 0");

    for(int i = 0; i < N1_ * N2_ * N3_; ++i)
        vDeriv_[i] /= (N1_ * N2_ * N3_);

    const double factor = growthFactor(); // growth factor, approx (Omega_m)^0.55
    for(int i = 0; i < N1_ * N2_ * N3_; ++i)
    {
        v_[i] *= factor;
        vDeriv_[i] *= factor;
    }

    vCalculated_ = true;
}

double
LymanAlpha3::vDeriv(int m1, int m2, int m3, int n1, int n2, int n3)
{
    check(m1 >= 0 && m1 < N1_, "");
    check(m2 >= 0 && m2 < N2_, "");
    check(m3 >= 0 && m3 < N3_, "");
    check(n1 >= 0 && n1 < N1_, "");
    check(n2 >= 0 && n2 < N2_, "");
    check(n3 >= 0 && n3 < N3_, "");

    if(!vCalculated_)
        calculateV();

    const int delta1 = std::abs(m1 - n1);
    const int delta2 = std::abs(m2 - n2);
    const double d1 = (m3 > n3 ? vDeriv_[(delta1 * N2_ + delta2) * N3_ + (m3 - n3)] : -vDeriv_[(delta1 * N2_ + delta2) * N3_ + (n3 - m3)]);
    const double d2 = std::exp(delta_[(n1 * N2_ + n2) * N3_ + n3]);

    return d1 * d2;
}

void
LymanAlpha3::calculateTau()
{
    check(tau_.size() == N1_ * N2_ * N3_, "");

    if(!vCalculated_)
        calculateV();

    for(int i = 0; i < N1_; ++i)
    {
        for(int j = 0; j < N2_; ++j)
        {
            for(int k = 0; k < N3_; ++k)
                tau_[(i * N2_ + j) * N3_ + k] = 0;

            for(int k = 0; k < N3_; ++k)
            {
                const double x = double(k) * L3_ / N3_;
                const double v = v_[(i * N2_ + j) * N3_ + k];
                const double xv = x + v;
                const double rho = deltaLN_[(i * N2_ + j) * N3_ + k] + 1;
                const double xMin = xv - 5 * b_;
                const double xMax = xv + 5 * b_;
                const int lMin = (int) std::floor(xMin / L3_ * N3_) - 1;
                const int lMax = (int) std::ceil(xMax / L3_ * N3_) + 1;
                for(int l = lMin; l <= lMax; ++l)
                {
                    const int m = periodicIndex(l);
                    const double u = l * L3_ / N3_;
                    const double d = u - xv;
                    tau_[(i * N2_ + j) * N3_ + m] += L3_ / N3_ * rho * rho * std::exp(-d * d / (b_ * b_));
                }
            }
        }
    }

    //calculateTauFactor();

    for(int i = 0; i < N1_ * N2_ * N3_; ++i)
        tau_[i] *= tauFactor_;

    tauCalculated_ = true;
}

void
LymanAlpha3::calculateTauDerivs()
{
    if(!vCalculated_)
        calculateV();

    tauDerivsDeltaX_.resize(N1_ * N2_);
    tauDerivsV_.resize(N1_ * N2_);

    for(int m1 = 0; m1 < N1_ * N2_; ++m1)
    {
        tauDerivsDeltaX_[m1].resize(N3_);
        tauDerivsV_[m1].resize(N3_);

        for(int i = 0; i < N3_; ++i)
        {
            tauDerivsDeltaX_[m1][i].resize(N3_, 0);
            tauDerivsV_[m1][i].resize(N3_, 0);

            // set to 0!
            for(int j = 0; j < N3_; ++j)
            {
                tauDerivsDeltaX_[m1][i][j] = 0;
                tauDerivsV_[m1][i][j] = 0;
            }
        }

        for(int n = 0; n < N3_; ++n)
        {
            const double x = double(n) * L3_ / N3_;
            const double v = v_[m1 * N3_ + n];
            const double xv = x + v;
            const double xMin = xv - 5 * b_;
            const double xMax = xv + 5 * b_;
            const int iMin = (int) std::floor(xMin / L3_ * N3_) - 1;
            const int iMax = (int) std::ceil(xMax / L3_ * N3_) + 1;

            for(int i = iMin; i <= iMax; ++i)
            {
                const int iShifted = periodicIndex(i);
                const double u = i * L3_ / N3_;
                const double d = u - xv;
                tauDerivsDeltaX_[m1][iShifted][n] += 2 * L3_ / N3_ * (deltaLN_[m1 * N3_ + n] + 1) * deltaDeriv(m1 * N3_ + n) * std::exp(-d * d / (b_ * b_));
                tauDerivsV_[m1][iShifted][n] += 2 * L3_ / N3_ * (deltaLN_[m1 * N3_ + n] + 1) * (deltaLN_[m1 * N3_ + n] + 1) * std::exp(-d * d / (b_ * b_)) * d / (b_ * b_);
            }
        }
    }

    tauDerivsCalculated_ = true;
}

/*
void
LymanAlpha3::calculateTauFactor()
{
    tauFactor_ = 1;
    while(true)
    {
        const double f = meanFlux() - 0.8;
        if(std::abs(f) < 1e-5)
            return;

        const double fPrime = meanFluxDeriv();

        // means the field is 0 everywhere
        if(fPrime == 0)
            return;

        tauFactor_ -= f / fPrime;
        if(tauFactor_ < 0)
            tauFactor_ = 0;
    }
}

double
LymanAlpha3::meanFlux() const
{
    double s = 0;
    for(int i = 0; i < tau_.size(); ++i)
        s += std::exp(-tauFactor_ * tau_[i]);

    return s / tau_.size();
}

double
LymanAlpha3::meanFluxDeriv() const
{
    double s = 0;
    for(int i = 0; i < tau_.size(); ++i)
        s -= tau_[i] * std::exp(-tauFactor_ * tau_[i]);

    return s / tau_.size();
}
*/

void
LymanAlpha3::calculateFlux()
{
    check(flux_.size() == N1_ * N2_ * N3_, "");
    
    if(!tauCalculated_)
        calculateTau();

    for(int i = 0; i < N1_ * N2_ * N3_; ++i)
        flux_[i] = std::exp(-tau_[i]);

    fluxCalculated_ = true;
}

double
LymanAlpha3::tauDeriv(int m1, int m2, int m3, int n1, int n2, int n3)
{
    check(m1 >= 0 && m1 < N1_, "");
    check(m2 >= 0 && m2 < N2_, "");
    check(m3 >= 0 && m3 < N3_, "");
    check(n1 >= 0 && n1 < N1_, "");
    check(n2 >= 0 && n2 < N2_, "");
    check(n3 >= 0 && n3 < N3_, "");

    if(!tauDerivsCalculated_)
        calculateTauDerivs();

    double res = 0;
    if(m1 == n1 && m2 == n2)
        res += tauDerivsDeltaX_[m1 * N2_ + m2][m3][n3];

    for(int i = 0; i < N3_; ++i)
        res += tauDerivsV_[m1 * N2_ + m2][m3][i] * vDeriv(m1, m2, i, n1, n2, n3);

    return res * tauFactor_;
}

double LymanAlpha3::tauDerivV(int m1, int m2, int m3, int k)
{
    check(m1 >= 0 && m1 < N1_, "");
    check(m2 >= 0 && m2 < N2_, "");
    check(m3 >= 0 && m3 < N3_, "");
    check(k >= 0 && k < N3_, "");

    if(!tauDerivsCalculated_)
        calculateTauDerivs();

    return tauDerivsV_[m1 * N2_ + m2][m3][k] * tauFactor_;
}

double LymanAlpha3::tauDerivDeltaOnly(int m1, int m2, int m3, int k)
{
    check(m1 >= 0 && m1 < N1_, "");
    check(m2 >= 0 && m2 < N2_, "");
    check(m3 >= 0 && m3 < N3_, "");
    check(k >= 0 && k < N3_, "");

    if(!tauDerivsCalculated_)
        calculateTauDerivs();

    return tauDerivsDeltaX_[m1 * N2_ + m2][m3][k] * tauFactor_;
}

double LymanAlpha3::fluxDeriv(int m1, int m2, int m3, int n1, int n2, int n3)
{
    if(!fluxCalculated_)
        calculateFlux();

    return -flux_[(m1 * N2_ + m2) * N3_ + m3] * tauDeriv(m1, m2, m3, n1, n2, n3);
}

double LymanAlpha3::fluxDerivV(int m1, int m2, int m3, int k)
{
    if(!fluxCalculated_)
        calculateFlux();

    return -flux_[(m1 * N2_ + m2) * N3_ + m3] * tauDerivV(m1, m2, m3, k);
}

double LymanAlpha3::fluxDerivDeltaOnly(int m1, int m2, int m3, int k)
{
    if(!fluxCalculated_)
        calculateFlux();

    return -flux_[(m1 * N2_ + m2) * N3_ + m3] * tauDerivDeltaOnly(m1, m2, m3, k);
}
