#include <vector>
#include <ctime>
#include <cmath>
#include <complex>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <sstream>

#include <macros.hpp>
#include <exception_handler.hpp>
#include <lbfgs_general.hpp>
#include <random.hpp>
#include <timer.hpp>
#include <math_constants.hpp>
#include <numerics.hpp>
#include <cosmo_mpi.hpp>

#include "lyman_alpha.hpp"
#include "power_spectrum.hpp"
#include "utils.hpp"

#include <fftw3.h>

namespace
{

class DeltaVector2
{
public:
    DeltaVector2(int n1 = 0, int n2 = 0, bool isComplex = false) : isComplex_(isComplex), x_(isComplex ? 0 : n1 * n2), c_(isComplex ? n1 * (n2 / 2 + 1) : 0), n1_(n1), n2_(n2) {}

    std::vector<double>& get() { return x_; }
    const std::vector<double>& get() const { return x_; }

    std::vector<std::complex<double> >& getComplex() { return c_; }
    const std::vector<std::complex<double> >& getComplex() const { return c_; }

    bool isComplex() const { return isComplex_; }

    // copy from other, multiplying with coefficient (for MPI, the correct coefficient should be passed for EVERY process)
    void copy(const DeltaVector2& other, double c = 1.)
    {
        if(isComplex_)
        {
            check(other.c_.size() == c_.size(), "");
            for(int i = 0; i < c_.size(); ++i)
                c_[i] = c * other.c_[i];

            return;
        }

        check(other.x_.size() == x_.size(), "");
        for(int i = 0; i < x_.size(); ++i)
            x_[i] = c * other.x_[i];
    }

    // set all the elements to 0
    void setToZero()
    {
        for(auto it = x_.begin(); it != x_.end(); ++it)
            *it = 0;

        for(auto it = c_.begin(); it != c_.end(); ++it)
            *it = std::complex<double>(0, 0);
    }

    // get the norm (for MPI, the master process should get the total norm)
    double norm() const
    {
        return std::sqrt(dotProduct(*this));
    }

    // dot product with another vector (for MPI, the master process should get the total norm)
    double dotProduct(const DeltaVector2& other) const
    {
        double res = 0;
        if(isComplex_)
        {
            check(other.n1_ == n1_, "");
            check(other.n2_ == n2_, "");
            for(int j = 0; j < n2_ / 2 + 1; ++j)
            {
                const int iMax = (j > 0 && j < n2_ / 2 ? n1_ : n1_ / 2 + 1);
                for(int i = 0; i < iMax; ++i)
                {
                    const int index = i * (n2_ / 2 + 1) + j;
                    res += (std::real(c_[index]) * std::real(other.c_[index]) + std::imag(c_[index]) * std::imag(other.c_[index]));
                }
            }
        }
        else
        {
            check(other.x_.size() == x_.size(), "");
            for(int i = 0; i < x_.size(); ++i)
                res += x_[i] * other.x_[i];
        }

        double total = res;
#ifdef COSMO_MPI
        CosmoMPI::create().reduce(&res, &total, 1, CosmoMPI::DOUBLE, CosmoMPI::SUM);
#endif
        return total;
    }

    // add another vector with a given coefficient (for MPI, the correct coefficient should be passed for EVERY process)
    void add(const DeltaVector2& other, double c = 1.)
    {
        if(isComplex_)
        {
            check(other.c_.size() == c_.size(), "");
            for(int i = 0; i < c_.size(); ++i)
                c_[i] += c * other.c_[i];

            return;
        }

        check(other.x_.size() == x_.size(), "");
        for(int i = 0; i < x_.size(); ++i)
            x_[i] += c * other.x_[i];
    }

    // swap
    void swap(DeltaVector2& other)
    {
        check(other.c_.size() == c_.size(), "");
        c_.swap(other.c_);

        check(other.x_.size() == x_.size(), "");
        x_.swap(other.x_);
    }

private:
    bool isComplex_;
    const int n1_, n2_;
    std::vector<double> x_;
    std::vector<std::complex<double> > c_;
};

class DeltaVector2Factory
{
public:
    DeltaVector2Factory(int N1, int N2, bool isComplex) : N1_(N1), N2_(N2), isComplex_(isComplex)
    {
        check(N1_ > 0, "");
        check(N2_ > 0, "");
    }

    // create a new DeltaVector with 0 elements
    // the user is in charge of deleting it
    DeltaVector2* giveMeOne()
    {
        return new DeltaVector2(N1_, N2_, isComplex_);
    }
private:
    const int N1_, N2_;
    const bool isComplex_;
};

class LBFGSFunction2
{
public:
    LBFGSFunction2(int N1, int N2, const std::vector<double>& data, const std::vector<double>& sigma, const std::vector<double>& pk, double L1, double L2, double b, bool useFlux = false) : N1_(N1), N2_(N2), data_(data), sigma_(sigma), pk_(pk), L1_(L1), L2_(L2), la_(N1, N2, std::vector<double>(data.size()), L1, L2, b), useFlux_(useFlux)
    {
        check(N1 > 0, "");
        check(N2 > 0, "");
        check(L1_ > 0, "");
        check(L2_ > 0, "");

        deltaK_.resize(N1 * (N2 / 2 + 1));
        pkDeriv_.resize(N1 * (N2 / 2 + 1));
        likeXDeriv_.resize(N1 * N2);
        buf_.resize(N1 * (N2 / 2 + 1));
        check(sigma_.size() == N1 * N2, "");
#ifdef CHECKS_ON
        for(int i = 0; i < N1 * N2; ++i)
        {
            check(sigma_[i] > 0, "");
        }
#endif
        check(pk_.size() == N1 * (N2 / 2 + 1), "");
    }

    void set(const DeltaVector2& x)
    {
        isComplex_ = x.isComplex();
        if(isComplex_)
        {
            deltaK_ = x.getComplex();
            deltaK2deltaX(N1_, N2_, deltaK_, &deltaX_, L1_, L2_);
            la_.reset(deltaX_);
            return;
        }
        la_.reset(x.get());
    }

    double value()
    {
        double r = 0;
        for(int i = 0; i < data_.size(); ++i)
        {
            const double s = sigma_[i];
            const double delta = (useFlux_ ? la_.getFlux()[i] : la_.getDeltaNonLin()[i]) - data_[i];
            r += delta * delta / (s * s);
        }

        double total = r;
#ifdef COSMO_MPI
        CosmoMPI::create().reduce(&r, &total, 1, CosmoMPI::DOUBLE, CosmoMPI::SUM);
#endif
        if(isComplex_)
            total += priorK(N1_, N2_, pk_, deltaK_);
        else
            total += priorX(N1_, N2_, pk_, la_.getDelta(), L1_, L2_, &deltaK_);

        return total;
    }

    void derivative(DeltaVector2 *res)
    {
        check(isComplex_ == res->isComplex(), "");
        if(isComplex_)
        {
            std::vector<std::complex<double> >& r = res->getComplex();
            priorKDeriv(N1_, N2_, pk_, deltaK_, &r);
            for(int i = 0; i < N1_; ++i)
            {
                for(int j = 0; j < N2_; ++j)
                {
                    if(useFlux_)
                    {
                        likeXDeriv_[i] = 0;
                        for(int k = 0; k < N1_; ++k)
                        {
                            for(int l = 0; l < N2_; ++l)
                            {
                                const double s = sigma_[k * N2_ + l];
                                const double delta = la_.getFlux()[k * N2_ + l] - data_[k * N2_ + l];
                                likeXDeriv_[i * N2_ + j] += 2 * delta * la_.fluxDeriv(k, l, i, j) / (s * s);
                            }
                        }
                    }
                    else
                    {
                        const double s = sigma_[i * N2_ + j];
                        const double delta = la_.getDeltaNonLin()[i * N2_ + j] - data_[i * N2_ + j];
                        likeXDeriv_[i * N2_ + j] = 2 * delta * la_.deltaDeriv(i * N2_ + j) / (s * s);
                    }
                }
            }

            deltaX2deltaK(N1_, N2_, likeXDeriv_, &buf_, L1_, L2_);

            for(int i = 0; i < N1_; ++i)
            {
                for(int j = 0; j < N2_ / 2 + 1; ++j)
                {
                    double factor = 2; // for conjugates
                    if((j == 0 || j == N2_ / 2) && (i == 0 || i == N2_ / 2))
                        factor = 1;
                    r[i * (N2_ / 2 + 1) + j] += buf_[i * (N2_ / 2 + 1) + j] * double(N1_ * N2_) / (L1_ * L1_ * L2_ * L2_) * factor;
                }
            }
        }
        else
        {
            std::vector<double>& r = res->get();
            priorXDeriv(N1_, N2_, pk_, la_.getDelta(), L1_, L2_, &r, &deltaK_, &pkDeriv_);
            check(r.size() == N1_ * N2_, "");
            for(int i = 0; i < N1_; ++i)
            {
                for(int j = 0; j < N2_; ++j)
                {
                    if(useFlux_)
                    {
                        for(int k = 0; k < N1_; ++k)
                        {
                            for(int l = 0; l < N2_; ++l)
                            {
                                const double s = sigma_[k * N2_ + l];
                                const double delta = la_.getFlux()[k * N2_ + l] - data_[k * N2_ + l];
                                r[i * N2_ + j] += 2 * delta * la_.fluxDeriv(k, l, i, j) / (s * s);
                            }
                        }
                    }
                    else
                    {
                        const double s = sigma_[i * N2_ + j];
                        const double delta = la_.getDeltaNonLin()[i * N2_ + j] - data_[i * N2_ + j];
                        r[i * N2_ + j] += 2 * delta * la_.deltaDeriv(i * N2_ + j) / (s * s);
                    }
                }
            }
        }
    }

private:
    const int N1_, N2_;
    const std::vector<double> data_;
    const std::vector<double> sigma_;
    const std::vector<double> pk_;
    const double L1_, L2_;
    const bool useFlux_;
    LymanAlpha2 la_;
    std::vector<std::complex<double> > deltaK_, pkDeriv_;
    std::vector<double> deltaX_;
    std::vector<double> likeXDeriv_;
    std::vector<std::complex<double> > buf_;
    bool isComplex_;
};

void lbfgsCallbackFunc(int iter, double f, double gradNorm, const std::vector<double>& x)
{
    std::stringstream fileName;
    fileName << "lbfgs_iters";
    if(CosmoMPI::create().numProcesses() > 1)
        fileName << "_" << CosmoMPI::create().processId();
    fileName << ".txt";
    std::ofstream out(fileName.str().c_str(), std::ios::app);
    out << f << ' ' << gradNorm; 
    for(int i = 0; i < x.size(); ++i)
        out << ' ' << x[i];
    out << std::endl;
    out.close();
}

} // namespace

int main(int argc, char *argv[])
{
    int N = 32;
    const double L = 32;
    const double b = 0.1;

    output_screen("Specify N as an argument or else it will be 32 by default. Specify \"fourier\" to do the fit in Fourier space. Specify \"flux\" as an argument to use the flux data instead of the density data. Specify \"out\" as an argument to write the iterations into a file. Specify \"random_start\" to start from a random point instead of 0. Specify \"many\" to do many runs of LBFGS with different starting points. Specify \"hmc\" as an argument to run hmc instead of lbfgs." << std::endl);

    if(argc > 1)
    {
        std::stringstream str;
        str << argv[1];
        str >> N;
        if(N < 1)
        {
            output_screen("Invalid argument " << argv[1] << std::endl);
            N = 32;
        }
    }

    bool outIters = false;
    bool hmc = false;
    bool flux = false;
    bool isComplex = false;
    bool randomStart = false;
    bool many = false;
    for(int i = 1; i < argc; ++i)
    {
        if(std::string(argv[i]) == std::string("out"))
            outIters = true;
        if(std::string(argv[i]) == std::string("hmc"))
            hmc = true;
        if(std::string(argv[i]) == std::string("flux"))
            flux = true;
        if(std::string(argv[i]) == std::string("fourier"))
            isComplex = true;
        if(std::string(argv[i]) == std::string("random_start"))
            randomStart = true;
        if(std::string(argv[i]) == std::string("many"))
            many = true;
    }

    SimplePowerSpectrum ps(0.015);
    

    // hack (for now)
    ps.normalize2(4.0, 0.25);
    std::vector<double> pk;
    discretePowerSpectrum(ps, L, L, N, N, &pk);
    check(pk.size() == N * (N / 2 + 1), "");

    // make sure there are no 0 elements in pk
    // TBD better
    for(int i = 0; i < pk.size(); ++i)
    {
        if(pk[i] == 0)
            pk[i] = 1e-5;
    }

    int seed = 100;
    std::vector<std::complex<double> > deltaK;
    generateDeltaK(N, N, pk, &deltaK, seed);
    std::vector<double> deltaX;
    deltaK2deltaX(N, N, deltaK, &deltaX, L, L);
    check(deltaX.size() == N * N, "");

    LymanAlpha2 la(N, N, deltaX, L, L, b);

    std::vector<double> data = (flux ? la.getFlux() : la.getDeltaNonLin());
    std::vector<double> sigma(N * N);
    for(int i = 0; i < N * N; ++i)
    {
        sigma[i] = std::abs(data[i]) / 20;

        if(sigma[i] < 1e-3)
            sigma[i] = 1e-3;
    }

    vector2binFile("la_delta_x2.dat", la.getDelta());
    vector2binFile("la_delta_x2_nl.dat", la.getDeltaNonLin());
    vector2binFile("la_data2.dat", data);
    vector2binFile("la_v2.dat", la.getV());
    vector2binFile("la_tau2.dat", la.getTau());
    vector2binFile("la_flux2.dat", la.getFlux());

    seed = 200;
    Math::GaussianGenerator g1(seed, 0.0, 1.0);

    for(int i = 0; i < N * N; ++i)
        data[i] += 0 * sigma[i] * g1.generate(); // no noise added for now

    // lbfgs stuff
    DeltaVector2Factory factory(N, N, isComplex);
    LBFGSFunction2 f(N, N, data, sigma, pk, L, L, b, flux);

    output_screen("Testing the function and derivatives..." << std::endl);
    seed = 300;
    std::vector<std::complex<double> > deltaKTest;
    generateDeltaK(N, N, pk, &deltaKTest, seed);
    std::vector<double> deltaXTest;
    deltaK2deltaX(N, N, deltaKTest, &deltaXTest, L, L);
    check(deltaXTest.size() == N * N, "");
    std::unique_ptr<DeltaVector2> testX(factory.giveMeOne());
    if(testX->isComplex())
        testX->getComplex() = deltaKTest;
    else
        testX->get() = deltaXTest;
    f.set(*testX);
    Timer timer1("TEST FUNCTION VALUE");
    timer1.start();
    const double testVal = f.value();
    const unsigned long timer1Duration = timer1.end();
    output_screen("Function calculation took " << timer1Duration << " microseconds." << std::endl);
    std::unique_ptr<DeltaVector2> testDerivs(factory.giveMeOne());
    Timer timer2("TEST FUNCTION DERIVATIVES");
    timer2.start();
    f.derivative(testDerivs.get());
    const unsigned long timer2Duration = timer2.end();
    output_screen("Function derivatives calculation took " << timer2Duration << " microseconds." << std::endl);
    double epsilon = 0.0000001;
    if(isComplex)
    {
        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N / 2 + 1; ++j)
            {
                // real part
                testX->getComplex() = deltaKTest;
                testX->getComplex()[i * (N / 2 + 1) + j] += std::complex<double>(1.0, 0.0) * epsilon;
                if((j == 0 || j == N / 2) && (i != 0 && i != N / 2))
                {
                    const int index1 = (N - i) * (N / 2 + 1) + j;
                    testX->getComplex()[index1] += epsilon * std::complex<double>(1.0, 0.0);
                }
                f.set(*testX);
                double pertVal = f.value();
                double numDeriv = (pertVal - testVal) / epsilon;
                if(!Math::areEqual(numDeriv, std::real(testDerivs->getComplex()[i * (N / 2 + 1) + j]), 1e-1))
                {
                    output_screen("PROBLEM: index (" << i << ", " << j << ") real part. Numerical derivative = " << numDeriv << ", analytic = " << std::real(testDerivs->getComplex()[i * (N / 2 + 1) + j]) << std::endl);
                }
                
                // imaginary part
                if((j == 0 || j == N / 2) && (i == 0 || i == N / 2))
                    continue;

                testX->getComplex() = deltaKTest;
                testX->getComplex()[i * (N / 2 + 1) + j] += std::complex<double>(0.0, 1.0) * epsilon;
                if((j == 0 || j == N / 2) && (i != 0 && i != N / 2))
                {
                    const int index1 = (N - i) * (N / 2 + 1) + j;
                    testX->getComplex()[index1] -= epsilon * std::complex<double>(0.0, 1.0);
                }
                f.set(*testX);
                pertVal = f.value();
                numDeriv = (pertVal - testVal) / epsilon;
                if(!Math::areEqual(numDeriv, std::imag(testDerivs->getComplex()[i * (N / 2 + 1) + j]), 1e-1))
                {
                    output_screen("PROBLEM: index (" << i << ", " << j << ") imag part. Numerical derivative = " << numDeriv << ", analytic = " << std::imag(testDerivs->getComplex()[i * (N / 2 + 1) + j]) << std::endl);
                }
            }
        }
    }
    else
    {
        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                testX->get() = deltaXTest;
                testX->get()[i * N + j] += epsilon;
                f.set(*testX);
                const double pertVal = f.value();
                const double numDeriv = (pertVal - testVal) / epsilon;
                if(!Math::areEqual(numDeriv, testDerivs->get()[i * N + j], 1e-1))
                {
                    output_screen("PROBLEM: index (" << i << ", " << j << ") numerical derivative = " << numDeriv << ", analytic = " << testDerivs->get()[i * N + j] << std::endl);
                }
            }
        }
    }
    output_screen("OK" << std::endl);

    std::unique_ptr<DeltaVector2> x(factory.giveMeOne());

    if(randomStart)
    {
        std::vector<std::complex<double> > deltaKStart;
        generateDeltaK(N, N, pk, &deltaKStart);
        std::vector<double> deltaXStart;
        deltaK2deltaX(N, N, deltaKStart, &deltaXStart, L, L);
        check(deltaXStart.size() == N * N, "");
        if(isComplex)
            x->getComplex() = deltaKStart;
        else
            x->get() = deltaXStart;
    }

    typedef Math::LBFGS_General<DeltaVector2, DeltaVector2Factory, LBFGSFunction2> MyLBFGS;
    MyLBFGS lbfgs(&factory, &f, *x, 10);
    epsilon = 1e-8;
    const double gradTol = 1e-5 * N * N * CosmoMPI::create().numProcesses();
    double minVal = lbfgs.minimize(x.get(), epsilon, gradTol, 10000000);

    std::vector<double> deltaXMin(N * N);
    if(isComplex)
    {
        check(x->getComplex().size() == N * (N / 2 + 1), "");
        deltaK2deltaX(N, N, x->getComplex(), &deltaXMin, L, L);
    }
    else
    {
        check(x->get().size() == N * N, "");
        deltaXMin = x->get();
    }

    if(many)
    {
        const int nRuns = 100;
        std::vector<std::complex<double> > deltaKStart;
        std::vector<double> deltaXStart;
        int seed = 10000;
        for(int i = 0; i < nRuns; ++i)
        {
            generateDeltaK(N, N, pk, &deltaKStart, 0);
            //generateDeltaK(pk, &deltaKStart, seed++);
            deltaK2deltaX(N, N, deltaKStart, &deltaXStart, L, L);
            check(deltaXStart.size() == N * N, "");
            if(isComplex)
                x->getComplex() = deltaKStart;
            else
                x->get() = deltaXStart;
            lbfgs.setStarting(*x);
            double val = lbfgs.minimize(x.get(), epsilon, gradTol, 10000000);
            if(val < minVal)
            {
                minVal = val;
                if(isComplex)
                {
                    check(x->getComplex().size() == N * (N / 2 + 1), "");
                    deltaK2deltaX(N, N, x->getComplex(), &deltaXMin, L, L);
                }
                else
                {
                    check(x->get().size() == N * N, "");
                    deltaXMin = x->get();
                }
            }
        }
    }

    output_screen("MINIMUM VALUE FOUND = " << minVal << std::endl);

    vector2binFile("la_min2.dat", deltaXMin);

    la.reset(deltaXMin);

    vector2binFile("la_min_v2.dat", la.getV());
    vector2binFile("la_min_tau2.dat", la.getTau());
    vector2binFile("la_min_flux2.dat", la.getFlux());
    return 0;
}

