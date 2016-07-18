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

#include <fftw3.h>

namespace
{

class DeltaVector
{
public:
    DeltaVector(int n = 0, bool isComplex = false) : isComplex_(isComplex), x_(isComplex ? 0 : n), c_(isComplex ? n / 2 + 1 : 0) {}

    std::vector<double>& get() { return x_; }
    const std::vector<double>& get() const { return x_; }

    std::vector<std::complex<double> >& getComplex() { return c_; }
    const std::vector<std::complex<double> >& getComplex() const { return c_; }

    bool isComplex() const { return isComplex_; }

    // copy from other, multiplying with coefficient (for MPI, the correct coefficient should be passed for EVERY process)
    void copy(const DeltaVector& other, double c = 1.)
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
    double dotProduct(const DeltaVector& other) const
    {
        double res = 0;
        if(isComplex_)
        {
            check(other.c_.size() == c_.size(), "");
            for(int i = 0; i < c_.size(); ++i)
                res += (std::real(c_[i]) * std::real(other.c_[i]) + std::imag(c_[i]) * std::imag(other.c_[i]));
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
    void add(const DeltaVector& other, double c = 1.)
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
    void swap(DeltaVector& other)
    {
        check(other.c_.size() == c_.size(), "");
        c_.swap(other.c_);

        check(other.x_.size() == x_.size(), "");
        x_.swap(other.x_);
    }

private:
    bool isComplex_;
    std::vector<double> x_;
    std::vector<std::complex<double> > c_;
};

class DeltaVectorFactory
{
public:
    DeltaVectorFactory(int N, bool isComplex) : N_(N), isComplex_(isComplex)
    {
        check(N_ > 0, "");
    }

    // create a new DeltaVector with 0 elements
    // the user is in charge of deleting it
    DeltaVector* giveMeOne()
    {
        return new DeltaVector(N_, isComplex_);
    }
private:
    const int N_;
    const bool isComplex_;
};

class LBFGSFunction
{
public:
    LBFGSFunction(const std::vector<double>& data, const std::vector<double>& sigma, const std::vector<double>& pk, double L, double b, bool useFlux = false) : data_(data), sigma_(sigma), pk_(pk), L_(L), la_(std::vector<double>(data.size()), L, b), useFlux_(useFlux)
    {
        const int N = data_.size();
        deltaK_.resize(N / 2 + 1);
        pkDeriv_.resize(N / 2 + 1);
        likeXDeriv_.resize(N);
        buf_ = fftw_malloc(sizeof(fftw_complex) * (N / 2  + 1));
        check(sigma_.size() == N, "");
#ifdef CHECKS_ON
        for(int i = 0; i < N; ++i)
        {
            check(sigma_[i] > 0, "");
        }
#endif
        check(pk_.size() == N / 2 + 1, "");
        check(L_ > 0, "");
    }

    ~LBFGSFunction()
    {
        fftw_free(buf_);
    }

    void set(const DeltaVector& x)
    {
        const int N = data_.size();
        isComplex_ = x.isComplex();
        if(isComplex_)
        {
            deltaK_ = x.getComplex();
            deltaK2deltaX(N, deltaK_, &deltaX_, L_);
            la_.reset(deltaX_);
            return;
        }
        la_.reset(x.get());
    }

    double value()
    {
        const int N = data_.size();
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
            total += priorK(N, pk_, deltaK_);
        else
            total += priorX(N, pk_, la_.getDelta(), L_, &deltaK_);

        return total;
    }

    void derivative(DeltaVector *res)
    {
        check(isComplex_ == res->isComplex(), "");
        const int N = data_.size();
        if(isComplex_)
        {
            std::vector<std::complex<double> >& r = res->getComplex();
            priorKDeriv(N, pk_, deltaK_, &r);
            for(int i = 0; i < N; ++i)
            {
                if(useFlux_)
                {
                    likeXDeriv_[i] = 0;
                    for(int j = 0; j < N; ++j)
                    {
                        const double s = sigma_[j];
                        const double delta = la_.getFlux()[j] - data_[j];
                        likeXDeriv_[i] += 2 * delta * la_.fluxDeriv(j, i) / (s * s);
                    }
                }
                else
                {
                    const double s = sigma_[i];
                    const double delta = la_.getDeltaNonLin()[i] - data_[i];
                    likeXDeriv_[i] = 2 * delta * la_.deltaDeriv(i) / (s * s);
                }
            }
            fftw_plan fwdPlan = fftw_plan_dft_r2c_1d(N, &(likeXDeriv_[0]), (fftw_complex*)buf_, FFTW_ESTIMATE);
            fftw_execute(fwdPlan);
            fftw_destroy_plan(fwdPlan);

            for(int i = 0; i < N / 2 + 1; ++i)
            {
                r[i] += std::complex<double>(((fftw_complex*)buf_)[i][0], ((fftw_complex*)buf_)[i][1]) / L_ * (i == 0 || i == N / 2 ? 1.0 : 2.0); // the last division is to take into account conjugates
            }
        }
        else
        {
            std::vector<double>& r = res->get();
            priorXDeriv(N, pk_, la_.getDelta(), L_, &r, &deltaK_, &pkDeriv_);
            check(r.size() == N, "");
            for(int i = 0; i < N; ++i)
            {
                if(useFlux_)
                {
                    for(int j = 0; j < N; ++j)
                    {
                        const double s = sigma_[j];
                        const double delta = la_.getFlux()[j] - data_[j];
                        r[i] += 2 * delta * la_.fluxDeriv(j, i) / (s * s);
                    }
                }
                else
                {
                    const double s = sigma_[i];
                    const double delta = la_.getDeltaNonLin()[i] - data_[i];
                    r[i] += 2 * delta * la_.deltaDeriv(i) / (s * s);
                }
            }
        }
    }

private:
    const std::vector<double> data_;
    const std::vector<double> sigma_;
    const std::vector<double> pk_;
    const double L_;
    const bool useFlux_;
    LymanAlpha la_;
    std::vector<std::complex<double> > deltaK_, pkDeriv_;
    std::vector<double> deltaX_;
    std::vector<double> likeXDeriv_;
    void *buf_;
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
    int N = 128;
    const double L = 100;
    const double b = 0.1;

    output_screen("Specify N as an argument or else it will be 128 by default. Specify \"fourier\" to do the fit in Fourier space. Specify \"flux\" as an argument to use the flux data instead of the density data. Specify \"out\" as an argument to write the iterations into a file. Specify \"random_start\" to start from a random point instead of 0. Specify \"many\" to do many runs of LBFGS with different starting points. Specify \"hmc\" as an argument to run hmc instead of lbfgs." << std::endl);

    if(argc > 1)
    {
        std::stringstream str;
        str << argv[1];
        str >> N;
        if(N < 1)
        {
            output_screen("Invalid argument " << argv[1] << std::endl);
            N = 128;
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
    ps.normalize(4.0, 1.0);
    std::vector<double> pk;
    discretePowerSpectrum(ps, L, N, &pk);
    check(pk.size() == N / 2 + 1, "");

    // make sure there are no 0 elements in pk
    // TBD better
    for(int i = 0; i < pk.size(); ++i)
    {
        if(pk[i] == 0)
            pk[i] = 1e-5;
    }

    std::ofstream out("pk.txt");
    for(int i = 0; i < pk.size(); ++i)
    {
        const double k = 2 * Math::pi / L * i;
        out << k << '\t' << pk[i] << std::endl;
    }
    out.close();

    int seed = 100;
    std::vector<std::complex<double> > deltaK;
    generateDeltaK(N, pk, &deltaK, seed);
    std::vector<double> deltaX;
    void *buf = fftw_malloc(sizeof(fftw_complex) * (N / 2  + 1));
    deltaK2deltaX(N, deltaK, &deltaX, L);
    check(deltaX.size() == N, "");

    LymanAlpha la(deltaX, L, b);
    out.open("lyman_alpha.txt");
    for(int i = 0; i < N; ++i)
        out << i * L / N << '\t' << la.getDelta()[i] << '\t' << la.getDeltaNonLin()[i] << '\t' << la.getV()[i] << '\t' << la.getTau()[i] << '\t' << la.getFlux()[i] << std::endl;
    out.close();

    std::vector<double> data = (flux ? la.getFlux() : la.getDeltaNonLin());
    std::vector<double> sigma(N);
    for(int i = 0; i < N; ++i)
    {
        sigma[i] = std::abs(data[i]) / 20;

        if(sigma[i] < 1e-3)
            sigma[i] = 1e-3;
    }

    seed = 200;
    Math::GaussianGenerator g1(seed, 0.0, 1.0);

    for(int i = 0; i < N; ++i)
        data[i] += 0 * sigma[i] * g1.generate(); // no noise added for now

    // lbfgs stuff
    DeltaVectorFactory factory(N, isComplex);
    LBFGSFunction f(data, sigma, pk, L, b, flux);

    output_screen("Testing the function and derivatives..." << std::endl);
    seed = 300;
    std::vector<std::complex<double> > deltaKTest;
    generateDeltaK(N, pk, &deltaKTest, seed);
    std::vector<double> deltaXTest;
    deltaK2deltaX(N, deltaKTest, &deltaXTest, L);
    check(deltaXTest.size() == N, "");
    std::unique_ptr<DeltaVector> testX(factory.giveMeOne());
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
    std::unique_ptr<DeltaVector> testDerivs(factory.giveMeOne());
    Timer timer2("TEST FUNCTION DERIVATIVES");
    timer2.start();
    f.derivative(testDerivs.get());
    const unsigned long timer2Duration = timer2.end();
    output_screen("Function derivatives calculation took " << timer2Duration << " microseconds." << std::endl);
    double epsilon = 0.000001;
    if(isComplex)
    {
        for(int i = 0; i < N / 2 + 1; ++i)
        {
            // real part
            testX->getComplex() = deltaKTest;
            testX->getComplex()[i] += std::complex<double>(1.0, 0.0) * epsilon;
            f.set(*testX);
            double pertVal = f.value();
            double numDeriv = (pertVal - testVal) / epsilon;
            if(!Math::areEqual(numDeriv, std::real(testDerivs->getComplex()[i]), 1e-3))
            {
                output_screen("PROBLEM: index " << i << " real part. Numerical derivative = " << numDeriv << ", analytic = " << std::real(testDerivs->getComplex()[i]) << std::endl);
            }
            
            // imaginary part
            if(i == 0 || i == N / 2)
                continue;

            testX->getComplex() = deltaKTest;
            testX->getComplex()[i] += std::complex<double>(0.0, 1.0) * epsilon;
            f.set(*testX);
            pertVal = f.value();
            numDeriv = (pertVal - testVal) / epsilon;
            if(!Math::areEqual(numDeriv, std::imag(testDerivs->getComplex()[i]), 1e-3))
            {
                output_screen("PROBLEM: index " << i << " imag part. Numerical derivative = " << numDeriv << ", analytic = " << std::imag(testDerivs->getComplex()[i]) << std::endl);
            }
        }
    }
    else
    {
        for(int i = 0; i < N; ++i)
        {
            testX->get() = deltaXTest;
            testX->get()[i] += epsilon;
            f.set(*testX);
            const double pertVal = f.value();
            const double numDeriv = (pertVal - testVal) / epsilon;
            if(!Math::areEqual(numDeriv, testDerivs->get()[i], 1e-3))
            {
                output_screen("PROBLEM: index " << i << " numerical derivative = " << numDeriv << ", analytic = " << testDerivs->get()[i] << std::endl);
            }
        }
    }
    output_screen("OK" << std::endl);

    std::unique_ptr<DeltaVector> x(factory.giveMeOne());

    if(randomStart)
    {
        std::vector<std::complex<double> > deltaKStart;
        generateDeltaK(N, pk, &deltaKStart);
        std::vector<double> deltaXStart;
        deltaK2deltaX(N, deltaKStart, &deltaXStart, L);
        check(deltaXStart.size() == N, "");
        if(isComplex)
            x->getComplex() = deltaKStart;
        else
            x->get() = deltaXStart;
    }

    typedef Math::LBFGS_General<DeltaVector, DeltaVectorFactory, LBFGSFunction> MyLBFGS;
    MyLBFGS lbfgs(&factory, &f, *x, 10);
    epsilon = 1e-8;
    const double gradTol = 1e-5 * N * CosmoMPI::create().numProcesses();
    double minVal = lbfgs.minimize(x.get(), epsilon, gradTol, 10000000);

    std::vector<double> deltaXMin(N);
    if(isComplex)
    {
        check(x->getComplex().size() == N / 2 + 1, "");
        deltaK2deltaX(N, x->getComplex(), &deltaXMin, L);
    }
    else
    {
        check(x->get().size() == N, "");
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
            generateDeltaK(N, pk, &deltaKStart, 0);
            //generateDeltaK(pk, &deltaKStart, seed++);
            deltaK2deltaX(N, deltaKStart, &deltaXStart, L);
            check(deltaXStart.size() == N, "");
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
                    check(x->getComplex().size() == N / 2 + 1, "");
                    deltaK2deltaX(N, x->getComplex(), &deltaXMin, L);
                }
                else
                {
                    check(x->get().size() == N, "");
                    deltaXMin = x->get();
                }
            }
        }
    }

    output_screen("MINIMUM VALUE FOUND = " << minVal << std::endl);

    out.open("delta_x.txt");
    for(int i = 0; i < N; ++i)
        out << i * L / N << '\t' << la.getDelta()[i] << '\t' << la.getDeltaNonLin()[i] << '\t' << data[i] << '\t' << deltaXMin[i] << std::endl;
    out.close();

    la.reset(deltaXMin);
    out.open("lyman_alpha_min.txt");
    for(int i = 0; i < N; ++i)
        out << i * L / N << '\t' << la.getDelta()[i] << '\t' << la.getDeltaNonLin()[i] << '\t' << la.getV()[i] << '\t' << la.getTau()[i] << '\t' << la.getFlux()[i] << std::endl;
    out.close();


    fftw_free(buf);
    return 0;
}

