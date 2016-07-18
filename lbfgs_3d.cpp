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
#include <conjugate_gradient_general.hpp>
#include <random.hpp>
#include <timer.hpp>
#include <math_constants.hpp>
#include <numerics.hpp>
#include <cosmo_mpi.hpp>
#include <parser.hpp>

#include "lyman_alpha.hpp"
#include "power_spectrum.hpp"
#include "utils.hpp"
#include "delta_vector.hpp"

#include <fftw3.h>

namespace
{

class LBFGSFunction3
{
public:
    LBFGSFunction3(int N1, int N2, int N3, const std::vector<double>& data, const std::vector<double>& sigma, const std::vector<double>& pk, double L1, double L2, double L3, double b, bool useFlux = false) : N1_(N1), N2_(N2), N3_(N3), data_(data), sigma_(sigma), pk_(pk), L1_(L1), L2_(L2), L3_(L3), la_(N1, N2, N3, std::vector<double>(data.size()), L1, L2, L3, b), useFlux_(useFlux)
    {
        check(N1 > 0, "");
        check(N2 > 0, "");
        check(N3 > 0, "");
        check(L1_ > 0, "");
        check(L2_ > 0, "");
        check(L3_ > 0, "");

        deltaK_.resize(N1 * N2 * (N3 / 2 + 1));
        pkDeriv_.resize(N1 * N2 * (N3 / 2 + 1));
        likeXDeriv_.resize(N1 * N2 * N3);
        buf_.resize(N1 * N2 * (N3 / 2 + 1));
        check(sigma_.size() == N1 * N2 * N3, "");
#ifdef CHECKS_ON
        for(int i = 0; i < N1 * N2 * N3; ++i)
        {
            check(sigma_[i] > 0, "");
        }
#endif
        check(pk_.size() == N1 * N2 * (N3 / 2 + 1), "");
    }

    void set(const DeltaVector3& x)
    {
        isComplex_ = x.isComplex();
        if(isComplex_)
        {
            deltaK_ = x.getComplex();
            deltaK2deltaX(N1_, N2_, N3_, deltaK_, &deltaX_, L1_, L2_, L3_);
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
            total += priorK(N1_, N2_, N3_, pk_, deltaK_);
        else
            total += priorX(N1_, N2_, N3_, pk_, la_.getDelta(), L1_, L2_, L3_, &deltaK_);

        return total;
    }

    void derivative(DeltaVector3 *res)
    {
        check(isComplex_ == res->isComplex(), "");
        if(isComplex_)
        {
            std::vector<std::complex<double> >& r = res->getComplex();
            priorKDeriv(N1_, N2_, N3_, pk_, deltaK_, &r);
            for(int i = 0; i < N1_; ++i)
            {
                for(int j = 0; j < N2_; ++j)
                {
                    for(int k = 0; k < N3_; ++k)
                    {
                        if(useFlux_)
                        {
                            likeXDeriv_[i] = 0;
                            for(int l = 0; l < N1_; ++l)
                            {
                                for(int m = 0; m < N2_; ++m)
                                {
                                    for(int n = 0; n < N3_; ++n)
                                    {
                                        const double s = sigma_[(l * N2_ + m) * N3_ + n];
                                        const double delta = la_.getFlux()[(l * N2_ + m) * N3_ + n] - data_[(l * N2_ + m) * N3_ + n];
                                        likeXDeriv_[(i * N2_ + j) * N3_ + k] += 2 * delta * la_.fluxDeriv(l, m, n, i, j, k) / (s * s);
                                    }
                                }
                            }
                        }
                        else
                        {
                            const int index = (i * N2_ + j) * N3_ + k;
                            const double s = sigma_[index];
                            const double delta = la_.getDeltaNonLin()[index] - data_[index];
                            likeXDeriv_[index] = 2 * delta * la_.deltaDeriv(index) / (s * s);
                        }
                    }
                }
            }

            deltaX2deltaK(N1_, N2_, N3_, likeXDeriv_, &buf_, L1_, L2_, L3_);

            for(int i = 0; i < N1_; ++i)
            {
                for(int j = 0; j < N2_; ++j)
                {
                    for(int k = 0; k < N3_ / 2 + 1; ++k)
                    {
                        double factor = 2; // for conjugates
                        if((k == 0 || k == N3_ / 2) && (j == 0 || j == N2_ / 2) && (i == 0 || i == N2_ / 2))
                            factor = 1;
                        r[(i * N2_ + j) * (N3_ / 2 + 1) + k] += buf_[(i * N2_ + j) * (N3_ / 2 + 1) + k] * double(N1_ * N2_ * N3_) / (L1_ * L1_ * L1_ * L2_ * L2_ * L2_) * factor;
                    }
                }
            }
        }
        else
        {
            std::vector<double>& r = res->get();
            priorXDeriv(N1_, N2_, N3_, pk_, la_.getDelta(), L1_, L2_, L3_, &r, &deltaK_, &pkDeriv_);
            check(r.size() == N1_ * N2_ * N3_, "");
            const int diffMax = std::min(N1_ / 2, 8);
            for(int i = 0; i < N1_; ++i)
            {
                for(int j = 0; j < N2_; ++j)
                {
                    for(int k = 0; k < N3_; ++k)
                    {
                        if(useFlux_)
                        {
                            for(int l1 = i - diffMax; l1 < i + diffMax; ++l1)
                            {
                                const int l = (l1 + N1_) % N1_;
                                for(int m1 = j - diffMax; m1 < j + diffMax; ++m1)
                                {
                                    const int m = (m1 + N2_) % N2_;
                                    for(int n = 0; n < N3_; ++n)
                                    {
                                        const double s = sigma_[(l * N2_ + m) * N3_ + n];
                                        const double delta = la_.getFlux()[(l * N2_ + m) * N3_ + n] - data_[(l * N2_ + m) * N3_ + n];
                                        r[(i * N2_ + j) * N3_ + k] += 2 * delta * la_.fluxDeriv(l, m, n, i, j, k) / (s * s);
                                    }
                                }
                            }

                            /*
                            const int index = (i * N2_ + j) * N3_ + k;
                            const double s = sigma_[index];
                            const double delta = la_.getFlux()[index] - data_[index];
                            const double d1 = 2 * delta * la_.deltaDeriv(index) * 0.02 * (1 + la_.getDeltaNonLin()[index]) / (s * s);
                            const double d2 = 2 * delta * la_.fluxDeriv(i, j, k, i, j, k) / (s * s);

                            //r[index] += d1;
                            r[index] += d2;
                            */
                        }
                        else
                        {
                            const int index = (i * N2_ + j) * N3_ + k;
                            const double s = sigma_[index];
                            const double delta = la_.getDeltaNonLin()[index] - data_[index];
                            r[index] += 2 * delta * la_.deltaDeriv(index) / (s * s);
                        }
                    }
                }
            }
        }
    }

private:
    const int N1_, N2_, N3_;
    const std::vector<double> data_;
    const std::vector<double> sigma_;
    const std::vector<double> pk_;
    const double L1_, L2_, L3_;
    const bool useFlux_;
    LymanAlpha3 la_;
    std::vector<std::complex<double> > deltaK_, pkDeriv_;
    std::vector<double> deltaX_;
    std::vector<double> likeXDeriv_;
    std::vector<std::complex<double> > buf_;
    bool isComplex_;
};

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

    void operator()(int iter, double f, double gradNorm, const DeltaVector3& x, const DeltaVector3& grad)
    {
        if(!out_)
            return;

        out_ << std::setprecision(10) << iter << "\t" << f << '\t' << gradNorm << std::endl;

        std::vector<double> deltaX = x.get();
        if(x.isComplex())
        {
            std::vector<std::complex<double> > deltaK = x.getComplex();
            deltaK2deltaX(N_, N_, N_, deltaK, &deltaX, L_, L_, L_, NULL, true);
        }
        std::stringstream fileNameStr;
        if(cg_)
            fileNameStr << "cg";
        else
            fileNameStr << "lbfgs";
        fileNameStr << "_iter_" << iter << ".dat";

        std::ofstream out;
        out.open(fileNameStr.str().c_str(), std::ios::out | std::ios::binary);
        StandardException exc;
        if(!out)
        {
            std::string exceptionStr = "Cannot write into file lbfgs_iter_ITERATION.dat";
            exc.set(exceptionStr);
            throw exc;
        }
        out.write(reinterpret_cast<char*>(&(deltaX[0])), N_ * N_ * N_ * sizeof(double));
        out.close();
    }

    void operator()(int iter, double f, double gradNorm, const DeltaVector3& x, const DeltaVector3& grad, const DeltaVector3& z)
    {
        (*this)(iter, f, gradNorm, x, grad);
    }

private:
    const int N_;
    const double L_;
    std::ofstream out_;
    const bool cg_;
};

} // namespace

int main(int argc, char *argv[])
{
    try {
        StandardException exc;

        if(argc < 2)
        {
            std::string exceptionStr = "Need to specify the parameters file";
            exc.set(exceptionStr);
            throw exc;
        }

        Parser parser(argv[1]);
        const int N = parser.getInt("N", 32);
        const int N1 = parser.getInt("N1", N);
        const int N2 = parser.getInt("N2", N);
        const int N3 = parser.getInt("N3", N);
        const double L = parser.getDouble("L", 32);
        const double L1 = parser.getDouble("L1", L);
        const double L2 = parser.getDouble("L2", L);
        const double L3 = parser.getDouble("L3", L);
        const double b = parser.getDouble("b", 0.1);
        const bool outIters = parser.getBool("out_iters", false);
        const bool hmc = parser.getBool("hmc", false);
        const bool flux = parser.getBool("flux", false);
        const bool isComplex = parser.getBool("fourier", false);
        const bool randomStart = parser.getBool("random_start", false);
        const bool many = parser.getBool("many", false);
        int seed = parser.getInt("seed", 100);
        const double noiseFrac = parser.getDouble("noise_frac", 0.1);
        const bool testDerivs = parser.getBool("test_derivs", false);
        const bool conjGrad = parser.getBool("conjugate_gradient", false);
        const bool mtLineSearch = parser.getBool("more_thuente", true);
        const double epsilon = parser.getDouble("epsilon", 1e-5);
        const int lbfgs_m = parser.getInt("lbfgs_m", 10);

        parser.dump();

        std::vector<int> bins;
        std::vector<double> kBinVals;
        const int nBins = powerSpectrumBins(N1, N2, N3, L1, L2, L3, &bins, &kBinVals);
        check(bins.size() == N1 * N2 * (N3 / 2 + 1), "");
        check(kBinVals.size() == nBins, "");
        vector2file("la3_k_bins.txt", kBinVals);
        std::unique_ptr<Math::TableFunction<double, double> > ps(psFromFile("p_2.txt"));

        std::vector<double> pBinVals(nBins);
        for(int i = 0; i < nBins; ++i)
            pBinVals[i] = ps->evaluate(kBinVals[i]);

        vector2file("la3_actual_ps.txt", pBinVals);

        std::vector<double> pk;
        discretePowerSpectrum(N1, N2, N3, L1, L2, L3, bins, pBinVals, &pk);
        check(pk.size() == N1 * N2 * (N3 / 2 + 1), "");

        // make sure there are no 0 elements in pk
        // TBD better
        for(int i = 0; i < pk.size(); ++i)
        {
            if(pk[i] == 0)
                pk[i] = 1e-5;
        }

        std::vector<std::complex<double> > deltaK;
        generateDeltaK(N1, N2, N3, pk, &deltaK, seed);
        std::vector<double> deltaX;
        deltaK2deltaX(N1, N2, N3, deltaK, &deltaX, L1, L2, L3);
        check(deltaX.size() == N1 * N2 * N3, "");

        LymanAlpha3 la(N1, N2, N3, deltaX, L1, L2, L3, b);

        std::vector<double> data = (flux ? la.getFlux() : la.getDeltaNonLin());
        std::vector<double> sigma(N1 * N2 * N3);
        for(int i = 0; i < N1 * N2 * N3; ++i)
        {
            sigma[i] = std::abs(data[i]) * noiseFrac;

            if(sigma[i] < 1e-4)
                sigma[i] = 1e-4;
        }

        vector2binFile("la_delta_x3.dat", la.getDelta());
        vector2binFile("la_delta_x3_nl.dat", la.getDeltaNonLin());
        vector2binFile("la_data3.dat", data);
        vector2binFile("la_v3.dat", la.getV());
        vector2binFile("la_tau3.dat", la.getTau());
        vector2binFile("la_flux3.dat", la.getFlux());

        seed = 200;
        Math::GaussianGenerator g1(seed, 0.0, 1.0);

        for(int i = 0; i < N1 * N2 * N3; ++i)
            data[i] += sigma[i] * g1.generate();

        // lbfgs stuff
        DeltaVector3Factory factory(N1, N2, N3, isComplex);
        LBFGSFunction3 f(N1, N2, N3, data, sigma, pk, L1, L2, L3, b, flux);

        if(testDerivs)
        {
            output_screen("Testing the function and derivatives..." << std::endl);
            seed = 300;
            std::vector<std::complex<double> > deltaKTest;
            generateDeltaK(N1, N2, N3, pk, &deltaKTest, seed);

            std::vector<double> deltaXTest;
            deltaK2deltaX(N1, N2, N3, deltaKTest, &deltaXTest, L1, L2, L3);
            check(deltaXTest.size() == N1 * N2 * N3, "");
            std::unique_ptr<DeltaVector3> testX(factory.giveMeOne());
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
            std::unique_ptr<DeltaVector3> testDerivs(factory.giveMeOne());
            Timer timer2("TEST FUNCTION DERIVATIVES");
            timer2.start();
            f.derivative(testDerivs.get());
            const unsigned long timer2Duration = timer2.end();
            output_screen("Function derivatives calculation took " << timer2Duration << " microseconds." << std::endl);
            double epsilon = 0.01;
            if(isComplex)
            {
                for(int i = 0; i < N1; i += 1)
                {
                    for(int j = 0; j < N2; j += 1)
                    {
                        for(int k = 0; k < N3 / 2 + 1; k += 1)
                        {
                            // real part
                            testX->getComplex() = deltaKTest;

                            epsilon = std::abs(std::real(deltaKTest[(i * N2 + j) * (N3 / 2 + 1) + k])) / 10000;
                            if(epsilon == 0)
                                epsilon = 1e-10;

                            testX->getComplex()[(i * N2 + j) * (N3 / 2 + 1) + k] += std::complex<double>(1.0, 0.0) * epsilon;
                            if((k == 0 || k == N3 / 2) && !((i == 0 || i == N1 / 2) && (j == 0 || j == N2 / 2)))
                            {
                                const int index1 = (((N1 - i) % N1) * N2 + (N2 - j) % N2) * (N3 / 2 + 1) + k;
                                testX->getComplex()[index1] += epsilon * std::complex<double>(1.0, 0.0);
                            }
                            f.set(*testX);
                            double pertVal = f.value();
                            double numDeriv = (pertVal - testVal) / epsilon;
                            if(!Math::areEqual(numDeriv, std::real(testDerivs->getComplex()[(i * N2 + j) * (N3 / 2 + 1) + k]), 1e-1))
                            {
                                output_screen("PROBLEM: index (" << i << ", " << j << ", " << k << ") real part. Numerical derivative = " << numDeriv << ", analytic = " << std::real(testDerivs->getComplex()[(i * N2 + j) * (N3 / 2 + 1) + k]) << std::endl);
                            }
                            
                            // imaginary part
                            if((k == 0 || k == N3 / 2) && (j == 0 || j == N2 / 2) && (i == 0 || i == N1 / 2))
                                continue;

                            testX->getComplex() = deltaKTest;

                            epsilon = std::abs(std::imag(deltaKTest[(i * N2 + j) * (N3 / 2 + 1) + k])) / 10000;
                            if(epsilon == 0)
                                epsilon = 1e-10;

                            testX->getComplex()[(i * N2 + j) * (N3 / 2 + 1) + k] += std::complex<double>(0.0, 1.0) * epsilon;
                            if((k == 0 || k == N3 / 2) && !((i == 0 || i == N1 / 2) && (j == 0 || j == N2 / 2)))
                            {
                                const int index1 = (((N1 - i) % N1) * N2 + (N2 - j) % N2) * (N3 / 2 + 1) + k;
                                testX->getComplex()[index1] -= epsilon * std::complex<double>(0.0, 1.0);
                            }
                            f.set(*testX);
                            pertVal = f.value();
                            numDeriv = (pertVal - testVal) / epsilon;
                            if(!Math::areEqual(numDeriv, std::imag(testDerivs->getComplex()[(i * N2 + j) * (N3 / 2 + 1) + k]), 1e-1))
                            {
                                output_screen("PROBLEM: index (" << i << ", " << j << ", " << k << ") imag part. Numerical derivative = " << numDeriv << ", analytic = " << std::imag(testDerivs->getComplex()[(i * N2 + j) * (N3 / 2 + 1) + k]) << std::endl);
                            }
                        }
                    }
                }
            }
            else
            {
                for(int i = 0; i < N1; i += 1)
                {
                    for(int j = 0; j < N2; j += 1)
                    {
                        for(int k = 0; k < N3; k += 1)
                        {
                            testX->get() = deltaXTest;
                            
                            epsilon = std::abs(deltaXTest[(i * N2 + j) * N3 + k]) / 10000;
                            if(epsilon == 0)
                                epsilon = 1e-10;

                            testX->get()[(i * N2 + j) * N3 + k] += epsilon;
                            f.set(*testX);
                            const double pertVal = f.value();
                            const double numDeriv = (pertVal - testVal) / epsilon;
                            if(!Math::areEqual(numDeriv, testDerivs->get()[(i * N2 + j) * N3 + k], 1e-1))
                            {
                                output_screen("PROBLEM: index (" << i << ", " << j << ", " << k << ") numerical derivative = " << numDeriv << ", analytic = " << testDerivs->get()[(i * N2 + j) * N3 + k] << std::endl);
                            }
                        }
                    }
                }
            }
            output_screen("OK" << std::endl);
        }

        std::unique_ptr<DeltaVector3> x(factory.giveMeOne());

        if(randomStart)
        {
            std::vector<std::complex<double> > deltaKStart;
            generateDeltaK(N1, N2, N3, pk, &deltaKStart);
            std::vector<double> deltaXStart;
            deltaK2deltaX(N1, N2, N3, deltaKStart, &deltaXStart, L1, L2, L3);
            check(deltaXStart.size() == N1 * N2 * N3, "");
            if(isComplex)
                x->getComplex() = deltaKStart;
            else
                x->get() = deltaXStart;
        }

        const double gradTol = 1e-5 * N1 * N2 * N3 * CosmoMPI::create().numProcesses();
        double minVal = 0;

        typedef Math::CG_General<DeltaVector3, DeltaVector3Factory, LBFGSFunction3> MyCG;
        MyCG cg(&factory, &f, *x);

        typedef Math::LBFGS_General<DeltaVector3, DeltaVector3Factory, LBFGSFunction3> MyLBFGS;
        LBFGSCallback cb("chi2.txt", N, L);
        MyLBFGS lbfgs(&factory, &f, *x, lbfgs_m, mtLineSearch);

        if(conjGrad)
            minVal = cg.minimize(x.get(), epsilon, gradTol, 10000000);
        else
            minVal = lbfgs.minimize(x.get(), epsilon, gradTol, 10000000, outIters ? &cb : NULL);

        std::vector<double> deltaXMin(N1 * N2 * N3);
        if(isComplex)
        {
            check(x->getComplex().size() == N1 * N2 * (N3 / 2 + 1), "");
            deltaK2deltaX(N1, N2, N3, x->getComplex(), &deltaXMin, L1, L2, L3);
        }
        else
        {
            check(x->get().size() == N1 * N2 * N3, "");
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
                generateDeltaK(N1, N2, N3, pk, &deltaKStart, 0);
                //generateDeltaK(pk, &deltaKStart, seed++);
                deltaK2deltaX(N1, N2, N3, deltaKStart, &deltaXStart, L1, L2, L3);
                check(deltaXStart.size() == N1 * N2 * N3, "");
                if(isComplex)
                    x->getComplex() = deltaKStart;
                else
                    x->get() = deltaXStart;
                double val = 0;
                if(conjGrad)
                {
                    cg.setStarting(*x);
                    val = cg.minimize(x.get(), epsilon, gradTol, 10000000);
                }
                else
                {
                    lbfgs.setStarting(*x);
                    val = lbfgs.minimize(x.get(), epsilon, gradTol, 10000000);
                }
                if(val < minVal)
                {
                    minVal = val;
                    if(isComplex)
                    {
                        check(x->getComplex().size() == N1 * N2 * (N3 / 2 + 1), "");
                        deltaK2deltaX(N1, N2, N3, x->getComplex(), &deltaXMin, L1, L2, L3);
                    }
                    else
                    {
                        check(x->get().size() == N1 * N2 * N3, "");
                        deltaXMin = x->get();
                    }
                }
            }
        }

        output_screen("MINIMUM VALUE FOUND = " << minVal << std::endl);

        vector2binFile("la_min3.dat", deltaXMin);

        la.reset(deltaXMin);

        vector2binFile("la_min_v3.dat", la.getV());
        vector2binFile("la_min_tau3.dat", la.getTau());
        vector2binFile("la_min_flux3.dat", la.getFlux());
    } catch (std::exception& e)
    {
        output_screen("EXCEPTION CAUGHT!!! " << std::endl << e.what() << std::endl);
        output_screen("Terminating!" << std::endl);
        return 1;
    }
    return 0;
}

