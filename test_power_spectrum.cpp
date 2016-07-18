#include <fstream>
#include <cmath>

#include <macros.hpp>
#include <exception_handler.hpp>
#include <random.hpp>
#include <numerics.hpp>
#include <histogram.hpp>
#include <matrix_impl.hpp>
#include <cosmo_mpi.hpp>

#include <fftw3.h>

#include "power_spectrum.hpp"

int main()
{
    int testRes = 0;

    SimplePowerSpectrum ps(0.015);
    ps.normalize(4.0, 1.0);

    const double L = 100;
    int N = 128;

    std::vector<double> pk;
    discretePowerSpectrum(ps, L, N, &pk);

    int seed = 1000;
    output_screen("Testing simulations of deltaK..." << std::endl);
    const int NSim = 10000;
    double mean = 0, mean2 = 0;
    std::vector<std::complex<double> > deltaK(N / 2 + 1);
    std::vector<double> deltaX(N);
    std::vector<double> realBuf(N);
    for(int i = 0; i < NSim; ++i)
    {
        generateDeltaK(N, pk, &deltaK, seed++, &realBuf);
        const double p = priorK(N, pk, deltaK);
        deltaK2deltaX(N, deltaK, &deltaX, L);
        const double p1 = priorX(N, pk, deltaX, L, &deltaK);
        if(!Math::areEqual(p1, p, 1e-5))
        {
            output_screen("FAIL: the prior in K space is " << p << " but in X space is " << p1 << ". They must be equal!" << std::endl);
            testRes = 1;
        }
        mean += p;
        mean2 += p * p;
    }

    mean /= NSim;
    mean2 /= NSim;

    double sigma = std::sqrt(mean2 - mean * mean);
    output_screen("simulations prior = " << mean << " +/- " << sigma << std::endl);
    output_screen("expected: " << N << " +/- " << std::sqrt(double(2 * N)) << std::endl);

    if(!Math::areEqual(sigma, std::sqrt(double(2 * N)), 0.1))
    {
        output_screen("FAIL: there is not good agreement between simulated and expected variances!" << std::endl);
        testRes = 1;
    }

    if(std::abs(mean - N) > 0.2 * sigma)
    {
        output_screen("FAIL: there is not good agreement between simulated and expected means!" << std::endl);
        testRes = 1;
    }

    output_screen("Testing prior derivatives in K space..." << std::endl);
    seed = 100;
    generateDeltaK(N, pk, &deltaK, seed, &realBuf);
    double pr = priorK(N, pk, deltaK);
    std::vector<std::complex<double> > deltaKDerivs;
    priorKDeriv(N, pk, deltaK, &deltaKDerivs);

    check(deltaKDerivs.size() == N / 2 + 1, "");
    const double epsilon = 0.0000001;
    for(int i = 0; i < N / 2 + 1; ++i)
    {
        if(pk[i] == 0)
        {
            if(std::abs(deltaK[i]) > 1e-10)
            {
                output_screen("PROBLEM: index " << i << " pk is 0 but deltaK is not!" << std::endl);
                testRes = 1;
            }

            if(std::abs(deltaKDerivs[i]) != 0)
            {
                output_screen("PROBLEM: index " << i << " pk is 0 but the prior derivative is not!" << std::endl);
                testRes = 1;
            }

            continue;
        }
        std::vector<std::complex<double> > deltaKPert = deltaK;
        deltaKPert[i] += epsilon * std::complex<double>(1.0, 0.0);
        double prPert = priorK(N, pk, deltaKPert);
        double numDeriv = (prPert - pr) / epsilon;
        if(!Math::areEqual(numDeriv, std::real(deltaKDerivs[i]), 1e-3))
        {
            output_screen("FAIL: index " << i << " real part. Numerical derivative = " << numDeriv << ", analytic = " << std::real(deltaKDerivs[i]) << "." << std::endl);
            testRes = 1;
        }

        if(i > 0 && i < N / 2)
        {
            // test the imaginary part as well
            deltaKPert = deltaK;
            deltaKPert[i] += epsilon * std::complex<double>(0.0, 1.0);
            prPert = priorK(N, pk, deltaKPert);
            numDeriv = (prPert - pr) / epsilon;
            if(!Math::areEqual(numDeriv, std::imag(deltaKDerivs[i]), 1e-3))
            {
                output_screen("FAIL: index " << i << " imaginary part. Numerical derivative = " << numDeriv << ", analytic = " << std::imag(deltaKDerivs[i]) << "." << std::endl);
                testRes = 1;
            }
        }
    }

    output_screen("Testing prior derivatives in X space..." << std::endl);
    seed = 200;
    generateDeltaK(N, pk, &deltaK, seed, &realBuf);
    deltaK2deltaX(N, deltaK, &deltaX, L);
    pr = priorX(N, pk, deltaX, L, &deltaK);
    std::vector<double> deltaXDerivs;
    priorXDeriv(N, pk, deltaX, L, &deltaXDerivs, &deltaK, &deltaKDerivs);
    for(int i = 0; i < N; ++i)
    {
        std::vector<double> deltaXPert = deltaX;
        deltaXPert[i] += epsilon;
        const double prPert = priorX(N, pk, deltaXPert, L, &deltaK);
        const double numDeriv = (prPert - pr) / epsilon;
        if(!Math::areEqual(numDeriv, deltaXDerivs[i], 1e-2))
        {
            output_screen("FAIL: index " << i << ". Numerical derivative = " << numDeriv << ", analytic = " << deltaXDerivs[i] << "." << std::endl);
            testRes = 1;
        }
    }

    output_screen("NOW 2D!!!" << std::endl);
    N = 16;

    std::vector<double> pk2;
    ps.normalize2(4.0, 1.0);
    discretePowerSpectrum(ps, L, L, N, N, &pk2);
    seed = 1000;
    output_screen("Testing simulations of deltaK..." << std::endl);
    mean = 0;
    mean2 = 0;
    std::vector<std::complex<double> > deltaK2(N * (N / 2 + 1));
    std::vector<double> deltaX2(N * N);
    std::vector<double> realBuf2(N * N);

    Math::SymmetricMatrix<double> covMatSym(N * N, N * N, 0.0);
    for(int i = 0; i < NSim; ++i)
    {
        generateDeltaK(N, N, pk2, &deltaK2, seed++, &realBuf2);
        const double p = priorK(N, N, pk2, deltaK2);
        deltaK2deltaX(N, N, deltaK2, &deltaX2, L, L, NULL, false);
        const double p1 = priorX(N, N, pk2, deltaX2, L, L, &deltaK2);
        if(!Math::areEqual(p1, p, 1e-3))
        {
            output_screen("FAIL: the prior in K space is " << p << " but in X space is " << p1 << ". They must be equal!" << std::endl);
            testRes = 1;
        }
        mean += p;
        mean2 += p * p;

        for(int j = 0; j < N * N; ++j)
        {
            for(int k = 0; k <= j; ++k)
                covMatSym(j, k) += deltaX2[j] * deltaX2[k];
        }
    }

    mean /= NSim;
    mean2 /= NSim;

    sigma = std::sqrt(mean2 - mean * mean);
    output_screen("simulations prior = " << mean << " +/- " << sigma << std::endl);
    output_screen("expected: " << N * N << " +/- " << std::sqrt(double(2 * N * N)) << std::endl);

    if(!Math::areEqual(sigma, std::sqrt(double(2 * N * N)), 0.1))
    {
        output_screen("FAIL: there is not good agreement between simulated and expected variances!" << std::endl);
        testRes = 1;
    }

    if(std::abs(mean - N * N) > 0.2 * sigma)
    {
        output_screen("FAIL: there is not good agreement between simulated and expected means!" << std::endl);
        testRes = 1;
    }

    Math::SymmetricMatrix<double> covMat(N * N, N * N);
    covarianceMatrix(N, N, pk2, L, L, &covMat, NULL, &realBuf2);

    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            const int index1 = i * N + j;
            for(int k = std::max(0, i - 2); k < std::min(N, i + 2); ++k)
            {
                for(int l = std::max(0, j - 2); l < std::min(N, j + 2); ++l)
                {
                    const int index2 = k * N + l;
                    const double sym = covMatSym(index1, index2) / NSim;
                    if(!Math::areEqual(sym, covMat(index1, index2), 0.1))
                    {
                        output_screen("FAIL: there is not good agreement between simulated and expected covariance matrices!" << std::endl << "Index: (" << i << ", " << j << "), (" << k << ", " << l << ") simulated = " << sym << " expected = " << covMat(index1, index2) << std::endl);
                        testRes = 1;
                    }
                }
            }
        }
        for(int j = 0; j <= i; ++j)
        {
        }
    }

    output_screen("Testing prior derivatives in K space..." << std::endl);
    seed = 100;
    generateDeltaK(N, N, pk2, &deltaK2, seed, &realBuf2);
    double pr2 = priorK(N, N, pk2, deltaK2);
    std::vector<std::complex<double> > deltaKDerivs2;
    priorKDeriv(N, N, pk2, deltaK2, &deltaKDerivs2);

    check(deltaKDerivs2.size() == N * (N / 2 + 1), "");
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N / 2 + 1; ++j)
        {
            const int index = i * (N / 2 + 1) + j;
            if(pk2[index] == 0)
            {
                if(std::abs(deltaK2[index]) > 1e-10)
                {
                    output_screen("PROBLEM: index (" << i << ", " << j << ") pk is 0 but deltaK is not!" << std::endl);
                    testRes = 1;
                }

                if(std::abs(deltaKDerivs2[index]) != 0)
                {
                    output_screen("PROBLEM: index (" << i << ", " << j << ") pk is 0 but the prior derivative is not!" << std::endl);
                    testRes = 1;
                }

                continue;
            }
            std::vector<std::complex<double> > deltaKPert2 = deltaK2;
            deltaKPert2[index] += epsilon * std::complex<double>(1.0, 0.0);
            // certain elements have redundancy, i.e. their conjugates are in the array
            if((j == 0 || j == N / 2) && (i != 0 && i != N / 2))
            {
                const int index1 = (N - i) * (N / 2 + 1) + j;
                deltaKPert2[index1] += epsilon * std::complex<double>(1.0, 0.0);
            }
            double prPert2 = priorK(N, N, pk2, deltaKPert2);
            double numDeriv = (prPert2 - pr2) / epsilon;
            if(!Math::areEqual(numDeriv, std::real(deltaKDerivs2[index]), 2e-2))
            {
                output_screen("FAIL: index (" << i << ", " << j << ") real part. Numerical derivative = " << numDeriv << ", analytic = " << std::real(deltaKDerivs2[index]) << "." << std::endl);
                testRes = 1;
            }

            // if it's real
            if((i == 0 && (j == 0 || j == N / 2)) || (i == N / 2 && (j == 0 || j == N / 2)))
                continue;

            deltaKPert2 = deltaK2;
            deltaKPert2[index] += epsilon * std::complex<double>(0.0, 1.0);
            if(j == 0 || j == N / 2)
            {
                const int index1 = (N - i) * (N / 2 + 1) + j;
                deltaKPert2[index1] -= epsilon * std::complex<double>(0.0, 1.0);
            }
            prPert2 = priorK(N, N, pk2, deltaKPert2);
            numDeriv = (prPert2 - pr2) / epsilon;
            if(!Math::areEqual(numDeriv, std::imag(deltaKDerivs2[index]), 2e-2))
            {
                output_screen("FAIL: index (" << i << ", " << j << ") imaginary part. Numerical derivative = " << numDeriv << ", analytic = " << std::imag(deltaKDerivs2[index]) << "." << std::endl);
                testRes = 1;
            }
        }
    }
    
    return testRes;
}
