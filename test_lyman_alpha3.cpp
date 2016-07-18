#include <fstream>
#include <memory>

#include <macros.hpp>
#include <exception_handler.hpp>
#include <random.hpp>
#include <numerics.hpp>
#include <lyman_alpha.hpp>
#include <cosmo_mpi.hpp>

#include "utils.hpp"
#include "power_spectrum.hpp"

int main()
{
    const int N = 8;
    const double L = 32.0;
    std::vector<double> delta;

    std::vector<int> bins;
    std::vector<double> kBinVals;
    const int nBins = powerSpectrumBins(N, N, N, L, L, L, &bins, &kBinVals);
    check(bins.size() == N * N * (N / 2 + 1), "");
    check(kBinVals.size() == nBins, "");

    std::unique_ptr<Math::TableFunction<double, double> > ps(psFromFile("p_2.txt"));

    std::vector<double> pBinVals(nBins);
    for(int i = 0; i < nBins; ++i)
        pBinVals[i] = ps->evaluate(kBinVals[i]);

    std::vector<double> pk;
    discretePowerSpectrum(N, N, N, L, L, L, bins, pBinVals, &pk);
    check(pk.size() == N * N * (N / 2 + 1), "");

    // make sure there are no 0 elements in pk
    // TBD better
    for(int i = 0; i < pk.size(); ++i)
    {
        if(pk[i] == 0)
            pk[i] = 1e-5;
    }

    int seed = 100;
    std::vector<std::complex<double> > deltaK;
    generateDeltaK(N, N, N, pk, &deltaK, seed);
    deltaK2deltaX(N, N, N, deltaK, &delta, L, L, L);
    check(delta.size() == N * N * N, "");

    const double b = 0.1;
    LymanAlpha3 la(N, N, N, delta, L, L, L, b);

    const std::vector<double> v = la.getV();
    const std::vector<double> tau = la.getTau();
    const std::vector<double> flux = la.getFlux();

    int testRes = 0;

    output_screen("TESTING velocity, tau, and flux derivatives..." << std::endl);
    bool derivTest = true;
    std::vector<std::vector<double> > vDerivs(N * N * N);
    std::vector<std::vector<double> > tauDerivs(N * N * N);
    std::vector<std::vector<double> > fluxDerivs(N * N * N);

    std::ofstream outDerivsV("la3_test_derivs_v.txt", std::ios::binary);
    for(int i = 0; i < N * N * N; ++i)
    {
        vDerivs[i].resize(N * N * N);
        tauDerivs[i].resize(N * N * N);
        fluxDerivs[i].resize(N * N * N);
        for(int j = 0; j < N * N * N; ++j)
        {
            vDerivs[i][j] = la.vDeriv(i / N / N, i / N % N, i % N, j / N / N, j / N % N, j % N);
            tauDerivs[i][j] = la.tauDeriv(i / N / N, i / N % N, i % N, j / N / N, j / N % N, j % N);
            fluxDerivs[i][j] = la.fluxDeriv(i / N / N, i / N % N, i % N, j / N / N, j / N % N, j % N);

            //outDerivs << i / N / N << ' ' << i / N % N << ' ' << i % N << ' ' << j / N / N << ' ' << j / N % N << ' ' << j % N << ' ' << vDerivs[i][j] << ' ' << tauDerivs[i][j] << ' ' << fluxDerivs[i][j] << std::endl;
        }
        outDerivsV.write((char*)(&(vDerivs[i][0])), N * N * N * sizeof(double));
    }
    outDerivsV.close();

    //const double d = 0.0001;
    for(int i = 0; i < N * N * N; ++i)
    {
        double d = std::abs(delta[i] / 10000);
        if(d < 1e-8)
            d = 1e-8;

        delta[i] += d;
        la.reset(delta);
        const std::vector<double>& vPert = la.getV();
        const std::vector<double>& tauPert = la.getTau();
        const std::vector<double>& fluxPert = la.getFlux();

        const int i1 = i / N / N;
        const int i2 = i / N % N;
        const int i3 = i % N;

        for(int j = 0; j < N * N * N; ++j)
        {
            double numDeriv = (vPert[j] - v[j]) / d;
            double actualDeriv = vDerivs[j][i];

            const int j1 = j / N / N;
            const int j2 = j / N % N;
            const int j3 = j % N;

            int d1 = std::abs(i1 - j1);
            d1 = std::min(d1, N - d1);
            int d2 = std::abs(i2 - j2);
            d2 = std::min(d2, N - d2);
            int d3 = std::abs(i3 - j3);
            d3 = std::min(d3, N - d3);

            /*
            if(d1 > 2 && d2 > 2)
            {
                if(std::abs(actualDeriv) > 0.2)
                {
                    output_screen("FAIL: for partial v_(" << j1 << ", " << j2 << ", " << j3 << ") / partial delta_(" << i1 << ", " << i2 << ", " << i3 << ")! Expected to be small, but it is " << actualDeriv << std::endl);
                    testRes = 1;
                    derivTest = false;
                }
                continue;
            }
            */

            if(!Math::areEqual(actualDeriv, numDeriv, 1e-2))
            {
                output_screen("FAIL: for partial v_(" << j1 << ", " << j2 << ", " << j3 << ") / partial delta_(" << i1 << ", " << i2 << ", " << i3 << ")! Expected " << numDeriv << ", actual " << actualDeriv << std::endl);
                testRes = 1;
                derivTest = false;
            }

            numDeriv = (tauPert[j] - tau[j]) / d;
            actualDeriv = tauDerivs[j][i];

            if(!(Math::areEqual(actualDeriv, numDeriv, 5e-2) || (std::abs(actualDeriv) < 1e-20) && std::abs(numDeriv) < 1e-6))
            {
                output_screen("FAIL: for partial tau_(" << j / N / N << ", " << j / N % N << ", " << j % N << ") / partial delta_(" << j / N / N << ", " << j / N % N << ", " << j % N << ")! Expected " << numDeriv << ", actual " << actualDeriv << std::endl);
                testRes = 1;
                derivTest = false;
            }

            numDeriv = (fluxPert[j] - flux[j]) / d;
            actualDeriv = fluxDerivs[j][i];

            if(!(Math::areEqual(actualDeriv, numDeriv, 1e-1) || (std::abs(actualDeriv) < 1e-8) && std::abs(numDeriv) < 1e-5))
            {
                output_screen("FAIL: for partial flux_(" << j / N / N << ", " << j / N % N << ", " << j % N << ") / partial delta_(" << j / N / N << ", " << j / N % N << ", " << j % N << ")! Expected " << numDeriv << ", actual " << actualDeriv << std::endl);
                testRes = 1;
                derivTest = false;
            }
        }

        delta[i] -= d;
    }
    if(derivTest)
    {
        output_screen("OK" << std::endl);
    }
    else
    {
        output_screen("Derivatives test FAILED!" << std::endl);
    }

    std::ofstream out("test_results.txt");

    for(int i = 0; i < N * N * N; ++i)
        out << i / N / N << '\t' << i / N % N << '\t' << i % N << '\t' << delta[i] << '\t' << v[i] << '\t' << tau[i] << '\t' << flux[i] << std::endl;
    out.close();

    return testRes;
}
