#include <fstream>

#include <macros.hpp>
#include <exception_handler.hpp>
#include <random.hpp>
#include <numerics.hpp>
#include <lyman_alpha.hpp>
#include <cosmo_mpi.hpp>

int main()
{
    const int N = 32;
    std::vector<double> delta(N * N);

    const int seed = 300;
    Math::GaussianGenerator gx(seed, 0, 1);
    std::vector<double> px(N * N, 1.0);
    for(int i = 0; i < N * N; ++i)
        delta[i] = std::sqrt(px[i]) * gx.generate();


    const double L = 100.0;
    const double b = 0.1;
    LymanAlpha2 la(N, N, delta, L, L, b);

    const std::vector<double> v = la.getV();
    const std::vector<double> tau = la.getTau();
    const std::vector<double> flux = la.getFlux();

    int testRes = 0;

    output_screen("TESTING velocity, tau, and flux derivatives..." << std::endl);
    bool derivTest = true;
    std::vector<std::vector<double> > vDerivs(N * N);
    std::vector<std::vector<double> > tauDerivs(N * N);
    std::vector<std::vector<double> > fluxDerivs(N * N);
    for(int i = 0; i < N * N; ++i)
    {
        vDerivs[i].resize(N * N);
        tauDerivs[i].resize(N * N);
        fluxDerivs[i].resize(N * N);
        for(int j = 0; j < N * N; ++j)
        {
            vDerivs[i][j] = la.vDeriv(i / N, i % N, j / N, j % N);
            tauDerivs[i][j] = la.tauDeriv(i / N, i % N, j / N, j % N);
            fluxDerivs[i][j] = la.fluxDeriv(i / N, i % N, j / N, j % N);
        }
    }

    const double d = 0.0001;
    for(int i = 0; i < N * N; ++i)
    {
        delta[i] += d;
        la.reset(delta);
        const std::vector<double>& vPert = la.getV();
        const std::vector<double>& tauPert = la.getTau();
        const std::vector<double>& fluxPert = la.getFlux();

        for(int j = 0; j < N * N; ++j)
        {
            double numDeriv = (vPert[j] - v[j]) / d;
            double actualDeriv = vDerivs[j][i];

            if(!Math::areEqual(actualDeriv, numDeriv, 1e-2))
            {
                output_screen("FAIL: for partial v_(" << j / N << ", " << j % N << ") / partial delta_(" << i / N << ", " << i % N << ")! Expected " << numDeriv << ", actual " << actualDeriv << std::endl);
                testRes = 1;
                derivTest = false;
            }

            numDeriv = (tauPert[j] - tau[j]) / d;
            actualDeriv = tauDerivs[j][i];

            if(!(Math::areEqual(actualDeriv, numDeriv, 1e-1) || std::abs(actualDeriv) < 1e-4))
            {
                output_screen("FAIL: for partial tau_(" << j / N << ", " << j % N << ") / partial delta_(" << i / N << ", " << i % N << ")! Expected " << numDeriv << ", actual " << actualDeriv << std::endl);
                testRes = 1;
                derivTest = false;
            }

            numDeriv = (fluxPert[j] - flux[j]) / d;
            actualDeriv = fluxDerivs[j][i];

            if(!(Math::areEqual(actualDeriv, numDeriv, 1e-1) || std::abs(actualDeriv) < 1e-4))
            {
                output_screen("FAIL: for partial flux_(" << j / N << ", " << j % N << " / partial delta_(" << i / N << ", " << i % N << ")! Expected " << numDeriv << ", actual " << actualDeriv << std::endl);
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

    for(int i = 0; i < N * N; ++i)
        out << i / N << '\t' << i % N << '\t' << delta[i] << '\t' << v[i] << '\t' << tau[i] << '\t' << flux[i] << std::endl;
    out.close();

    return testRes;
}
