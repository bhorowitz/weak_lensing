#include <macros.hpp>
#include <matrix_impl.hpp>

#include "power_spectrum.hpp"
#include "utils.hpp"
#include "lin_alg.hpp"

#include <fftw3.h>

void ftMatrixFull(int N, double L, const Math::Matrix<double>& mat, Math::Matrix<double> *res, Math::Matrix<double> *resIm)
{
    check(N > 0, "");
    check(L > 0, "");
    check(mat.rows() == N * N, "");
    check(mat.cols() == N * N, "");

    std::vector<std::complex<double> > in(N * N), out(N * N);
    Math::Matrix<std::complex<double> > intermediate(N * N, N * N, 0);

    res->resize(N * N, N * N);
    if(resIm)
        resIm->resize(N * N, N * N);

    for(int i = 0; i < N * N; ++i)
    {
        for(int j = 0; j < N * N; ++j)
            in[j] = mat(i, j);

        fftw_plan fwdPlan = fftw_plan_dft_2d(N, N, reinterpret_cast<fftw_complex*>(&(in[0])), reinterpret_cast<fftw_complex*>(&(out[0])), -1, FFTW_ESTIMATE);
        check(fwdPlan, "");
        fftw_execute(fwdPlan);
        fftw_destroy_plan(fwdPlan);

        for(int j = 0; j < N * N; ++j)
            intermediate(i, j) = out[j];
    }

    const double L4 = L * L * L * L;

    for(int j = 0; j < N * N; ++j)
    {
        for(int i = 0; i < N * N; ++i)
            in[i] = intermediate(i, j);

        fftw_plan fwdPlan = fftw_plan_dft_2d(N, N, reinterpret_cast<fftw_complex*>(&(in[0])), reinterpret_cast<fftw_complex*>(&(out[0])), 1, FFTW_ESTIMATE);
        check(fwdPlan, "");
        fftw_execute(fwdPlan);
        fftw_destroy_plan(fwdPlan);

        for(int i = 0; i < N * N; ++i)
        {
            check(std::abs(std::imag(out[i])) < 1e-5, "");

            (*res)(i, j) = std::real(out[i]) / L4;
            if(resIm)
                (*resIm)(i, j) = std::imag(out[i]) / L4;
        }
    }
}

void ftMatrixFullWL(int N, double L, const Math::Matrix<double>& mat, Math::Matrix<double> *res, Math::Matrix<double> *resIm)
{
    check(N > 0, "");
    check(L > 0, "");
    check(mat.rows() == 2 * N * N, "");
    check(mat.cols() == 2 * N * N, "");

    std::vector<std::complex<double> > in(N * N), out1(N * N), out2(N * N);
    Math::Matrix<std::complex<double> > intermediate(2 * N * N, N * N, 0);

    res->resize(N * N, N * N);
    if(resIm)
        resIm->resize(N * N, N * N);

    for(int i = 0; i < 2 * N * N; ++i)
    {
        for(int j = 0; j < N * N; ++j)
            in[j] = mat(i, j);

        fftw_plan fwdPlan1 = fftw_plan_dft_2d(N, N, reinterpret_cast<fftw_complex*>(&(in[0])), reinterpret_cast<fftw_complex*>(&(out1[0])), -1, FFTW_ESTIMATE);
        check(fwdPlan1, "");
        fftw_execute(fwdPlan1);
        fftw_destroy_plan(fwdPlan1);

        for(int j = 0; j < N * N; ++j)
            in[j] = mat(i, N * N + j);

        fftw_plan fwdPlan2 = fftw_plan_dft_2d(N, N, reinterpret_cast<fftw_complex*>(&(in[0])), reinterpret_cast<fftw_complex*>(&(out2[0])), -1, FFTW_ESTIMATE);
        check(fwdPlan2, "");
        fftw_execute(fwdPlan2);
        fftw_destroy_plan(fwdPlan2);

        for(int j = 0; j < N * N; ++j)
        {
            int i1 = j / N;
            int j1 = j % N;
            if(j1 > N / 2)
            {
                j1 = N - j1;
                if(i1 != 0)
                    i1 = N - i1;
            }
            double c2, s2;
            getC2S2(N, N, L, L, i1, j1, &c2, &s2);
            intermediate(i, j) = c2 * out1[j] + s2 * out2[j];
        }
    }

    const double L4 = L * L * L * L;

    for(int j = 0; j < N * N; ++j)
    {
        for(int i = 0; i < N * N; ++i)
            in[i] = intermediate(i, j);

        fftw_plan fwdPlan1 = fftw_plan_dft_2d(N, N, reinterpret_cast<fftw_complex*>(&(in[0])), reinterpret_cast<fftw_complex*>(&(out1[0])), 1, FFTW_ESTIMATE);
        check(fwdPlan1, "");
        fftw_execute(fwdPlan1);
        fftw_destroy_plan(fwdPlan1);

        for(int i = 0; i < N * N; ++i)
            in[i] = intermediate(N * N + i, j);

        fftw_plan fwdPlan2 = fftw_plan_dft_2d(N, N, reinterpret_cast<fftw_complex*>(&(in[0])), reinterpret_cast<fftw_complex*>(&(out2[0])), 1, FFTW_ESTIMATE);
        check(fwdPlan2, "");
        fftw_execute(fwdPlan2);
        fftw_destroy_plan(fwdPlan2);

        for(int i = 0; i < N * N; ++i)
        {
            int i1 = i / N;
            int j1 = i % N;
            if(j1 > N / 2)
            {
                j1 = N - j1;
                if(i1 != 0)
                    i1 = N - i1;
            }
            double c2, s2;
            getC2S2(N, N, L, L, i1, j1, &c2, &s2);

            std::complex<double> myOut = c2 * out1[i] + s2 * out2[i];

            check(std::abs(std::imag(myOut)) < 1e-5, "");

            (*res)(i, j) = std::real(myOut) / L4;
            if(resIm)
                (*resIm)(i, j) = std::imag(myOut) / L4;
        }
    }
}

void linearAlgebraCalculation(int N, double L, const std::vector<double> &pk, bool weakLensing, const std::vector<double> &mask, const std::vector<double> &sigmaNoise, const std::vector<double> &dataX, const std::vector<double> &dataGamma1, const std::vector<double> &dataGamma2, std::vector<std::complex<double> > &wfk, std::vector<double> &wfx, std::vector<double> &b, Math::Matrix<double> &fisher, std::vector<double> &signal, std::vector<double> &theta, Math::Matrix<double> *invHessian)
{
    output_screen("Testing Wiener filter..." << std::endl);
    Math::SymmetricMatrix<double> C(N * N, N * N);

    std::vector<double> realBuffer(N * N);
    std::vector<std::complex<double> > complexBuffer(N * (N / 2 + 1));

    covarianceMatrix(N, N, pk, L, L, &C, &complexBuffer, &realBuffer);

    if(weakLensing)
    {
        std::vector<double> pkGamma1 = pk, pkGamma2 = pk, pkGamma12 = pk;
        Math::SymmetricMatrix<double> CGamma1(N * N, N * N), CGamma2(N * N, N * N), CGamma12(N * N, N * N);

        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N / 2 + 1; ++j)
            {
                const int index = i * (N / 2 + 1) + j;
                double c2 = 0, s2 = 0;
                getC2S2(N, N, L, L, i, j, &c2, &s2);
                pkGamma1[index] *= (c2 * c2);
                pkGamma2[index] *= (s2 * s2);
                pkGamma12[index] *= (c2 * s2);
            }
        }

        covarianceMatrix(N, N, pkGamma1, L, L, &CGamma1, &complexBuffer, &realBuffer);
        covarianceMatrix(N, N, pkGamma2, L, L, &CGamma2, &complexBuffer, &realBuffer);
        covarianceMatrix(N, N, pkGamma12, L, L, &CGamma12, &complexBuffer, &realBuffer);

        C.resize(2 * N * N, 2 * N * N);
        for(int i = 0; i < N * N; ++i)
        {
            for(int j = 0; j < N * N; ++j)
            {
                C(N * N + i, j) = CGamma12(i, j);
                if(j <= i)
                {
                    C(i, j) = CGamma1(i, j);
                    C(N * N + i, N * N + j) = CGamma2(i, j);
                }
            }
        }
    }

    std::vector<int> goodPixels;
    for(int i = 0; i < N * N; ++i)
        if(mask[i] != 0)
            goodPixels.push_back(i);

    if(weakLensing)
    {
        for(int i = 0; i < N * N; ++i)
            if(mask[i] != 0)
                goodPixels.push_back(N * N + i);
    }

    Math::SymmetricMatrix<double> CTrunc(goodPixels.size(), goodPixels.size(), 0.0);
    for(int i = 0; i < goodPixels.size(); ++i)
    {
        for(int j = 0; j <= i; ++j)
            CTrunc(i, j) = C(goodPixels[i], goodPixels[j]);
    }

    Math::SymmetricMatrix<double> NTrunc(goodPixels.size(), goodPixels.size(), 0.0);

    // add noise matrix
    for(int i = 0; i < goodPixels.size(); ++i)
    {
        NTrunc(i, i) = sigmaNoise[goodPixels[i] % (N * N)] * sigmaNoise[goodPixels[i] % (N * N)];
        CTrunc(i, i) += NTrunc(i, i);
    }

    Math::Matrix<double> CTruncMat = CTrunc;

    const int invRes = CTruncMat.invert();
    check(invRes == 0, "");

    std::vector<double> dataXTrunc(goodPixels.size());
    if(weakLensing)
    {
        for(int i = 0; i < goodPixels.size(); ++i)
        {
            if(i < goodPixels.size() / 2)
                dataXTrunc[i] = dataGamma1[goodPixels[i]];
            else
            {
                check(goodPixels[i] >= N * N, "");
                dataXTrunc[i] = dataGamma2[goodPixels[i] - N * N];
            }
        }
    }
    else
    {
        for(int i = 0; i < goodPixels.size(); ++i)
            dataXTrunc[i] = dataX[goodPixels[i]];
    }


    Math::Matrix<double> dataMat(dataXTrunc);
    Math::Matrix<double> CInvData = CTruncMat * dataMat;

    wfk.resize(N * (N / 2 + 1));

    if(weakLensing)
    {
        std::vector<double> CInvDataVec1(N * N, 0);
        std::vector<double> CInvDataVec2(N * N, 0);
        for(int i = 0; i < goodPixels.size() / 2; ++i)
        {
            CInvDataVec1[goodPixels[i]] = CInvData(i, 0);
            CInvDataVec2[goodPixels[i]] = CInvData(goodPixels.size() / 2 + i, 0);
        }

        std::vector<std::complex<double> > wfk1(N * (N / 2 + 1));
        fftw_plan fwdPlan1 = fftw_plan_dft_r2c_2d(N, N, &(CInvDataVec1[0]), reinterpret_cast<fftw_complex*>(&(wfk1[0])), FFTW_ESTIMATE);
        check(fwdPlan1, "");
        fftw_execute(fwdPlan1);
        fftw_destroy_plan(fwdPlan1);

        std::vector<std::complex<double> > wfk2(N * (N / 2 + 1));
        fftw_plan fwdPlan2 = fftw_plan_dft_r2c_2d(N, N, &(CInvDataVec2[0]), reinterpret_cast<fftw_complex*>(&(wfk2[0])), FFTW_ESTIMATE);
        check(fwdPlan2, "");
        fftw_execute(fwdPlan2);
        fftw_destroy_plan(fwdPlan2);

        for(int i = 0; i < N; ++i)
        {
            for(int j = 0; j < N / 2 + 1; ++j)
            {
                const int index = i * (N / 2 + 1) + j;
                double c2 = 0, s2 = 0;
                getC2S2(N, N, L, L, i, j, &c2, &s2);
                wfk[index] = wfk1[index] * c2 + wfk2[index] * s2;
            }
        }
    }
    else
    {
        std::vector<double> CInvDataVec(N * N, 0);
        for(int i = 0; i < goodPixels.size(); ++i)
            CInvDataVec[goodPixels[i]] = CInvData(i, 0);


        fftw_plan fwdPlan = fftw_plan_dft_r2c_2d(N, N, &(CInvDataVec[0]), reinterpret_cast<fftw_complex*>(&(wfk[0])), FFTW_ESTIMATE);
        check(fwdPlan, "");
        fftw_execute(fwdPlan);
        fftw_destroy_plan(fwdPlan);
    }

    for(int i = 0; i < N * (N / 2 + 1); ++i)
        wfk[i] *= (pk[i] / (L * L));

    wfx.resize(N * N);
    deltaK2deltaX(N, N, wfk, &wfx, L, L, &complexBuffer, true);

    // now calculate the noise bias
    Math::Matrix<double> CInvNCInvTrunc(goodPixels.size(), goodPixels.size(), 0.0);
    Math::Matrix<double>::multiplyMatrices(CTruncMat, NTrunc, &CInvNCInvTrunc);
    CInvNCInvTrunc *= CTruncMat;

    Math::SymmetricMatrix<double> CInvNCInv(N * N, N * N, 0);
    if(weakLensing)
        CInvNCInv.resize(2 * N * N, 2 * N * N);

    for(int i = 0; i < goodPixels.size(); ++i)
    {
        for(int j = 0; j <= i; ++j)
            CInvNCInv(goodPixels[i], goodPixels[j]) = CInvNCInvTrunc(i, j);
    }

    if(invHessian)
    {
        Math::Matrix<double> NInv(N * N, N * N, 0);
        if(weakLensing)
            NInv.resize(2 * N * N, 2 * N * N, 0);

        for(int i = 0; i < goodPixels.size(); ++i)
            NInv(goodPixels[i], goodPixels[i]) = 1.0 / (sigmaNoise[goodPixels[i] % (N * N)] * sigmaNoise[goodPixels[i] % (N * N)]);

        Math::Matrix<double> NInvFT;
        if(weakLensing)
            ftMatrixFullWL(N, L, NInv, &NInvFT);
        else
            ftMatrixFull(N, L, NInv, &NInvFT);

        *invHessian = NInvFT;
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
            check(pk[indexNew] != 0, "");
            (*invHessian)(i, i) += (L * L) / pk[indexNew];
        }
    }

    Math::Matrix<double> CInvNCInvFT;
    if(weakLensing)
        ftMatrixFullWL(N, L, CInvNCInv, &CInvNCInvFT);
    else
        ftMatrixFull(N, L, CInvNCInv, &CInvNCInvFT);
    matrix2file("CInvNCInvFT.txt", CInvNCInvFT);

    std::vector<int> bins;
    std::vector<double> kBinVals;
    const int nBins = powerSpectrumBins(N, N, L, L, &bins, &kBinVals);
    check(bins.size() == N * (N / 2 + 1), "");
    check(kBinVals.size() == nBins, "");

    std::vector<int> binsFull;
    const int nBinsFull = powerSpectrumBinsFull(N, N, L, L, &binsFull);
    check(nBinsFull == nBins, "");

    b.clear();
    b.resize(nBins, 0);
    for(int i = 0; i < N * N; ++i)
    {
        const int l = binsFull[i];
        check(l >= 0 && l < nBins, "");
        b[l] += CInvNCInvFT(i, i);
    }

    // calculate fisher matrix!
    Math::SymmetricMatrix<double> CInv(N * N, N * N, 0);
    if(weakLensing)
        CInv.resize(2 * N * N, 2 * N * N);
    for(int i = 0; i < goodPixels.size(); ++i)
    {
        for(int j = 0; j <= i; ++j)
            CInv(goodPixels[i], goodPixels[j]) = CTruncMat(i, j);
    }

    Math::Matrix<double> CInvFT;
    if(weakLensing)
        ftMatrixFullWL(N, L, CInv, &CInvFT);
    else
        ftMatrixFull(N, L, CInv, &CInvFT);

    fisher.resize(nBins, nBins, 0);
    for(int i = 0; i < N * N; ++i)
    {
        const int l1 = binsFull[i];
        check(l1 >= 0 && l1 < nBins, "");
        for(int j = 0; j < N * N; ++j)
        {
            const int l2 = binsFull[j];
            check(l2 >= 0 && l2 < nBins, "");
            fisher(l1, l2) += CInvFT(i, j) * CInvFT(j, i) / 2;
        }
    }

    signal.clear();
    signal.resize(nBins, 0);

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
        if(pk[indexNew] != 0)
            s = wfk[indexNew] / pk[indexNew];
        signal[l] += std::real(s) * std::real(s) + std::imag(s) * std::imag(s);
    }

    std::vector<double> FTheta(nBins);
    for(int i = 0; i < nBins; ++i)
    {
        FTheta[i] = (signal[i] - b[i]) / 2;
        if(FTheta[i] < 0)
            FTheta[i] = 0;
    }

    Math::Matrix<double> FThetaMat(FTheta, true);

    /*
    const int invFisherRes = fisher.invert();
    check(invFisherRes == 0, "");
    Math::Matrix<double> theta = fisher * FThetaMat;
    */

    theta.resize(nBins);
    for(int i = 0; i < nBins; ++i)
    {
        double sumF = 0;
        for(int k = 0; k < nBins; ++k)
            sumF += fisher(i, k);
        theta[i] = FThetaMat(i, 0) / sumF;
        theta[i] /= (L * L);
    }
}

