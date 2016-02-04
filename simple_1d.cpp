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
#include <markov_chain.hpp>
#include <random.hpp>
#include <lbfgs.hpp>
#include <timer.hpp>
#include <math_constants.hpp>
#include <numerics.hpp>
#include <hmc_general.hpp>
#include <cosmo_mpi.hpp>

/*
#ifdef COSMO_OMP
#include <omp.h>
#endif
*/

#include <fftw3.h>

namespace
{

void generateModes(const std::vector<double>& pk, std::vector<std::complex<double> > *deltaK, double L = 1, int seed = 0)
{
    check(!pk.empty(), "");
    check(L > 0, "");

    if(seed == 0)
        seed = std::time(0);

    const size_t n = pk.size();
    deltaK->clear();

    Math::GaussianGenerator g(seed, 0, 1);
    for(int i = 0; i < n / 2 + 1; ++i)
    {
        check(pk[i] >= 0, "");
        const double re = g.generate() * std::sqrt(L * pk[i] / 2);
        const double im = g.generate() * std::sqrt(L * pk[i] / 2);
        deltaK->push_back(std::complex<double>(re, im));
    }
}

// -2ln(like)
class NLLike : public Math::RealFunctionMultiDim
{
public:
    NLLike(const std::vector<double>& delta, const std::vector<double>& sigma, const std::vector<double>& ps) : delta_(delta), sigma_(sigma), ps_(ps)
    {
        check(!delta_.empty(), "");
        check(delta_.size() == sigma_.size(), "");
        check(ps_.size() == delta_.size(), "");
    }

    virtual double evaluate(const std::vector<double>& x) const
    {
        check(x.size() == delta_.size(), "");
        double res = 0;
        for(int i = 0; i < x.size(); ++i)
        {
            const double xLN = std::exp(x[i]) - 1;
            const double d = xLN - delta_[i];
            const double s = sigma_[i];
            check(s > 0, "");
            res += d * d / (s * s);
            
            // add prior
            check(ps_[i] > 0, "");
            res += x[i] * x[i] / ps_[i];
        }

        double totalRes = 0;
#ifdef COSMO_MPI
        CosmoMPI::create().reduce(&res, &totalRes, 1, CosmoMPI::DOUBLE, CosmoMPI::SUM);
#else
        totalRes = res;
#endif

        return totalRes;
    }

private:
    const std::vector<double> delta_, sigma_, ps_;
};

class NLLikeGrad : public Math::RealFunctionMultiToMulti
{
public:
    NLLikeGrad(const std::vector<double>& delta, const std::vector<double>& sigma, const std::vector<double>& ps) : delta_(delta), sigma_(sigma), ps_(ps)
    {
        check(!delta_.empty(), "");
        check(delta_.size() == sigma_.size(), "");
        check(ps_.size() == delta_.size(), "");
    }

    virtual void evaluate(const std::vector<double>& x, std::vector<double> *res) const
    {
        check(x.size() == delta_.size(), "");
        res->resize(x.size());

        for(int i = 0; i < x.size(); ++i)
        {
            const double xLN = std::exp(x[i]) - 1;
            const double d = xLN - delta_[i];
            const double s = sigma_[i];
            check(s > 0, "");

            res->at(i) = 2 * d * std::exp(x[i]) / (s * s);

            // add prior
            res->at(i) += 2 * x[i] / ps_[i];
        }
    }

private:
    const std::vector<double> delta_, sigma_, ps_;
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

class NLLikeHMCTraits
{
public:
    NLLikeHMCTraits(int n, const NLLike& like, const NLLikeGrad& likeGrad, const std::vector<double>& mass, int burnin = 100) : n_(n), like_(like), likeGrad_(likeGrad), mass_(mass), burnin_(burnin), iter_(0), paramSum_(n, 0), paramSqSum_(n, 0), corSum_(n, 0), prev_(n, 0)
    {
        check(n_ > 0, "");
        check(mass_.size() == n_, "");
        check(burnin_ >= 0, "");

        std::stringstream outFileName;
        outFileName << "hmc_results/hmc_" << n_;
        if(CosmoMPI::create().numProcesses() > 1)
            outFileName << "_" << CosmoMPI::create().processId();
        outFileName << ".txt";
        out_.open(outFileName.str().c_str());
        if(!out_)
        {
            StandardException exc;
            std::stringstream excStr;
            excStr << "Cannot write into file " << outFileName.str() << ".";
            exc.set(excStr.str());
            throw exc;
        }
    }

    ~NLLikeHMCTraits()
    {
        out_.close();
    }

    int nPar() const { return n_; }
    void getStarting(std::vector<double> *x) const
    {
        x->clear();
        x->resize(n_, 0);
    }

    void getMasses(std::vector<double> *m) const
    {
        *m = mass_;
    }

    void set(const std::vector<double>& x) { check(x.size() == n_, ""); x_ = x; }
    void get(std::vector<double> *x) const { *x = x_; }
    double like() const { return like_.evaluate(x_); }
    void likeDerivatives(std::vector<double> *d) const { likeGrad_.evaluate(x_, d); }
    void output(const std::vector<double>& x, double like)
    {
        out_ << 1 << '\t' << like;
        check(x.size() == n_, "");
        for(int i = 0; i < n_; ++i)
            out_ << '\t' << x[i];
        out_ << std::endl;
        
        if(iter_ > burnin_)
        {
            for(int i = 0; i < n_; ++i)
            {
                paramSum_[i] += x[i];
                paramSqSum_[i] += x[i] * x[i];
                corSum_[i] += x[i] * prev_[i];
            }
        }

        prev_ = x;
        ++iter_;
    }

    bool stop() const
    {
        int doStop = 1;

        if(iter_ <= burnin_ + 10)
            doStop = 0;

        for(int i = 0; i < n_; ++i)
        {
            if(stdMean(i) > 0.25 * stdev(i))
                doStop = 0;
        }

        int allStop = 0;
#ifdef COSMO_MPI
        CosmoMPI::create().reduce(&doStop, &allStop, 1, CosmoMPI::INT, CosmoMPI::MIN);
#else
        allStop = doStop;
#endif
        CosmoMPI::create().bcast(&allStop, 1, CosmoMPI::INT);

        return allStop;
    }

private:

    double stdMean(int i) const
    {
        check(i >= 0 && i < n_, "");
        if(iter_ <= burnin_)
            return std::numeric_limits<double>::max();

        const double mean = paramSum_[i] / (iter_ - burnin_);
        const double meanSq = paramSqSum_[i] / (iter_ - burnin_);
        const double stdev = std::sqrt(meanSq - mean * mean);
        double stdMean = stdev / std::sqrt(double(iter_ - burnin_));
        const double cor = (corSum_[i] / (iter_ - burnin_) - mean * mean) / (stdev * stdev);
        if(cor < 1 && cor > -1)
            stdMean *= std::sqrt((1 + cor) / (1 - cor));
        return stdMean;
    }

    double mean(int i) const
    {
        check(i >= 0 && i < n_, "");
        if(iter_ <= burnin_)
            return std::numeric_limits<double>::max();

        const double mean = paramSum_[i] / (iter_ - burnin_);
        return mean;
    }

    double stdev(int i) const
    {
        check(i >= 0 && i < n_, "");
        if(iter_ <= burnin_)
            return std::numeric_limits<double>::max();

        const double mean = paramSum_[i] / (iter_ - burnin_);
        const double meanSq = paramSqSum_[i] / (iter_ - burnin_);
        const double stdev = std::sqrt(meanSq - mean * mean);
        return stdev;
    }

private:
    int n_;
    const NLLike& like_;
    const NLLikeGrad& likeGrad_;
    std::vector<double> x_;
    std::ofstream out_;
    const std::vector<double> mass_;
    const int burnin_;
    int iter_;

    std::vector<double> paramSum_, paramSqSum_, corSum_, prev_;
};

} // namespace

int main(int argc, char *argv[])
{
    int N = 32;
    const double L = 1;

    output_screen("Specify N as an argument or else it will be 32 by default. Specify \"out\" as an argument to write the iterations into a file. Specify \"hmc\" as an argument to run hmc instead of lbfgs." << std::endl);

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
    for(int i = 1; i < argc; ++i)
    {
        if(std::string(argv[i]) == std::string("out"))
            outIters = true;
        if(std::string(argv[i]) == std::string("hmc"))
            hmc = true;
    }

    int seed = 100 + 10 * CosmoMPI::create().processId();

    // power spectrum of 1 / N 
    //std::vector<double> pk(N, 1.0 / double(N));

    // generate modes
    std::vector<std::complex<double> > deltaK;
    //generateModes(pk, &deltaK, L, seed);

    //check(deltaK.size() == N / 2 + 1, "");

    // fourier transform
    std::vector<double> deltaX(N);
    //fftw_complex *dk = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2  + 1));
    for(int i = 0; i < N / 2 + 1; ++i)
    {
        //dk[i][0] = deltaK[i].real();
        //dk[i][1] = deltaK[i].imag();
    }
    //fftw_plan plan = fftw_plan_dft_c2r_1d(N, dk, &(deltaX[0]), FFTW_ESTIMATE);
    //fftw_execute(plan);
    //fftw_destroy_plan(plan);
    //fftw_free(dk);

    // generate modes directly in x space (overwrites the above)
    seed = 300;
    Math::GaussianGenerator gx(seed, 0, 1);
    std::vector<double> px(N, 1.0);

    // let's make sure that we have the exact same thing with or without mpi
    for(int i = 0; i < N * CosmoMPI::create().processId(); ++i)
        gx.generate();

    for(int i = 0; i < N; ++i)
        deltaX[i] = std::sqrt(px[i]) * gx.generate();

    // lognormal transform
    std::vector<double> deltaXLN(N);
    for(int i = 0; i < N; ++i)
        deltaXLN[i] = std::exp(deltaX[i]) - 1;

    // sigmaX is deltaXLN / 10
    std::vector<double> sigmaX(N);
    for(int i = 0; i < N; ++i)
    {
        sigmaX[i] = std::abs(deltaXLN[i]) / 10;

        // for whatever reason
        if(sigmaX[i] == 0)
            sigmaX[i] = 1;
    }

    // add noise
    const int noiseSeed = seed + 1;
    Math::GaussianGenerator gen(noiseSeed, 0, 1);
    std::vector<double> deltaXLNNoisy(N);
    //
    // let's make sure that we have the exact same thing with or without mpi
    for(int i = 0; i < N * CosmoMPI::create().processId(); ++i)
        gen.generate();
    for(int i = 0; i < N; ++i)
    {
        deltaXLNNoisy[i] = deltaXLN[i] + gen.generate() * sigmaX[i];
    }

    // likelihood and grad
    NLLike like(deltaXLNNoisy, sigmaX, px);
    NLLikeGrad likeGrad(deltaXLNNoisy, sigmaX, px);

    if(hmc)
    {
        std::vector<double> mass(N);
        for(int i = 0; i < N; ++i)
        {
            const double xErr = sigmaX[i] / (deltaXLNNoisy[i] + 1);
            mass[i] = 1.0 / (xErr * xErr) * N * CosmoMPI::create().numProcesses();
        }
        const int burnin = 100;
        NLLikeHMCTraits hmcTraits(N, like, likeGrad, mass, burnin);
        Timer timer("HMC");
        timer.start();
        Math::HMCGeneral<NLLikeHMCTraits> hmcGen(&hmcTraits, 1.0, 10);
        hmcGen.run(10000000);
        timer.end();

        std::stringstream chainFileName;
        chainFileName << "hmc_results/hmc_" << N;
        if(CosmoMPI::create().numProcesses() > 1)
            chainFileName << "_" << CosmoMPI::create().processId();
        chainFileName << ".txt";
        MarkovChain chain(chainFileName.str().c_str(), burnin);
        std::stringstream hmcPostFileName;
        hmcPostFileName << "hmc_results/hmc_" << N;
        if(CosmoMPI::create().numProcesses() > 1)
            hmcPostFileName << "_" << CosmoMPI::create().processId();
        hmcPostFileName << "_post.txt";
        std::ofstream outHMCPost(hmcPostFileName.str().c_str());
        StandardException exc;
        if(!outHMCPost)
        {
            std::stringstream exceptionStr;
            exceptionStr << "Cannot write into file " << hmcPostFileName.str() << ".";
            exc.set(exceptionStr.str());
            throw exc;
        }
        for(int i = 0; i < N; ++i)
        {
            std::unique_ptr<Posterior1D> p(chain.posterior(i));
            const double median = p->median();
            double l1, l2, u1, u2;
            p->get1SigmaTwoSided(l1, u1);
            p->get2SigmaTwoSided(l2, u2);

            std::stringstream postFileName;
            postFileName << "hmc_results/hmc_" << N;
            postFileName << "_par_" << i << ".txt";
            if(CosmoMPI::create().numProcesses() > 1)
                postFileName << "_" << CosmoMPI::create().processId();
            //p->writeIntoFile(postFileName.str().c_str());


            // calculate actual;
            const double xMin = -5 * std::sqrt(px[i]);
            const double xMax = -xMin;
            const int n = 10000;
            const double d = (xMax - xMin) / n;
            std::vector<double> cumul(n + 1, 0);
            for(int j = 1; j <= n; ++j)
            {
                const double x = xMin + j * d;
                const double y = std::exp(x) - 1;
                const double dy = (y - deltaXLNNoisy[i]);
                const double f = std::exp(-x * x / (2 * px[i]) - dy * dy / (2 * sigmaX[i] * sigmaX[i]));
                cumul[j] = cumul[j - 1] + f * d;
            }
            const double norm = cumul[n];

            const double realMed = (std::lower_bound(cumul.begin(), cumul.end(), 0.5 * norm) - cumul.begin()) * d + xMin;
            const double realL1 = (std::lower_bound(cumul.begin(), cumul.end(), (1 - 0.683) / 2 * norm) - cumul.begin()) * d + xMin;
            const double realL2 = (std::lower_bound(cumul.begin(), cumul.end(), (1 - 0.955) / 2 * norm) - cumul.begin()) * d + xMin;
            const double realU1 = (std::lower_bound(cumul.begin(), cumul.end(), (1 - (1 - 0.683) / 2) * norm) - cumul.begin()) * d + xMin;
            const double realU2 = (std::lower_bound(cumul.begin(), cumul.end(), (1 - (1 - 0.955) / 2) * norm) - cumul.begin()) * d + xMin;
            outHMCPost << i << '\t' << median << '\t' << l1 << '\t' << u1 << '\t' << l2 << '\t' << u2 << '\t' << realMed << '\t' << realL1 << '\t' << realU1 << '\t' << realL2 << '\t' << realU2 << '\t' << std::endl;
        }
        outHMCPost.close();
    }
    else
    {
        Math::LBFGS lbfgs(N, like, likeGrad, 100);
        std::vector<double> x(N, 0);
        const double epsilon = 1e-10;
        const double gradTol = 1e-5 * N * CosmoMPI::create().numProcesses();

        if(outIters)
        {
            std::stringstream fileName;
            fileName << "lbfgs_iters";
            if(CosmoMPI::create().numProcesses() > 1)
                fileName << "_" << CosmoMPI::create().processId();
            fileName << ".txt";
            std::ofstream out(fileName.str().c_str());
            out.close();
        }

        Timer timer("LBFGS");
        timer.start();
        if(outIters)
            lbfgs.minimize(&x, epsilon, gradTol, 10000000, lbfgsCallbackFunc);
        else
            lbfgs.minimize(&x, epsilon, gradTol, 10000000);
        timer.end();

        const double funcMin = like.evaluate(x);
        if(CosmoMPI::create().isMaster())
        {
            output_screen("Function mimimum is: " << funcMin << std::endl);
        }

        // write deltaX, deltaXLN, deltaXLNNoisy, and x into a file
        std::stringstream deltaFileName;
        deltaFileName << "delta_x";
        if(CosmoMPI::create().numProcesses() > 1)
            deltaFileName << "_" << CosmoMPI::create().processId();
        deltaFileName << ".txt";
        std::ofstream out(deltaFileName.str().c_str());
        for(int i = 0; i < N; ++i)
            out << i << '\t' << deltaX[i] << '\t' << deltaXLN[i] << '\t' << deltaXLNNoisy[i] << '\t' << x[i] << std::endl;
        out.close();
    }

    return 0;
}
