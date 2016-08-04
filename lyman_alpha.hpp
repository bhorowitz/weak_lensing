#ifndef COSMO_PP_LYMAN_ALPHA_HPP
#define COSMO_PP_LYMAN_ALPHA_HPP

#include <vector>
#include <complex>

class LymanAlpha
{
public:
    LymanAlpha(const std::vector<double>& delta, double L = 1, double b = 0.01);
    ~LymanAlpha();

    void reset(const std::vector<double>& delta);

    const std::vector<double>& getDelta() const { return delta_; }
    const std::vector<double>& getDeltaNonLin() const { return deltaLN_; }
    const std::vector<double>& getV()
    {
        if(!vCalculated_)
            calculateV();
        return v_;
    }
    const std::vector<double>& getTau()
    {
        if(!tauCalculated_)
            calculateTau();
        return tau_;
    }
    const std::vector<double>& getFlux()
    {
        if(!fluxCalculated_)
            calculateFlux();
        return flux_;
    }

    /// partial deltaNonLin_n / partial delta_n (non-diagonal terms are 0 in the lognormal model)
    double deltaDeriv(int n) const;

    /// partial v_m / partial delta_n
    double vDeriv(int m, int n);

    /// partial tau_m / partial delta_n
    double tauDeriv(int m, int n);

    /// partial flux_m / partial delta_n
    double fluxDeriv(int m, int n);

private:
    void calculateV();
    void calculateTau();
    void calculateTauDerivs();
    void calculateFlux();

    inline int periodicIndex(int i) const
    {
        if(i < 0)
        {
            const int j = -i;
            const int n = (j - 1) / N_ + 1;
            i += n * N_;
        }
        check(i >= 0, "");
        return i % N_;
    }

private:
    const double L_;
    const double b_;
    const int N_;
    std::vector<double> delta_;
    std::vector<double> deltaLN_;
    std::vector<double> v_;
    std::vector<double> vDeriv_;
    std::vector<double> tau_;
    std::vector<double> flux_;
    std::vector<std::vector<double> > tauDerivs_;

    std::vector<std::complex<double> > complexBuffer_;

    bool vCalculated_;
    bool tauCalculated_;
    bool fluxCalculated_;
    bool tauDerivsCalculated_;
};

class LymanAlpha2
{
public:
    LymanAlpha2(int N1, int N2, const std::vector<double>& delta, double L1 = 1, double L2 = 1, double b = 0.01);
    ~LymanAlpha2();

    void reset(const std::vector<double>& delta);

    const std::vector<double>& getDelta() const { return delta_; }
    const std::vector<double>& getDeltaNonLin() const { return deltaLN_; }
    const std::vector<double>& getV()
    {
        if(!vCalculated_)
            calculateV();
        return v_;
    }
    const std::vector<double>& getTau()
    {
        if(!tauCalculated_)
            calculateTau();
        return tau_;
    }
    const std::vector<double>& getFlux()
    {
        if(!fluxCalculated_)
            calculateFlux();
        return flux_;
    }

    /// partial deltaNonLin_n / partial delta_n (non-diagonal terms are 0 in the lognormal model)
    double deltaDeriv(int n) const;

    /// partial v_(m1, m2) / partial delta_(n1, n2)
    double vDeriv(int m1, int m2, int n1, int n2);

    /// partial tau_(m1, m2) / partial delta_(n1, n2)
    double tauDeriv(int m1, int m2, int n1, int n2);

    /// partial flux_(m1, m2) / partial delta_(n1, n2)
    double fluxDeriv(int m1, int m2, int n1, int n2);

private:
    void calculateV();
    void calculateTau();
    void calculateTauDerivs();
    void calculateFlux();

    inline int periodicIndex(int i) const
    {
        if(i < 0)
        {
            const int j = -i;
            const int n = (j - 1) / N2_ + 1;
            i += n * N2_;
        }
        check(i >= 0, "");
        return i % N2_;
    }

private:
    const double L1_, L2_;
    const double b_;
    const int N1_, N2_;
    std::vector<double> delta_;
    std::vector<double> deltaLN_;
    std::vector<double> v_;
    std::vector<double> vDeriv_;
    std::vector<double> tau_;
    std::vector<double> flux_;
    std::vector<std::vector<std::vector<double> > > tauDerivsDeltaX_, tauDerivsV_;

    std::vector<std::complex<double> > complexBuffer_;

    bool vCalculated_;
    bool tauCalculated_;
    bool fluxCalculated_;
    bool tauDerivsCalculated_;
};

class LymanAlpha3
{
public:
    LymanAlpha3(int N1, int N2, int N3, const std::vector<double>& delta, double L1 = 1, double L2 = 1, double L3 = 1, double b = 0.01);
    ~LymanAlpha3();

    void reset(const std::vector<double>& delta);

    const std::vector<double>& getDelta() const { return delta_; }
    const std::vector<double>& getDeltaNonLin() const { return deltaLN_; }
    const std::vector<double>& getV()
    {
        if(!vCalculated_)
            calculateV();
        return v_;
    }
    const std::vector<double>& getTau()
    {
        if(!tauCalculated_)
            calculateTau();
        return tau_;
    }
    const std::vector<double>& getFlux()
    {
        if(!fluxCalculated_)
            calculateFlux();
        return flux_;
    }

    /// partial deltaNonLin_n / partial delta_n (non-diagonal terms are 0 in the lognormal model)
    double deltaDeriv(int n) const;

    /// partial v_(m1, m2, m3) / partial delta_(n1, n2, n3)
    double vDeriv(int m1, int m2, int m3, int n1, int n2, int n3);

    /// partial tau_(m1, m2, m3) / partial delta_(n1, n2, n3)
    double tauDeriv(int m1, int m2, int m3, int n1, int n2, int n3);

    /// partial flux_(m1, m2, m3) / partial delta_(n1, n2, n3)
    double fluxDeriv(int m1, int m2, int m3, int n1, int n2, int n3);

    /// partial tau_(m1, m2, m3) / partial v_(m1, m2, k)
    double tauDerivV(int m1, int m2, int m3, int k);

    /// partial tau_(m1, m2, m3) / partial delta_(m1, m2, k). NOTE: This excludes partial tau / partial v * partial v / partial delta
    double tauDerivDeltaOnly(int m1, int m2, int m3, int k);

    /// partial flux_(m1, m2, m3) / partial v_(m1, m2, k)
    double fluxDerivV(int m1, int m2, int m3, int k);

    /// partial flux_(m1, m2, m3) / partial delta_(m1, m2, k). NOTE: This excludes partial flux / partial v * partial v / partial delta
    double fluxDerivDeltaOnly(int m1, int m2, int m3, int k);

    static constexpr double growthFactor() { return 0.5; }

private:
    void calculateV();
    void calculateTau();
    void calculateTauDerivs();
    void calculateFlux();

    inline int periodicIndex(int i) const
    {
        if(i < 0)
        {
            const int j = -i;
            const int n = (j - 1) / N3_ + 1;
            i += n * N3_;
        }
        check(i >= 0, "");
        return i % N3_;
    }

private:
    const double L1_, L2_, L3_;
    const double b_;
    const int N1_, N2_, N3_;
    std::vector<double> delta_;
    std::vector<double> deltaLN_;
    std::vector<double> v_;
    std::vector<double> vDeriv_;
    std::vector<double> tau_;
    std::vector<double> flux_;
    std::vector<std::vector<std::vector<double> > > tauDerivsDeltaX_, tauDerivsV_;

    std::vector<std::complex<double> > complexBuffer_;

    bool vCalculated_;
    bool tauCalculated_;
    bool fluxCalculated_;
    bool tauDerivsCalculated_;
    double tauFactor_;
};

#endif

