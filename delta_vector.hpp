#ifndef LA_DELTA_K_VECTOR_HPP
#define LA_DELTA_K_VECTOR_HPP

#include <vector>
#include <complex>

class DeltaVector3
{
public:
    DeltaVector3(int n1 = 0, int n2 = 0, int n3 = 0, bool isComplex = false);

    std::vector<double>& get() { return x_; }
    const std::vector<double>& get() const { return x_; }

    std::vector<std::complex<double> >& getComplex() { return c_; }
    const std::vector<std::complex<double> >& getComplex() const { return c_; }

    bool isComplex() const { return isComplex_; }

    // copy from other, multiplying with coefficient (for MPI, the correct coefficient should be passed for EVERY process)
    void copy(const DeltaVector3& other, double c = 1.);

    // set all the elements to 0
    void setToZero();

    // get the norm (for MPI, the master process should get the total norm)
    double norm() const { return std::sqrt(dotProduct(*this)); }

    // dot product with another vector (for MPI, the master process should get the total norm)
    double dotProduct(const DeltaVector3& other) const;

    // add another vector with a given coefficient (for MPI, the correct coefficient should be passed for EVERY process)
    void add(const DeltaVector3& other, double c = 1.);

    // swap
    void swap(DeltaVector3& other);

private:
    bool isComplex_;
    const int n1_, n2_, n3_;
    std::vector<double> x_;
    std::vector<std::complex<double> > c_;
};

class DeltaVector3Factory
{
public:
    DeltaVector3Factory(int N1, int N2, int N3, bool isComplex) : N1_(N1), N2_(N2), N3_(N3), isComplex_(isComplex)
    {
        check(N1_ > 0, "");
        check(N2_ > 0, "");
        check(N3_ > 0, "");
    }

    // create a new DeltaVector with 0 elements
    // the user is in charge of deleting it
    DeltaVector3* giveMeOne()
    {
        return new DeltaVector3(N1_, N2_, N3_, isComplex_);
    }
private:
    const int N1_, N2_, N3_;
    const bool isComplex_;
};

#endif

