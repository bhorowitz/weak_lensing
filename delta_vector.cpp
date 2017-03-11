#include <macros.hpp>
#include <exception_handler.hpp>
#include <cosmo_mpi.hpp>

#include "delta_vector.hpp"

DeltaVector3::DeltaVector3(int n1, int n2, int n3, bool isComplex) : isComplex_(isComplex), x_(isComplex ? 0 : n1 * n2 * n3), c_(isComplex ? n1 * n2 * (n3 / 2 + 1) : 0), n1_(n1), n2_(n2), n3_(n3)
{
}

void
DeltaVector3::copy(const DeltaVector3& other, double c)
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

    extraParams_.resize(other.extraParams_.size());
    for(int i = 0; i < extraParams_.size(); ++i)
        extraParams_[i] = other.extraParams_[i] * c;
}

void
DeltaVector3::setToZero()
{
    for(auto it = x_.begin(); it != x_.end(); ++it)
        *it = 0;

    for(auto it = c_.begin(); it != c_.end(); ++it)
        *it = std::complex<double>(0, 0);

    for(double& x : extraParams_)
        x = 0;
}

double
DeltaVector3::dotProduct(const DeltaVector3& other) const
{
    double res = 0;
    if(isComplex_)
    {
        check(other.n1_ == n1_, "");
        check(other.n2_ == n2_, "");
        check(other.n3_ == n3_, "");
        for(int k = 0; k < n3_ / 2 + 1; ++k)
        {
            const int jMax = (k > 0 && k < n3_ / 2 ? n2_ : n2_ / 2 + 1);
            for(int j = 0; j < jMax; ++j)
            {
                int iMax = n1_;
                if((k == 0 || k == n3_ / 2) && (j == 0 || j == n3_ / 2))
                    iMax - n1_ / 2 + 1;
                for(int i = 0; i < iMax; ++i)
                {
                    const int index = (i * n2_ + j) * (n3_ / 2 + 1) + k;
                    res += (std::real(c_[index]) * std::real(other.c_[index]) + std::imag(c_[index]) * std::imag(other.c_[index]));
                }
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

    check(extraParams_.size() == other.extraParams_.size(), "");
    for(int i = 0; i < extraParams_.size(); ++i)
        total += extraParams_[i] * other.extraParams_[i];
    return total;
}

void
DeltaVector3::add(const DeltaVector3& other, double c)
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

    check(extraParams_.size() == other.extraParams_.size(), "");
    for(int i = 0; i < extraParams_.size(); ++i)
        extraParams_[i] += c * other.extraParams_[i];
}

void
DeltaVector3::swap(DeltaVector3& other)
{
    check(other.c_.size() == c_.size(), "");
    c_.swap(other.c_);

    check(other.x_.size() == x_.size(), "");
    x_.swap(other.x_);

    check(extraParams_.size() == other.extraParams_.size(), "");
    extraParams_.swap(other.extraParams_);
}
