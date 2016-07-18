#ifndef LA_UTILS_HPP
#define LA_UTILS_HPP

#include <vector>
#include <fstream>

#include <macros.hpp>
#include <exception_handler.hpp>
#include <matrix.hpp>
#include <matrix_impl.hpp>
#include <function.hpp>
#include <table_function.hpp>
#include <math_constants.hpp>

class PSDiff : public Math::RealFunction
{
public:
    PSDiff(const Math::RealFunction& a, const Math::RealFunction& b) : a_(a), b_(b) {}
    virtual double evaluate(double x) const
    {
        const double diff = a_.evaluate(x) - b_.evaluate(x);
        if(diff > 0)
            return diff;
        return 0;
    }
private:
    const Math::RealFunction& a_;
    const Math::RealFunction& b_;
};

inline void writePSToFile(const char *fileName, const Math::RealFunction& pk, double kMin, double kMax, int n = 10000)
{
    check(kMin >= 0, "");
    check(kMax > kMin, "");
    check(n > 1, "");

    StandardException exc;
    std::ofstream out(fileName);
    if(!out)
    {
        std::stringstream exceptionStr;
        exceptionStr << "Cannot write into file " << fileName << ".";
        exc.set(exceptionStr.str());
        throw exc;
    }

    const double deltaK = (kMax - kMin) / n;
    for(int i = 0; i <= n; ++i)
    {
        double k = kMin + deltaK * i;
        if(i == n)
            k = kMax;

        out << k << '\t' << pk.evaluate(k) << std::endl;
    }
    out.close();
}

inline Math::TableFunction<double, double>* psFromFile(const char* fileName)
{
    std::ifstream in(fileName);
    StandardException exc;
    if(!in)
    {
        std::stringstream exceptionStr;
        exceptionStr << "Cannot read from file " << fileName << ".";
        exc.set(exceptionStr.str());
        throw exc;
    }

    Math::TableFunction<double, double>* f = new Math::TableFunction<double, double>;
    while(!in.eof())
    {
        std::string s;
        std::getline(in, s);
        if(s[0] == '#')
            continue;

        if(s == "")
            break;

        std::stringstream str(s);
        double k, v;
        str >> k >> v;
        (*f)[k] = v;
    }

    return f;
}

template<typename T>
void vector2file(const char *fileName, const std::vector<T> &v)
{
    StandardException exc;
    std::ofstream out(fileName);
    if(!out)
    {
        std::stringstream exceptionStr;
        exceptionStr << "Cannot write into file " << fileName << ".";
        exc.set(exceptionStr.str());
        throw exc;
    }

    for(int i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if(i < v.size() - 1)
            out << '\t';
    }
    out << std::endl;
    out.close();
}

template<typename T>
void matrix2file(const char *fileName, const Math::Matrix<T> &m)
{
    StandardException exc;
    std::ofstream out(fileName);
    if(!out)
    {
        std::stringstream exceptionStr;
        exceptionStr << "Cannot write into file " << fileName << ".";
        exc.set(exceptionStr.str());
        throw exc;
    }

    for(int i = 0; i < m.rows(); ++i)
    {
        for(int j = 0; j < m.cols(); ++j)
        {
            out << m(i, j);
            if(j < m.cols() - 1)
                out << '\t';
        }
        out << std::endl;
    }
    out.close();
}

inline void vector2binFile(const char *fileName, const std::vector<double>& v)
{
    StandardException exc;
    std::ofstream out(fileName, std::ios::out | std::ios::binary);
    if(!out)
    {
        std::stringstream exceptionStr;
        exceptionStr << "Cannot write into file " << fileName << ".";
        exc.set(exceptionStr.str());
        throw exc;
    }
    out.write(reinterpret_cast<char*>(const_cast<double*>(&(v[0]))), v.size() * sizeof(double));
    out.close();
}

inline void getC2S2(int N1, int N2, double L1, double L2, int i, int j, double *c2, double *s2)
{
    check(N1 > 0, "");
    check(N2 > 0, "");
    check(L1 > 0, "");
    check(L2 > 0, "");
    check(i >= 0 && i < N1, "");
    check(j >= 0 && j < N2 / 2 + 1, "");

    const int iShifted = (i < (N1 + 1) / 2 ? i : -(N1 - i));
    const double k1 = 2 * Math::pi / L1 * iShifted;
    int jShifted = (j < (N2 + 1) / 2 ? j : -(N2 - j));
    if(j == N2 / 2 && i > N1 / 2)
        jShifted = -jShifted;
    const double k2 = 2 * Math::pi / L2 * jShifted;
    const double k = std::sqrt(k1 * k1 + k2 * k2);
    if(k == 0)
    {
        *c2 = 0;
        *s2 = 0;
        return;
    }

    check(k > 0, "");
    const double c = k1 / k, s = k2 / k;
    *c2 = 2 * c * c - 1;
    *s2 = 2 * c * s;
}

#endif

