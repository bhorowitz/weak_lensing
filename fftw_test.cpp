#include <vector>
#include <complex>

#include <macros.hpp>

#include <fftw3.h>

int main()
{
    int n = 3;
    std::vector<double> in(n);
    in[0] = 1;
    in[1] = 2;
    in[2] = 3;
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (n / 2  + 1));
    std::vector<double> back(n);

    fftw_plan fwdPlan = fftw_plan_dft_r2c_1d(n, &(in[0]), out, FFTW_ESTIMATE);
    check(fwdPlan, "");
    fftw_plan backPlan = fftw_plan_dft_c2r_1d(n, out, &(back[0]), FFTW_PRESERVE_INPUT | FFTW_ESTIMATE);
    check(backPlan, "");

    fftw_execute(fwdPlan);
    fftw_execute(backPlan);

    output_screen("Input:" << std::endl);
    for(int i = 0; i < n; ++i)
        output_screen(i << '\t' << in[i] << std::endl);

    output_screen("Output:" << std::endl);
    for(int i = 0; i < n / 2 + 1; ++i)
        output_screen(i << '\t' << out[i][0] << " + " << out[i][1] << " i" << std::endl);

    output_screen("Back:" << std::endl);
    for(int i = 0; i < n; ++i)
        output_screen(i << '\t' << back[i] << std::endl);

    fftw_destroy_plan(fwdPlan);
    fftw_destroy_plan(backPlan);
    fftw_free(out);

    const int N = 2;
    std::vector<std::complex<double> > c(N * (N / 2 + 1), std::complex<double>(1, 1));
    std::vector<double> x(N * N);
    fftw_plan backPlan2 = fftw_plan_dft_c2r_2d(N, N, reinterpret_cast<fftw_complex*>(&(c[0])), &(x[0]), FFTW_ESTIMATE);
    check(backPlan2, "");
    fftw_execute(backPlan2);
    fftw_destroy_plan(backPlan2);
    output_screen(x[0] << '\t' << x.back() << std::endl);

    return 0;
}
