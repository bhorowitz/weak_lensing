#include <vector>

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
    fftw_plan backPlan = fftw_plan_dft_c2r_1d(n, out, &(back[0]), FFTW_PRESERVE_INPUT | FFTW_ESTIMATE);

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
    return 0;
}
