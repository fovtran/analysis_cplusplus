#include <iostream>
#include <arrayfire.h>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>

const int N = 2;
const int GRID_DIM = 1024;
const double CELL_SIZE = 100.0;
const double G = 1.0;
const double DT = 0.001;

struct Vec2
{
    float x, y;
    Vec2(float x = 0, float y = 0) :
        x(x), y(y) { }
};

int wrapIndex(int i, int i_max)
{
    return ((i % i_max) + i_max) % i_max;
}


af::array makeGrid(const Vec2* p, const float* m)
{
    float* grid_h = new float[GRID_DIM * GRID_DIM];
    memset(grid_h, 0, sizeof(float) * GRID_DIM * GRID_DIM);

    for(int i = 0; i < N; i++)
    {
        int x = wrapIndex(roundf(p[i].x / CELL_SIZE), GRID_DIM);
        int y = wrapIndex(roundf(p[i].y / CELL_SIZE), GRID_DIM);
        grid_h[y * GRID_DIM + x] += m[i];
    }

    af::array grid(GRID_DIM, GRID_DIM, grid_h);
    delete[] grid_h;
    return grid;
}


int main()
{
    af::setBackend(AF_BACKEND_CUDA);
    af::info();
    Vec2* p = new Vec2[N];
    Vec2* v = new Vec2[N];
    float* m = new float[N];

    std::fill(v, v + N, 0.0f);
    std::fill(m, m + N, 1.0f);
//  for(int i = 0; i < N; i++)
//      p[i] = Vec2(rand() % GRID_DIM / 5 + GRID_DIM / 2, rand() % GRID_DIM / 5 + GRID_DIM / 2);

    //add one massive object at the center
    m[0] = 1000000.0;
    p[0] = Vec2(GRID_DIM * CELL_SIZE / 2, GRID_DIM * CELL_SIZE / 2);

    m[1] = 0.0;
    p[1] = Vec2(GRID_DIM * CELL_SIZE / 4, GRID_DIM * CELL_SIZE / 4);

    af::array rng = af::range(GRID_DIM);
    af::array k2 = 2.0 * af::Pi * af::select(rng > GRID_DIM / 2, (rng - GRID_DIM) , rng) / (GRID_DIM * CELL_SIZE);
    //af::array k2 = 2.0 * af::Pi * rng / (GRID_DIM * CELL_SIZE);
    k2(0) = 1.0;
    k2 = af::tile((k2 * k2).T(), GRID_DIM) + af::tile(k2 * k2, 1, GRID_DIM);

    af::Window wnd(GRID_DIM, GRID_DIM);

    //load the grid with bodies
    af::array grid = makeGrid(p, m);
    while(!wnd.close())
    {
        af::timer t = af::timer::start();

        //fft the density grid
        grid = af::fft2(grid);

        //apply the formula from the paper
        grid *= -G / (af::Pi * k2);
        //invert the fft to get the potentials
        grid = af::ifft2(grid);
        grid = af::abs(grid);

        //compute the gradients in the potential
        af::array dx, dy;
        af::grad(dx, dy, grid);

        float* dx_h = dx.host<float>();
        float* dy_h = dy.host<float>();

        //use the gradients to time step each of the bodies
        #pragma omp parallel for simd
        for(int i = 0; i < N; i++)
        {
            int xi = wrapIndex(roundf(p[i].x / CELL_SIZE), GRID_DIM);
            int yi = wrapIndex(roundf(p[i].y / CELL_SIZE), GRID_DIM);

            v[i].x += dx_h[yi * GRID_DIM + xi] * DT / m[i];
            v[i].y += dy_h[yi * GRID_DIM + xi] * DT / m[i];
            p[i].x += v[i].x * DT;
            p[i].y += v[i].y * DT;
        }
        af::freeHost(dx_h);
        af::freeHost(dy_h);

        //display the potential

        wnd.image(af::abs(grid) / af::max<float>(af::abs(grid)));
        //remake the new grid
        grid = makeGrid(p, m);

        std::cout << "dt = " << af::timer::stop(t) << std::endl;

    }

    delete[] p;
    delete[] v;
    delete[] m;
    return 0;
}