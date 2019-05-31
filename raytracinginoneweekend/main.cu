#include <iostream>
#include <time.h>
#include "vec3.h"

__global__ void render(vec3 *buffer, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    buffer[pixel_index] = vec3( float(i) / max_x, float(j) / max_y, 0.2f);
}

int main() {
    int nx = 1200;
    int ny = 600;

    int num_pixels = nx*ny;
    size_t buffer_size = 3*num_pixels*sizeof(vec3);
    vec3 *buffer;
    cudaMallocManaged((void **)&buffer, buffer_size)

    clock_t start, stop;
    start = clock();
    dim3 block_size(nx/9,ny/9);//tamanho de cada bloco
    dim3 size_grid(8,8);//divisao da imagem
    div<<<size_grid, block_size>>>(buffer, nx, ny);//chama a gpu
    cudaDeviceSynchronize()

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "time:" << timer_seconds;

    
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*3*nx + i*3;
            float r = buffer[pixel_index + 0];
            float g = buffer[pixel_index + 1];
            float b = buffer[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    cudaFree(buffer);
}