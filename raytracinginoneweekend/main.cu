#include <iostream>
#include <time.h>

__global__ void div(float *buffer, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x*3 + i*3;
    buffer[pixel_index + 0] = float(i) / max_x;//r
    buffer[pixel_index + 1] = float(j) / max_y;//g
    buffer[pixel_index + 2] = 0.2;//b
}

int main() {
    int nx = 1200;
    int ny = 600;
    int num_pixels = nx*ny;
    size_t buffer_size = 3*num_pixels*sizeof(float);
    float *buffer;

    dim3 block_size(nx/9,ny/9);//tamanho de cada bloco
    dim3 size_grid(8,8);//divisao da imagem
    div<<<size_grid, block_size>>>(buffer, nx, ny);//chama a gpu

    
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
    cudaFree(buffer)
}