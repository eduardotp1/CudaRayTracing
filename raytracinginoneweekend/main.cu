#include <iostream>
#include <time.h>
#include <float.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"



__device__ vec3 color(const ray& r, hitable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

__global__ void div(vec3 *buffer, int max_x, int max_y,
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
                       hitable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    buffer[pixel_index] = color(r, world);
}

__global__ void create_world(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hitable_list(d_list,2);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

int main() {
    int nx = 1200;
    int ny = 600;
    int num_pixels = nx*ny;
    size_t buffer_size = num_pixels*sizeof(vec3);

    vec3 *buffer;
    cudaMallocManaged((void **)&buffer, buffer_size);

    hitable **d_list;
    cudaMalloc((void **)&d_list, 2*sizeof(hitable *));
    hitable **d_world;
    cudaMalloc((void **)&d_world, sizeof(hitable *)));
    create_world<<<1,1>>>(d_list,d_world);
    cudaGetLastError();
    cudaDeviceSynchronize();

    clock_t start, stop;
    start = clock();
    
    dim3 block_size(nx/9,ny/9);
    dim3 size_grid(8,8);
    div<<<block_size, size_grid>>>(buffer, nx, ny,
                                vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0),
                                d_world);
    cudaGetLastError();
    cudaDeviceSynchronize();
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "time:" << timer_seconds;

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*buffer[pixel_index].r());
            int ig = int(255.99*buffer[pixel_index].g());
            int ib = int(255.99*buffer[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    cudaDeviceSynchronize();
    free_world<<<1,1>>>(d_list,d_world);
    cudaGetLastError();
    cudaFree(d_list);
    cudaFree(d_world);
    cudaFree(buffer);
    cudaDeviceReset();
}