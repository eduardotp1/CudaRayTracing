//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
// This code was change following the tutorial of Roger Allen https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/
//==================================================================================================

#include <iostream>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"
#include "vec3.h"
#include "ray.h"
#include <chrono>

// traca os raios de luz
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

// hitable *random_scene() {
//     int n = 500;
//     hitable **list = new hitable*[n+1];
//     list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
//     int i = 1;
//     for (int a = -11; a < 11; a++) {
//         for (int b = -11; b < 11; b++) {
//             float choose_mat = drand48();
//             vec3 center(a+0.9*drand48(),0.2,b+0.9*drand48()); 
//             if ((center-vec3(4,0.2,0)).length() > 0.9) { 
//                 if (choose_mat < 0.8) {  // diffuse
//                     list[i++] = new sphere(center, 0.2, new lambertian(vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
//                 }
//                 else if (choose_mat < 0.95) { // metal
//                     list[i++] = new sphere(center, 0.2,
//                             new metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())),  0.5*drand48()));
//                 }
//                 else {  // glass
//                     list[i++] = new sphere(center, 0.2, new dielectric(1.5));
//                 }
//             }
//         }
//     }

//     list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
//     list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
//     list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

//     return new hitable_list(list,i);
// }

// pinta a imagem
__global__ void rgb(vec3 *fb, int max_x, int max_y,vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hitable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    fb[pixel_index] = color(r, world);
}

// instancia as esferas
__global__ void create_sphere(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *(d_list+2)   = new sphere(vec3(0,10,-1), 15);
        *(d_list+3) = new sphere(vec3(0,20,-1), 10);
        *(d_list+4)   = new sphere(vec3(0,2,-1), 3);
        *(d_list+5) = new sphere(vec3(0,30,-1), 2);
        *d_world    = new hitable_list(d_list,6);
    }
}
// deleta memorias
__global__ void free_memory(hitable **d_list, hitable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

int main() {
    int nx = 1200;
    int ny = 800;
    // int ns = 10;
    int tx = 8;//divisoes que vai ser cortada a imagem
    int ty = 8;//divisoes que vai ser cortada a imagem
    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);
    using namespace std::chrono;
    high_resolution_clock::time_point begin = high_resolution_clock::now();

    // allocate FB
    vec3 *fb;
    cudaMallocManaged((void **)&fb, fb_size);
    hitable **d_list;
    cudaMalloc((void **)&d_list, 6*sizeof(hitable *));
    hitable **d_world;
    cudaMalloc((void **)&d_world, sizeof(hitable *));
    create_sphere<<<1,1>>>(d_list,d_world);
    cudaDeviceSynchronize();

    dim3 block_size(nx/tx+1,ny/ty+1);//tamanho de cada grid
    dim3 size_grid(tx,ty);//tamanho do grid
    rgb<<<block_size, size_grid>>>(fb, nx, ny, vec3(-2.0, -1.0, -1.0), vec3(4.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), vec3(0.0, 0.0, 0.0),d_world);//manda para a GPU calcular
    cudaDeviceSynchronize();

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    // hitable *list[5];

    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;

            // size_t pixel_index = j*3*nx + i*3;
            int ir = int(255.99*fb[pixel_index][0]);
            int ig = int(255.99*fb[pixel_index][1]);
            int ib = int(255.99*fb[pixel_index][2]);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(end - begin);
    std::cerr << "Tempo: " << time_span.count();

    cudaDeviceSynchronize();
    free_memory<<<1,1>>>(d_list,d_world);
    cudaFree(d_list);
    cudaFree(d_world);
    cudaFree(fb);
    cudaDeviceReset();
}