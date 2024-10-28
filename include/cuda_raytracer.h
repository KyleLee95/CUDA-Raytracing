#ifndef CUDA_RAYTRACER_H
#define CUDA_RAYTRACER_H

#include "raytracer.h"

__global__ void renderKernel(Vec3 *framebuffer, int width, int height,
                             Vec3 camera_origin, float viewport_width,
                             float viewport_height, float focal_length,
                             const Sphere *spheres, int numSpheres);

#endif // CUDA_RAYTRACER_H
