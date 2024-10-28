#include "cuda_raytracer.h"
#include "raytracer.h"
#include "utils.h"
#include <cuda_runtime.h>

__device__ Vec3 trace(const Ray &ray, const Sphere *spheres, int numSpheres) {
  float t_min = FLT_MAX;
  const Sphere *hit_sphere = nullptr;
  for (int i = 0; i < numSpheres; ++i) {
    float t;
    bool hit = spheres[i].intersect(ray, t);
    if (hit && t < t_min) {
      t_min = t;
      hit_sphere = &spheres[i];
    }
  }
  if (hit_sphere) {
    Vec3 hit_point = ray.origin + ray.direction * t_min;
    Vec3 normal = (hit_point - hit_sphere->center).normalize();
    return (normal + Vec3(1, 1, 1)) * 0.5f; // Simple shading
  }
  return Vec3(0.5f, 0.7f, 1.0f); // Background color
}

__global__ void renderKernel(Vec3 *framebuffer, int width, int height,
                             Vec3 camera_origin, float viewport_width,
                             float viewport_height, float focal_length,
                             const Sphere *spheres, int numSpheres) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float u = (double(x) / (width - 1)) * viewport_width - viewport_width / 2;
  float v = (double(y) / (height - 1)) * viewport_height - viewport_height / 2;
  Vec3 direction = Vec3(u, v, -focal_length).normalize();
  Ray ray(camera_origin, direction);

  framebuffer[y * width + x] = trace(ray, spheres, numSpheres);
}
