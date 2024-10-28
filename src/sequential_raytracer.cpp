#include "raytracer.h"
#include "utils.h"
#include <vector>

Vec3 trace(const Ray &ray, const std::vector<Sphere> &spheres) {
  float t_min = FLT_MAX;
  const Sphere *hit_sphere = nullptr;
  for (const auto &sphere : spheres) {
    float t;
    if (sphere.intersect(ray, t) && t < t_min) {
      t_min = t;
      hit_sphere = &sphere;
    }
  }
  if (hit_sphere) {
    Vec3 hit_point = ray.origin + ray.direction * t_min;
    Vec3 normal = (hit_point - hit_sphere->center).normalize();
    return (normal + Vec3(1, 1, 1)) * 0.5f; // Simple shading
  }
  return Vec3(0.5f, 0.7f, 1.0f); // Background color
}

void render(int width, int height, const std::string &sceneFile,
            const std::string &outputFile) {
  std::vector<Sphere> spheres;
  loadScene(spheres, sceneFile);
  std::vector<Vec3> framebuffer(width * height);

  Vec3 camera_origin(0, 0, 0);
  float viewport_height = 2.0f;
  float viewport_width = (float)width / (float)height * viewport_height;
  float focal_length = 1.0f;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float u = (double(x) / (width - 1)) * viewport_width - viewport_width / 2;
      float v =
          (double(y) / (height - 1)) * viewport_height - viewport_height / 2;
      Vec3 direction = Vec3(u, v, -focal_length).normalize();
      Ray ray(camera_origin, direction);
      framebuffer[y * width + x] = trace(ray, spheres);
    }
  }

  savePPM(framebuffer.data(), width, height, outputFile);
}
