#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <cfloat> // For FLT_MAX
#include <cmath>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

struct Vec3 {
  float x, y, z;
  CUDA_HOSTDEV Vec3() : x(0), y(0), z(0) {}
  CUDA_HOSTDEV Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
  CUDA_HOSTDEV Vec3 operator+(const Vec3 &v) const {
    return Vec3(x + v.x, y + v.y, z + v.z);
  }
  CUDA_HOSTDEV Vec3 operator-(const Vec3 &v) const {
    return Vec3(x - v.x, y - v.y, z - v.z);
  }
  CUDA_HOSTDEV Vec3 operator*(float s) const {
    return Vec3(x * s, y * s, z * s);
  }
  CUDA_HOSTDEV Vec3 operator/(float s) const {
    return Vec3(x / s, y / s, z / s);
  }
  CUDA_HOSTDEV Vec3 normalize() const {
    float len = sqrtf(x * x + y * y + z * z);
    return Vec3(x / len, y / len, z / len);
  }
  CUDA_HOSTDEV float dot(const Vec3 &v) const {
    return x * v.x + y * v.y + z * v.z;
  }
};

struct Ray {
  Vec3 origin, direction;
  CUDA_HOSTDEV Ray() : origin(), direction() {} // Default constructor
  CUDA_HOSTDEV Ray(const Vec3 &o, const Vec3 &d) : origin(o), direction(d) {}
};

struct Sphere {
  Vec3 center;
  float radius;
  CUDA_HOSTDEV Sphere() : center(), radius(0) {} // Default constructor
  CUDA_HOSTDEV Sphere(const Vec3 &c, float r) : center(c), radius(r) {}
  CUDA_HOSTDEV bool intersect(const Ray &ray, float &t) const {
    Vec3 oc = ray.origin - center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0)
      return false;
    t = (-b - sqrtf(discriminant)) / (2.0f * a);
    return true;
  }
};

#endif // RAYTRACER_H
