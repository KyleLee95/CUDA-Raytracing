#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
struct Vec3 {
  float x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

struct Sphere {
  Vec3 center;
  float radius;
  Sphere(const Vec3 &c, float r) : center(c), radius(r) {}
};

void saveScene(const std::vector<Sphere> &spheres,
               const std::string &filename) {
  std::ofstream file(filename);
  file << spheres.size() << "\n";
  for (const auto &sphere : spheres) {
    file << sphere.center.x << " " << sphere.center.y << " " << sphere.center.z
         << " " << sphere.radius << "\n";
  }
  file.close();
}

void generateRandomScenes(int numScenes, int maxSpheres,
                          const std::string &filePrefix) {
  std::srand(std::time(0));
  for (int i = 1; i <= numScenes; ++i) {
    int numSpheres = std::rand() % maxSpheres + 1;
    std::vector<Sphere> spheres;
    for (int j = 0; j < numSpheres; ++j) {
      float x = (std::rand() % 2000 - 1000) / 100.0f;
      float y = (std::rand() % 2000 - 1000) / 100.0f;
      float z = (std::rand() % 2000 - 1000) / 100.0f;
      float radius = (std::rand() % 100 + 1) / 100.0f;
      spheres.emplace_back(Vec3(x, y, z), radius);
    }
    std::string filename = filePrefix + std::to_string(i) + ".txt";
    saveScene(spheres, filename);
    std::cout << "Generated " << filename << " with " << numSpheres
              << " spheres.\n";
  }
}

int main() {
  int numScenes = 10;
  int maxSpheres = 10;
  std::string filePrefix = "scene_";
  generateRandomScenes(numScenes, maxSpheres, filePrefix);
  return 0;
}
