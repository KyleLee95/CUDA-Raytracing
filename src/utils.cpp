#include "utils.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void loadScene(std::vector<Sphere> &spheres, const std::string &filename) {
  std::ifstream file(filename);
  size_t numSpheres;
  file >> numSpheres;
  spheres.resize(numSpheres);
  for (auto &sphere : spheres) {
    file >> sphere.center.x >> sphere.center.y >> sphere.center.z >>
        sphere.radius;
  }
  file.close();
}

bool savePPM(const Vec3 *framebuffer, int width, int height,
             const std::string &filename) {
  // Ensure the output directory exists
  std::filesystem::create_directories("output");

  // Create the full path to the file in the output directory
  std::string filepath = "output/" + filename;

  std::ofstream file(filepath);
  if (!file) {
    std::cerr << "Error: Could not open file " << filepath << " for writing."
              << std::endl;
    return false;
  }

  file << "P3\n" << width << " " << height << "\n255\n";
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      int pixel_index = j * width + i;
      int r = static_cast<int>(255.999 * framebuffer[pixel_index].x);
      int g = static_cast<int>(255.999 * framebuffer[pixel_index].y);
      int b = static_cast<int>(255.999 * framebuffer[pixel_index].z);
      file << r << ' ' << g << ' ' << b << '\n';
    }
  }
  file.close();
  return true;
}

bool readPPM(const std::string &filename, std::vector<Vec3> &framebuffer,
             int &width, int &height) {
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    return false;
  }

  std::string format;
  file >> format;
  if (format != "P3") {
    std::cerr << "Error: Unsupported PPM format in file " << filename
              << std::endl;
    return false;
  }

  file >> width >> height;
  int max_val;
  file >> max_val;
  framebuffer.resize(width * height);

  for (int i = 0; i < width * height; ++i) {
    int r, g, b;
    file >> r >> g >> b;
    framebuffer[i] = Vec3(r / 255.0f, g / 255.0f, b / 255.0f);
  }

  file.close();
  return true;
}

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

bool compareImages(const std::vector<Vec3> &image1,
                   const std::vector<Vec3> &image2, int width, int height,
                   float tolerance, const std::string &outputCsvFile,
                   int sphereCount) {
  if (image1.size() != image2.size()) {
    std::cerr << "Error: Image sizes do not match." << std::endl;
    return false;
  }

  int mismatchCount = 0;
  bool imagesMatch = true;

  for (size_t i = 0; i < image1.size(); ++i) {
    if (fabs(image1[i].x - image2[i].x) > tolerance ||
        fabs(image1[i].y - image2[i].y) > tolerance ||
        fabs(image1[i].z - image2[i].z) > tolerance) {
      std::cerr << "Mismatch at pixel " << i << ": " << "image1(" << image1[i].x
                << ", " << image1[i].y << ", " << image1[i].z << ") vs "
                << "image2(" << image2[i].x << ", " << image2[i].y << ", "
                << image2[i].z << ")" << std::endl;
      mismatchCount++;
      imagesMatch = false;
    }
  }

  std::filesystem::create_directories("output");

  std::string filepath = "output/" + outputCsvFile;

  std::ofstream csvFile(filepath, std::ios::app);
  if (!csvFile.is_open()) {
    std::cerr << "Error: Could not open CSV file " << filepath
              << " for writing." << std::endl;
    return false;
  }

  csvFile << width << "x" << height << "," << sphereCount << ","
          << mismatchCount << "\n";
  csvFile.close();

  if (!imagesMatch) {
    std::cerr << "There were " << mismatchCount << " mismatched pixels."
              << std::endl;
  } else {
    std::cout << "Images match perfectly!" << std::endl;
  }

  return imagesMatch;
}

void writeTrialToCsv(double trial_time, int width, int height,
                     const std::string &sceneFile, const std::string &type,
                     int numSpheres) {
  std::filesystem::create_directories("output");

  std::string filepath = "output/raytracer_results.csv";

  std::ofstream file(filepath, std::ios::app);
  if (!file) {
    std::cerr << "Error: Could not open file " << filepath << " for writing."
              << std::endl;
    return;
  }

  std::string image_size = std::to_string(width) + "x" + std::to_string(height);
  std::string scene_name = sceneFile.substr(sceneFile.find_last_of("/\\") + 1);
  scene_name = scene_name.substr(0, scene_name.find_last_of('.'));

  file << trial_time << "," << image_size << "," << scene_name << ","
       << numSpheres << "," << type << "\n";

  file.close();
}
