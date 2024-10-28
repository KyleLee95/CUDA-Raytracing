#include "utils.h"
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <image1.ppm> <image2.ppm> <scene-file>" << std::endl;
    return 1;
  }

  std::string imageFile1 = argv[1];
  std::string imageFile2 = argv[2];
  std::string sceneFile = argv[3];

  std::vector<Vec3> image1, image2;
  int width1, height1, width2, height2;

  if (!readPPM(imageFile1, image1, width1, height1) ||
      !readPPM(imageFile2, image2, width2, height2)) {
    return 1;
  }

  if (width1 != width2 || height1 != height2) {
    std::cerr << "Error: Image dimensions do not match." << std::endl;
    return 1;
  }

  // Read the number of spheres from the scene file
  std::ifstream sceneFileStream(sceneFile);
  if (!sceneFileStream.is_open()) {
    std::cerr << "Error: Could not open scene file." << std::endl;
    return 1;
  }

  int sphereCount;
  sceneFileStream >> sphereCount;
  sceneFileStream.close();

  float tolerance = 1e-2f;
  std::string outputCsvFile = "image_comparison_results.csv";

  bool imagesMatch = compareImages(image1, image2, width1, height1, tolerance,
                                   outputCsvFile, sphereCount);
  if (imagesMatch) {
    std::cout << "Images match." << std::endl;
  } else {
    std::cout << "Images do not match." << std::endl;
  }

  return 0;
}
