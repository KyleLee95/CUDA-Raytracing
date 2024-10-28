#include "raytracer.h"
#include "utils.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
struct Arguments {
  std::string sceneFile;
  int width;
  int height;
  int sphereCount;
};

bool parseArguments(int argc, char **argv, Arguments &args) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [-d width height] <sceneFile>"
              << std::endl;
    return false;
  }

  args.width = 800;     // Default width
  args.height = 600;    // Default height
  args.sphereCount = 0; // Default sphere count

  int index = 1;

  // Check for the "-d" flag
  if (strcmp(argv[1], "-d") == 0) {
    if (argc < 5) {
      std::cerr << "Usage: " << argv[0] << " [-d width height] <sceneFile>"
                << std::endl;
      return false;
    }
    args.width = std::stoi(argv[2]);
    args.height = std::stoi(argv[3]);
    index = 4;
  }

  // The last argument should be the scene file
  if (index >= argc) {
    std::cerr << "Error: Scene file not provided." << std::endl;
    return false;
  }

  args.sceneFile = argv[index];

  // Read the sphere count from the scene file
  std::ifstream sceneFile(args.sceneFile);
  if (!sceneFile.is_open()) {
    std::cerr << "Error: Could not open scene file " << args.sceneFile
              << std::endl;
    return false;
  }

  sceneFile >> args.sphereCount;
  if (sceneFile.fail()) {
    std::cerr << "Error: Could not read sphere count from scene file "
              << args.sceneFile << std::endl;
    return false;
  }

  return true;
}

std::string generateOutputFilename(const std::string &sceneFile,
                                   const std::string &prefix) {
  size_t startPos = sceneFile.find_last_of('/');
  if (startPos == std::string::npos) {
    startPos = 0;
  } else {
    startPos++;
  }
  size_t endPos = sceneFile.find_last_of('.');
  std::string baseName = sceneFile.substr(startPos, endPos - startPos);
  return prefix + baseName + ".ppm";
}

void render(int width, int height, const std::string &sceneFile,
            const std::string &outputFile);

int main(int argc, char **argv) {
  Arguments args;
  if (!parseArguments(argc, argv, args)) {
    return 1;
  }
  for (int i = 0; i < 10; i++) {

    std::string outputFile =
        generateOutputFilename(args.sceneFile, "output_sequential_");
    auto start = std::chrono::high_resolution_clock::now();

    render(args.width, args.height, args.sceneFile, outputFile);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "CPU Rendering Time: " << diff.count() << " s\n";

    writeTrialToCsv(diff.count(), args.width, args.height, args.sceneFile,
                    "sequential", args.sphereCount);
  }
  return 0;
}
