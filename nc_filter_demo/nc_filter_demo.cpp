#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

struct Color {
    float r, g, b;
};

struct CubeFile {
    int size;
    std::vector<Color> lut;

    bool isEmpty() {
        return size == NULL;
    }

};

CubeFile parseCubeFile(const std::string& filename) {
    std::ifstream file(filename);
    CubeFile cubeFile;

    if (!file.is_open())
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return cubeFile;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("LUT_3D_SIZE") != std::string::npos) {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> cubeFile.size;
            continue;
        }

        Color color;
        std::istringstream iss(line);
        iss >> color.r >> color.g >> color.b;
        cubeFile.lut.push_back(color);
    }
    return cubeFile;
}

Color applyLut(CubeFile cubefile, const Color inputColor) {  // input color and output color is all [0-1]
    int size = cubefile.size;
    float x = inputColor.r * (size - 1);
    float y = inputColor.g * (size - 1);
    float z = inputColor.b * (size - 1);

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int z0 = static_cast<int>(z);

    int x1 = min(x0 + 1, (size-1));
    int y1 = min(y0 + 1, (size-1));
    int z1 = min(z0 + 1, (size-1));

    float dx = x - x0;
    float dy = y - y0;
    float dz = z - z0;

    Color c000 = cubefile.lut[x0 + size * (y0 + size * z0)];
    Color c001 = cubefile.lut[x0 + size * (y0 + size * z1)];
    Color c010 = cubefile.lut[x0 + size * (y1 + size * z0)];
    Color c011 = cubefile.lut[x0 + size * (y1 + size * z1)];
    Color c100 = cubefile.lut[x1 + size * (y0 + size * z0)];
    Color c101 = cubefile.lut[x1 + size * (y0 + size * z1)];
    Color c110 = cubefile.lut[x1 + size * (y1 + size * z0)];
    Color c111 = cubefile.lut[x1 + size * (y1 + size * z1)];

    float r = c000.r * (1 - dx) * (1 - dy) * (1 - dz) +
        c001.r * (1 - dx) * (1 - dy) * dz +
        c010.r * (1 - dx) * dy * (1 - dz) +
        c011.r * (1 - dx) * dy * dz +
        c100.r * dx * (1 - dy) * (1 - dz) +
        c101.r * dx * (1 - dy) * dz +
        c110.r * dx * dy * (1 - dz) +
        c111.r * dx * dy * dz;

    float g = c000.g * (1 - dx) * (1 - dy) * (1 - dz) +
        c001.g * (1 - dx) * (1 - dy) * dz +
        c010.g * (1 - dx) * dy * (1 - dz) +
        c011.g * (1 - dx) * dy * dz +
        c100.g * dx * (1 - dy) * (1 - dz) +
        c101.g * dx * (1 - dy) * dz +
        c110.g * dx * dy * (1 - dz) +
        c111.g * dx * dy * dz;

    float b = c000.b * (1 - dx) * (1 - dy) * (1 - dz) +
        c001.b * (1 - dx) * (1 - dy) * dz +
        c010.b * (1 - dx) * dy * (1 - dz) +
        c011.b * (1 - dx) * dy * dz +
        c100.b * dx * (1 - dy) * (1 - dz) +
        c101.b * dx * (1 - dy) * dz +
        c110.b * dx * dy * (1 - dz) +
        c111.b * dx * dy * dz;

    return { r, g, b };
}




int main()
{
    Mat inputImage = imread("DSC08335.tif");
    //Mat inputImage = imread("test.png");


    if (inputImage.empty()) {
        std::cerr << "Error: unable to read input image." << std::endl;
        return -1;
    }

    CubeFile cubefile = parseCubeFile("classic_neg.cube");

    if (cubefile.isEmpty()) {
        std::cerr << "Error: unable to parse cube file." << std::endl;
        return -1;
    }

    #pragma omp parallel for
    for (int x = 0; x < inputImage.rows; x++) {
        for (int y = 0; y < inputImage.cols; y++) {
            Vec3b pixel = inputImage.at<Vec3b>(x, y);
            Color inputColor = { pixel[2] / 255.f,
                                pixel[1] / 255.f,
                                pixel[0] / 255.f }; // in opencv ,3 channel is b g r
            Color outputColor = applyLut(cubefile, inputColor);
            pixel[2] = outputColor.r * 255;
            pixel[1] = outputColor.g * 255;
            pixel[0] = outputColor.b * 255;
            inputImage.at<Vec3b>(x, y) = pixel;
        }
    }

    cv::imwrite("output_image.tif", inputImage);
    // cv::namedWindow("test window");
    // cv::imshow("test window", outputImage);
    // cv::waitKey(0);
}

