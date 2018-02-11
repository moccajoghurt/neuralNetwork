#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include <string>
#include <vector>

using namespace std;

typedef unsigned char uchar_t;
typedef unsigned int uint_t;

typedef struct Size {
    uint_t width;
    uint_t height;
} Size;

typedef struct RGBPixel {
    uchar_t r;
    uchar_t g;
    uchar_t b;
} RGBPixel;

typedef struct GreyPixel {
    uchar_t p;
} GreyPixel;


class FeatureMapImage {
public:
    Size& getSize() {
        return size;
    }
    bool hasColors() {
        return coloredImage;
    }
    void setRGBFeatureMap(vector<vector<RGBPixel> > RGBFm) {
        RGBFeatureMap = RGBFm;
    }
    void setGreyscaleFeatureMap(vector<vector<GreyPixel> > GreyscaleFm) {
        GreyscaleFeatureMap = GreyscaleFm;
    }
    void setHasColors(bool b) {
        coloredImage = b;
    }
    void setSize(Size s) {
        size = s;
    }
    vector<vector<RGBPixel> >& getRGBFeatureMap() {
        return RGBFeatureMap;
    }
    vector<vector<GreyPixel> >& getGreyscaleFeatureMap() {
        return GreyscaleFeatureMap;
    }
private:
    vector<vector<RGBPixel> > RGBFeatureMap;
    vector<vector<GreyPixel> > GreyscaleFeatureMap;
    bool coloredImage;
    Size size;
};


class FilterMatrix {
public:
    FeatureMapImage& applyFilter(FeatureMapImage&, FilterMatrix);
private:
    Size filterMatrixSize;
    int filterMatrix[3][3];
};

class SharpenFilterMatrix : FilterMatrix {
    int filterMatrix[3][3] = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };
};

class Edge0FilterMatrix : FilterMatrix {
    int filterMatrix[3][3] = {
        {1, 0, -1},
        {0, 0, 0},
        {-1, 0, 1}
    };
};

class Edge1FilterMatrix : FilterMatrix {
    int filterMatrix[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };
};

class Edge2FilterMatrix : FilterMatrix {
    int filterMatrix[3][3] = {
        {-1, -1, -1},
        {-1, 8, -1},
        {-1, -1, -1}
    };
};

class EmbossFilterMatrix : FilterMatrix {
    int filterMatrix[3][3] = {
        {-2, -1, 0},
        {-1, 1, 1},
        {0, 1, 2}
    };
};

class IdentityFilterMatrix : FilterMatrix {
    int identityFilterMatrix[3][3] = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };
};

class Convolution {
public:
    Convolution();

    FeatureMapImage& convolute(Size targetSize = {0, 0}, int stepSize = 1);

private:
    string imgFolderPath;
    Size featureMapSize;
    
};

#endif // _CONVOLUTION_H_