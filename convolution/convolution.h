#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include <string>
#include <vector>
#include <iostream>

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
    FeatureMapImage(bool hasColors, Size size, vector<vector<RGBPixel> > RGBFeatureMap) {
        this->coloredImage = hasColors;
        this->size = size;
        this->RGBFeatureMap = RGBFeatureMap;
    }
    FeatureMapImage(bool hasColors, Size size, vector<vector<GreyPixel> > GreyscaleFeatureMap) {
        this->coloredImage = hasColors;
        this->size = size;
        this->GreyscaleFeatureMap = GreyscaleFeatureMap;
    }
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
    virtual Size& getSize() {
        return filterMatrixSize;
    }
    virtual vector<vector<int> >& getFilterMatrixVec() {
        return filterMatrixVec;
    }
protected:
    Size filterMatrixSize;
    vector<vector<int> > filterMatrixVec;
};

class SharpenFilterMatrix : public FilterMatrix {
public:
    SharpenFilterMatrix() {
        filterMatrixSize = {3, 3};
        for (int i = 0; i < filterMatrixSize.height; i++) {
            vector<int> filterMatrixRow;
            filterMatrixRow.assign(filterMatrix[i], filterMatrix[i] + filterMatrixSize.width);
            filterMatrixVec.push_back(filterMatrixRow);
        }
    }
    const int filterMatrix[3][3] = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };
};

class Edge0FilterMatrix : public FilterMatrix {
public:
    Edge0FilterMatrix() {
        filterMatrixSize = {3, 3};
        for (int i = 0; i < filterMatrixSize.height; i++) {
            vector<int> filterMatrixRow;
            filterMatrixRow.assign(filterMatrix[i], filterMatrix[i] + filterMatrixSize.width);
            filterMatrixVec.push_back(filterMatrixRow);
        }
    }
    const int filterMatrix[3][3] = {
        {1, 0, -1},
        {0, 0, 0},
        {-1, 0, 1}
    };
};

class Edge1FilterMatrix : public FilterMatrix {
public:
    Edge1FilterMatrix() {
        filterMatrixSize = {3, 3};
        for (int i = 0; i < filterMatrixSize.height; i++) {
            vector<int> filterMatrixRow;
            filterMatrixRow.assign(filterMatrix[i], filterMatrix[i] + filterMatrixSize.width);
            filterMatrixVec.push_back(filterMatrixRow);
        }
    }
    const int filterMatrix[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };
};

class Edge2FilterMatrix : public FilterMatrix {
public:
    Edge2FilterMatrix() {
        filterMatrixSize = {3, 3};
        for (int i = 0; i < filterMatrixSize.height; i++) {
            vector<int> filterMatrixRow;
            filterMatrixRow.assign(filterMatrix[i], filterMatrix[i] + filterMatrixSize.width);
            filterMatrixVec.push_back(filterMatrixRow);
        }
    }
    const int filterMatrix[3][3] = {
        {-1, -1, -1},
        {-1, 8, -1},
        {-1, -1, -1}
    };
};

class EmbossFilterMatrix : public FilterMatrix {
public:
    EmbossFilterMatrix() {
        filterMatrixSize = {3, 3};
        for (int i = 0; i < filterMatrixSize.height; i++) {
            vector<int> filterMatrixRow;
            filterMatrixRow.assign(filterMatrix[i], filterMatrix[i] + filterMatrixSize.width);
            filterMatrixVec.push_back(filterMatrixRow);
        }
    }
    const int filterMatrix[3][3] = {
        {-2, -1, 0},
        {-1, 1, 1},
        {0, 1, 2}
    };
};

class IdentityFilterMatrix : public FilterMatrix {
public:
    IdentityFilterMatrix() {
        filterMatrixSize = {3, 3};
        for (int i = 0; i < filterMatrixSize.height; i++) {
            vector<int> filterMatrixRow;
            filterMatrixRow.assign(filterMatrix[i], filterMatrix[i] + filterMatrixSize.width);
            filterMatrixVec.push_back(filterMatrixRow);
        }
    }
    const int filterMatrix[3][3] = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };
};

class Convolution {
public:
    Convolution();

    static FeatureMapImage& wideConvolve(FeatureMapImage& f, FilterMatrix& filt, int stepSize = 1); // with zero-padding
    static FeatureMapImage& maxPool(FeatureMapImage& f, const Size windowSize);
};

#endif // _CONVOLUTION_H_