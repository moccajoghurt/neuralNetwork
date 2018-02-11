#include <string>
using namespace std;

typedef struct Size {
    int width;
    int height;
} size;

enum FilterType {
    SHARPEN_FILTER,
    EDGE_FILTER0,
    EDGE_FILTER1,
    EDGE_FILTER2,
    EMBOSS_FILTER,
    IDENTITY
};


class FilterMatrix {
public:
    int** applyFilter(FilterType type) {

    }
private:
    Size filterMatrixSize;
    int sharpenFilterMatrix[3][3] = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };

    int edgeFilterMatrix0[3][3] = {
        {1, 0, -1},
        {0, 0, 0},
        {-1, 0, 1}
    };

    int edgeFilterMatrix1[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };

    int edgeFilterMatrix2[3][3] = {
        {-1, -1, -1},
        {-1, 8, -1},
        {-1, -1, -1}
    };

    int embossFilterMatrix[3][3] = {
        {-2, -1, 0},
        {-1, 1, 1},
        {0, 1, 2}
    };

    int identityFilterMatrix[3][3] = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };
};

class FeatureMap {
public:
    Size getSize();
private:
    int** featureMap;
};

class Convolution {
public:
    // TODO: use CImg-Library for image import
    Convolution();

    FeatureMap convolute(Size targetSize = {0, 0}, int stepSize = 1);

private:
    string imgFolderPath;
    Size featureMapSize;
    
};