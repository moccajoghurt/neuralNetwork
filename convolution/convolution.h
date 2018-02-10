#include <string>
using namespace std;

typedef struct Size {
    int width;
    int height;
} size;

typedef struct FilterMatrixTypes {

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

} FilterMatrixTypes;

class FilterMatrix {

private:
    Size filterMatrixSize;
};

class Convolution {
public:
    Convolution();

private:
    string imgFolderPath;
    Size featureMapSize;
    
};