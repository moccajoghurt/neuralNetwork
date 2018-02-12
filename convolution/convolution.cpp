#include "convolution.h"

using namespace std;

FeatureMapImage& Convolution::wideConvolve(FeatureMapImage& featureMapImage, FilterMatrix& filterMatrix, int stepSize) {
    int featureMapWidth = featureMapImage.getSize().width;
    int featureMapHeight = featureMapImage.getSize().height;

    if (featureMapImage.hasColors()) {
        
        vector<vector<RGBPixel> > newRGBImg;
        vector<vector<RGBPixel> >& RGBFeatureMap = featureMapImage.getRGBFeatureMap();
        vector<vector<int> >& filterMatrixVec = filterMatrix.getFilterMatrixVec();
        for (int i = 0; i < featureMapWidth; i += stepSize) {
            vector<RGBPixel> RGBPixelCol;
            for (int n = 0; n < featureMapHeight; n += stepSize) {
                int newRPixel = 0;
                int newGPixel = 0;
                int newBPixel = 0;
                int filterMatrixWidth = filterMatrix.getSize().width;
                int filterMatrixHeight = filterMatrix.getSize().height;
                for (int x = 0; x < filterMatrixWidth; x++) {
                    for (int y = 0; y < filterMatrixHeight; y++) {
                        int xPosRelativeToFilter = i + (-filterMatrixWidth/2 + x);
                        if (xPosRelativeToFilter < 0 || xPosRelativeToFilter >= featureMapWidth) {
                            continue;
                        }
                        int yPosRelativeToFilter = n + (-filterMatrixHeight/2 + y);
                        if (yPosRelativeToFilter < 0 || yPosRelativeToFilter >= featureMapHeight) {
                            continue;
                        }
                        newRPixel += RGBFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].r * filterMatrixVec[x][y];
                        newGPixel += RGBFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].g * filterMatrixVec[x][y];
                        newBPixel += RGBFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].b * filterMatrixVec[x][y];
                    }
                }
                newRPixel = newRPixel > 255 ? 255 : newRPixel;
                newGPixel = newGPixel > 255 ? 255 : newGPixel;
                newBPixel = newBPixel > 255 ? 255 : newBPixel;
                newRPixel = newRPixel < 0 ? 0 : newRPixel;
                newGPixel = newGPixel < 0 ? 0 : newGPixel;
                newBPixel = newBPixel < 0 ? 0 : newBPixel;
                RGBPixelCol.push_back({(uchar_t)newRPixel, (uchar_t)newGPixel, (uchar_t)newBPixel});
            }
            newRGBImg.push_back(RGBPixelCol);
        }
        featureMapImage.setSize({newRGBImg.size(), newRGBImg[0].size()});
        featureMapImage.setRGBFeatureMap(newRGBImg);
    } else {
        vector<vector<GreyPixel> > newGreyScaleImg;
        vector<vector<GreyPixel> >& GreyscaleFeatureMap = featureMapImage.getGreyscaleFeatureMap();
        vector<vector<int> >& filterMatrixVec = filterMatrix.getFilterMatrixVec();
        for (int i = 0; i < featureMapWidth; i += stepSize) {
            vector<GreyPixel> greyPixelCol;
            for (int n = 0; n < featureMapHeight; n += stepSize) {
                uint_t newPixelValue = 0;
                int filterMatrixWidth = filterMatrix.getSize().width;
                int filterMatrixHeight = filterMatrix.getSize().height;
                for (int x = 0; x < filterMatrixWidth; x++) {
                    for (int y = 0; y < filterMatrixHeight; y++) {
                        int xPosRelativeToFilter = i + (-filterMatrixWidth/2 + x);
                        if (xPosRelativeToFilter < 0 || xPosRelativeToFilter >= featureMapWidth) {
                            continue;
                        }
                        int yPosRelativeToFilter = n + (-filterMatrixHeight/2 + y);
                        if (yPosRelativeToFilter < 0 || yPosRelativeToFilter >= featureMapHeight) {
                            continue;
                        }
                        newPixelValue += GreyscaleFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].p * filterMatrixVec[x][y];
                    }
                }
                newPixelValue = newPixelValue > 255 ? 255 : newPixelValue;
                newPixelValue = newPixelValue < 0 ? 0 : newPixelValue;
                greyPixelCol.push_back({(uchar_t)newPixelValue});
            }
            newGreyScaleImg.push_back(greyPixelCol);
        }
        featureMapImage.setSize({newGreyScaleImg.size(), newGreyScaleImg[0].size()});
        featureMapImage.setGreyscaleFeatureMap(newGreyScaleImg);
    }
}

FeatureMapImage& Convolution::maxPool(FeatureMapImage& featureMapImage, const Size windowSize) {
    int featureMapWidth = featureMapImage.getSize().width;
    int featureMapHeight = featureMapImage.getSize().height;

    if (featureMapWidth % windowSize.width != 0 || featureMapHeight % windowSize.height != 0) {
        cout << "maxPool(): warning! dimensions of featureMap and slidingWindow are not divisible. Data loss at corners will occur!" << endl;
    }

    if (featureMapImage.hasColors()) {
        vector<vector<RGBPixel> > newRGBImg;
        vector<vector<RGBPixel> >& RGBFeatureMap = featureMapImage.getRGBFeatureMap(); // for the old vals
        for (int i = 0; i < featureMapWidth; i += windowSize.width) {
            vector<RGBPixel> RGBPixelCol;
            for (int n = 0; n < featureMapHeight; n += windowSize.height) {
                uchar_t newRPixel = 0;
                uchar_t newGPixel = 0;
                uchar_t newBPixel = 0;
                for (int x = 0; x < windowSize.width; x++) {
                    for (int y = 0; y < windowSize.height; y++) {
                        int xPosRelativeToFilter = i + x;
                        if (xPosRelativeToFilter < 0 || xPosRelativeToFilter >= featureMapWidth) {
                            continue;
                        }
                        int yPosRelativeToFilter = n + y;
                        if (yPosRelativeToFilter < 0 || yPosRelativeToFilter >= featureMapHeight) {
                            continue;
                        }
                        if (RGBFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].r > newRPixel) {
                            newRPixel = RGBFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].r;
                        }
                        if (RGBFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].g > newGPixel) {
                            newGPixel = RGBFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].g;
                        }
                        if (RGBFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].b > newBPixel) {
                            newBPixel = RGBFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].b;
                        }
                    }
                }
                RGBPixelCol.push_back({newRPixel, newGPixel, newBPixel});
            }
            newRGBImg.push_back(RGBPixelCol);
        }
        featureMapImage.setSize({newRGBImg.size(), newRGBImg[0].size()});
        featureMapImage.setRGBFeatureMap(newRGBImg);
    } else {
        vector<vector<GreyPixel> > newGreyScaleImg;
        vector<vector<GreyPixel> >& GreyscaleFeatureMap = featureMapImage.getGreyscaleFeatureMap();
        for (int i = 0; i < featureMapWidth; i += windowSize.width) {
            vector<GreyPixel> greyPixelCol;
            for (int n = 0; n < featureMapHeight; n += windowSize.height) {
                uchar_t newPixelValue = 0;
                for (int x = 0; x < windowSize.width; x++) {
                    for (int y = 0; y < windowSize.height; y++) {
                        int xPosRelativeToFilter = i + x;
                        if (xPosRelativeToFilter < 0 || xPosRelativeToFilter >= featureMapWidth) {
                            continue;
                        }
                        int yPosRelativeToFilter = n + y;
                        if (yPosRelativeToFilter < 0 || yPosRelativeToFilter >= featureMapHeight) {
                            continue;
                        }
                        if (GreyscaleFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].p > newPixelValue) {
                            newPixelValue = GreyscaleFeatureMap[xPosRelativeToFilter][yPosRelativeToFilter].p;
                        }
                    }
                }
                greyPixelCol.push_back({newPixelValue});
            }
            newGreyScaleImg.push_back(greyPixelCol);
        }
        featureMapImage.setSize({newGreyScaleImg.size(), newGreyScaleImg[0].size()});
        featureMapImage.setGreyscaleFeatureMap(newGreyScaleImg);
    }
}
