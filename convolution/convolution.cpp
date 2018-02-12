#include "convolution.h"

using namespace std;

FeatureMapImage& Convolution::wideConvolve(FeatureMapImage& featureMapImage, FilterMatrix& filterMatrix, int stepSize) {
    int featureMapWidth = featureMapImage.getSize().width;
    int featureMapHeight = featureMapImage.getSize().height;

    if (featureMapImage.hasColors()) {
        
        vector<vector<RGBPixel> > newRGBImg;
        for (int i = 0; i < featureMapWidth; i += stepSize) {
            vector<RGBPixel> RGBPixelCol;
            for (int n = 0; n < featureMapHeight; n += stepSize) {
                int newRPixel = 0;
                int newGPixel = 0;
                int newBPixel = 0;
                int filterMatrixWidth = filterMatrix.getSize().width;
                int filterMatrixHeight = filterMatrix.getSize().height;
                vector<vector<RGBPixel> >& RGBFeatureMap = featureMapImage.getRGBFeatureMap();
                vector<vector<int> >& filterMatrixVec = filterMatrix.getFilterMatrixVec();
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
        featureMapImage.setSize({newRGBImg[0].size(), newRGBImg.size()});
        featureMapImage.setRGBFeatureMap(newRGBImg);
    } else {
        vector<vector<GreyPixel> > newGreyScaleImg;
        for (int i = 0; i < featureMapWidth; i += stepSize) {
            vector<GreyPixel> greyPixelCol;
            for (int n = 0; n < featureMapHeight; n += stepSize) {
                uint_t newPixelValue = 0;
                int filterMatrixWidth = filterMatrix.getSize().width;
                int filterMatrixHeight = filterMatrix.getSize().height;
                vector<vector<GreyPixel> >& GreyscaleFeatureMap = featureMapImage.getGreyscaleFeatureMap();
                vector<vector<int> >& filterMatrixVec = filterMatrix.getFilterMatrixVec();
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
        featureMapImage.setSize({newGreyScaleImg[0].size(), newGreyScaleImg.size()});
        featureMapImage.setGreyscaleFeatureMap(newGreyScaleImg);
    }
}

FeatureMapImage& Convolution::maxPool(FeatureMapImage& featureMapImage, const Size windowSize) {
    int featureMapWidth = featureMapImage.getSize().width;
    int featureMapHeight = featureMapImage.getSize().height;

    if (featureMapImage.hasColors()) {
        
        for (int i = 0; i < featureMapWidth; i += windowSize.width) {
            vector<RGBPixel> RGBPixelCol;
            for (int n = 0; n < featureMapHeight; n += windowSize.height) {
                int newRPixel = 0;
                int newGPixel = 0;
                int newBPixel = 0;
                vector<vector<RGBPixel> >& RGBFeatureMap = featureMapImage.getRGBFeatureMap(); // for the old vals
                for (int x = 0; x < windowSize.width; x++) {
                    for (int y = 0; y < windowSize.height; y++) {
                        int xPosRelativeToFilter = i + (-windowSize.width/2 + x);
                        if (xPosRelativeToFilter < 0 || xPosRelativeToFilter >= featureMapWidth) {
                            continue;
                        }
                        int yPosRelativeToFilter = n + (-windowSize.height/2 + y);
                        if (yPosRelativeToFilter < 0 || yPosRelativeToFilter >= featureMapHeight) {
                            continue;
                        }
                        
                    }
                }
                RGBPixelCol.push_back({(uchar_t)newRPixel, (uchar_t)newGPixel, (uchar_t)newBPixel});
            }
            newRGBImg.push_back(RGBPixelCol);
        }
        featureMapImage.setSize({newRGBImg[0].size(), newRGBImg.size()});
        featureMapImage.setRGBFeatureMap(newRGBImg);
    } else {
        vector<vector<GreyPixel> > newGreyScaleImg;
        for (int i = 0; i < featureMapWidth; i += stepSize) {
            vector<GreyPixel> greyPixelCol;
            for (int n = 0; n < featureMapHeight; n += stepSize) {
                uint_t newPixelValue = 0;
                vector<vector<GreyPixel> >& GreyscaleFeatureMap = featureMapImage.getGreyscaleFeatureMap();
                for (int x = 0; x < filterMatrixWidth; x++) {
                    for (int y = 0; y < filterMatrixHeight; y++) {
                        int xPosRelativeToFilter = i + (-windowSize.width/2 + x);
                        if (xPosRelativeToFilter < 0 || xPosRelativeToFilter >= featureMapWidth) {
                            continue;
                        }
                        int yPosRelativeToFilter = n + (-windowSize.height/2 + y);
                        if (yPosRelativeToFilter < 0 || yPosRelativeToFilter >= featureMapHeight) {
                            continue;
                        }
                        newPixelValue += 0;
                    }
                }
                greyPixelCol.push_back({(uchar_t)newPixelValue});
            }
            newGreyScaleImg.push_back(greyPixelCol);
        }
        featureMapImage.setSize({newGreyScaleImg[0].size(), newGreyScaleImg.size()});
        featureMapImage.setGreyscaleFeatureMap(newGreyScaleImg);
    }
}
