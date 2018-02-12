#include <iostream>
#include "convolution.h"
#include "CimgHelper.h" // for debugging, remove later

using namespace std;

void wideConvolutionCalculatesCorrectValuesTest() {
    // test with greyscale, colored implementation is almost identical but way more time-consuming to test
    vector<vector<GreyPixel> > gfm;
    vector<GreyPixel> row1;
    row1.push_back({105});
    row1.push_back({102});
    row1.push_back({100});
    vector<GreyPixel> row2;
    row2.push_back({103});
    row2.push_back({99});
    row2.push_back({103});
    vector<GreyPixel> row3;
    row3.push_back({101});
    row3.push_back({98});
    row3.push_back({104});
    gfm.push_back(row1);
    gfm.push_back(row2);
    gfm.push_back(row3);
    FeatureMapImage greyFm(false, {3, 3}, gfm);
    SharpenFilterMatrix sfm;
    Convolution::wideConvolve(greyFm, sfm);
    if (
        greyFm.getGreyscaleFeatureMap()[0][0].p == 255 &&
        greyFm.getGreyscaleFeatureMap()[0][1].p == 206 &&
        greyFm.getGreyscaleFeatureMap()[0][2].p == 255 &&
        greyFm.getGreyscaleFeatureMap()[1][0].p == 210 &&
        greyFm.getGreyscaleFeatureMap()[1][1].p == 89 &&
        greyFm.getGreyscaleFeatureMap()[1][2].p == 212 &&
        greyFm.getGreyscaleFeatureMap()[2][0].p == 255 &&
        greyFm.getGreyscaleFeatureMap()[2][1].p == 186 &&
        greyFm.getGreyscaleFeatureMap()[2][2].p == 255
    ) {
        cout << "wideConvolutionCalculatesCorrectValuesTest() success" << endl;
    } else {
        cout << "[0][0]: " << (uint_t)greyFm.getGreyscaleFeatureMap()[0][0].p << endl;
        cout << "[0][1]: " << (uint_t)greyFm.getGreyscaleFeatureMap()[0][1].p << endl;
        cout << "[0][2]: " << (uint_t)greyFm.getGreyscaleFeatureMap()[0][2].p << endl;
        cout << "[1][0]: " << (uint_t)greyFm.getGreyscaleFeatureMap()[1][0].p << endl;
        cout << "[1][1]: " << (uint_t)greyFm.getGreyscaleFeatureMap()[1][1].p << endl;
        cout << "[1][2]: " << (uint_t)greyFm.getGreyscaleFeatureMap()[1][2].p << endl;
        cout << "[2][0]: " << (uint_t)greyFm.getGreyscaleFeatureMap()[2][0].p << endl;
        cout << "[2][1]: " << (uint_t)greyFm.getGreyscaleFeatureMap()[2][1].p << endl;
        cout << "[2][2]: " << (uint_t)greyFm.getGreyscaleFeatureMap()[2][2].p << endl;
        cout << "wideConvolutionCalculatesCorrectValuesTest() failed" << endl;
    }
}

void poolingCreatesCorrectFeatureMapSize() {
    vector<vector<GreyPixel> > gfm;
    for (int i = 0; i < 10; i++) {
        vector<GreyPixel> gCol;
        for (int n = 0; n < 16; n++) {
            gCol.push_back({0});
        }
        gfm.push_back(gCol);
    }
    FeatureMapImage greyFm(false, {10, 16}, gfm);
    FeatureMapImage greyFm1(false, {10, 16}, gfm);
    Convolution::maxPool(greyFm, {2, 2});
    Convolution::maxPool(greyFm1, {2, 4});
    if (
        greyFm.getSize().width == 5 && greyFm.getSize().height == 8 &&
        greyFm1.getSize().width == 5 && greyFm1.getSize().height == 4
    ) {
        cout << "poolingCreatesCorrectFeatureMapSize() success" << endl;
    } else {
        cout << "poolingCreatesCorrectFeatureMapSize() failed" << endl;
    }
}

void maxPoolingCreatesCorrectFeatureMapData() {
    vector<vector<GreyPixel> > gfm;
    vector<GreyPixel> row1;
    row1.push_back({105});
    row1.push_back({102});
    row1.push_back({100});
    row1.push_back({99});
    vector<GreyPixel> row2;
    row2.push_back({103});
    row2.push_back({92});
    row2.push_back({107});
    row2.push_back({99});
    vector<GreyPixel> row3;
    row3.push_back({101});
    row3.push_back({98});
    row3.push_back({104});
    row3.push_back({99});
    vector<GreyPixel> row4;
    row4.push_back({104});
    row4.push_back({120});
    row4.push_back({106});
    row4.push_back({99});
    gfm.push_back(row1);
    gfm.push_back(row2);
    gfm.push_back(row3);
    gfm.push_back(row4);
    FeatureMapImage greyfm(false, {4, 4}, gfm);
    Convolution::maxPool(greyfm, {2, 2});
    if (
        greyfm.getGreyscaleFeatureMap()[0][0].p == 105 &&
        greyfm.getGreyscaleFeatureMap()[0][1].p == 107 &&
        greyfm.getGreyscaleFeatureMap()[1][0].p == 120 &&
        greyfm.getGreyscaleFeatureMap()[1][1].p == 106
    ) {
        cout << "maxPoolingCreatesCorrectFeatureMapData() success" << endl;
    } else {
        cout << "maxPoolingCreatesCorrectFeatureMapData() failed" << endl;
    }
}

void playField() {
    FeatureMapImage colorFm = CimgHelper::importRGBImage("Dog.bmp");
    FeatureMapImage greyFm = CimgHelper::importGreyscaleImage("Dog.bmp");

    Edge2FilterMatrix sfm;
    CimgHelper::displayFeatureMapImage(greyFm);
    for (int i = 0; i < 2; i++) {
        Convolution::wideConvolve(greyFm, sfm);
        Convolution::maxPool(greyFm, {3, 3});
        CimgHelper::displayFeatureMapImage(greyFm);
    }
}

int main() {
    wideConvolutionCalculatesCorrectValuesTest();
    poolingCreatesCorrectFeatureMapSize();
    maxPoolingCreatesCorrectFeatureMapData();
    
    playField();
}