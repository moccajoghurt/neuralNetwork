#include <iostream>
#include "convolution.h"
#include "CimgHelper.h"

using namespace std;

void wideConvolutionCalculatesCorrectValuesTest() {
    // test with greyscale
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
    if (greyFm.getGreyscaleFeatureMap()[1][1].p == 89) {
        cout << "wideConvolutionCalculatesCorrectValuesTest() success" << endl;
    } else {
        cout << "wideConvolutionCalculatesCorrectValuesTest() failed" << endl;
    }
}

int main() {
    // FeatureMapImage colorFm = CimgHelper::importRGBImage("Cat.bmp");
    // FeatureMapImage greyFm = CimgHelper::importGreyscaleImage("Cat.bmp");

    // Edge0FilterMatrix sfm;
    // CimgHelper::displayFeatureMapImage(colorFm);
    // Convolution::wideConvolve(colorFm, sfm);
    // CimgHelper::displayFeatureMapImage(colorFm);

    // CimgHelper::displayFeatureMapImage(greyFm);
    // Convolution::wideConvolve(greyFm, sfm);
    // CimgHelper::displayFeatureMapImage(greyFm);

    wideConvolutionCalculatesCorrectValuesTest();
}