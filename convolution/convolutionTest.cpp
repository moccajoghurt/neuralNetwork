#include <iostream>
#include "convolution.h"
#include "CimgHelper.h"

using namespace std;

int main() {
    FeatureMapImage colorFm = CimgHelper::importRGBImage("Cat.bmp");
    FeatureMapImage greyFm = CimgHelper::importGreyscaleImage("Cat.bmp");
    CimgHelper::displayFeatureMapImage(colorFm);
    CimgHelper::displayFeatureMapImage(greyFm);
}