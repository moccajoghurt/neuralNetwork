#ifndef _CIMGHELPER_H_
#define _CIMGHELPER_H_

#include <iostream>
#include "external/Cimg.h"
#include "convolution.h"

using namespace std;
using namespace cimg_library;

class CimgHelper {
public:
    static FeatureMapImage importRGBImage(string path) {
        CImg<unsigned char> image(path.c_str());
        vector<vector<RGBPixel> > imgBuf;
        for (int i = 0; i < image.width(); i++) {
            vector<RGBPixel> imgRow;
            for (int n = 0; n < image.height(); n++) {
                uchar_t r = (uchar_t)image(i, n, 0, 0);
                uchar_t g = (uchar_t)image(i, n, 0, 1);
                uchar_t b = (uchar_t)image(i, n, 0, 2);
                RGBPixel rgbPixel = {r, g, b};
                imgRow.push_back(rgbPixel);
            }
            imgBuf.push_back(imgRow);
        }
        FeatureMapImage featureMapImage(true, {imgBuf.size(), imgBuf[0].size()}, imgBuf);
        return featureMapImage;
    }

    static FeatureMapImage importGreyscaleImage(string path) {
        CImg<unsigned char> image(path.c_str());
        CImg<unsigned char> greyImage(image.channel(0));
        vector<vector<GreyPixel> > imgBuf;
        for (int i = 0; i < image.width(); i++) {
            vector<GreyPixel> imgRow;
            for (int n = 0; n < image.height(); n++) {
                uchar_t r = (uchar_t)image(i, n, 0, 0);
                GreyPixel greyPixel = {r};
                imgRow.push_back(greyPixel);
            }
            imgBuf.push_back(imgRow);
        }
        FeatureMapImage featureMapImage(false, {imgBuf.size(), imgBuf[0].size()}, imgBuf);
        return featureMapImage;
    }

    static void displayFeatureMapImage(FeatureMapImage featureMapImage) {
        CImg<unsigned char> imgBuf;
        int imgWidth = featureMapImage.getSize().width;
        int imgHeight = featureMapImage.getSize().height;
        if (featureMapImage.hasColors()) {
            imgBuf = CImg<unsigned char>(imgWidth, imgHeight, 1, 3, 0);
            for (int i = 0; i < imgWidth; i++) {
                for (int n = 0; n < imgHeight; n++) {
                    RGBPixel p = {
                        featureMapImage.getRGBFeatureMap()[i][n].r, 
                        featureMapImage.getRGBFeatureMap()[i][n].g,
                        featureMapImage.getRGBFeatureMap()[i][n].b
                    };
                    const unsigned char color[] = {p.r, p.g, p.b};
                    imgBuf.draw_point(i, n, color);
                }
            }

        } else {
            imgBuf = CImg<unsigned char>(imgWidth, imgHeight, 1, 1, 0);
            for (int i = 0; i < imgWidth; i++) {
                for (int n = 0; n < imgHeight; n++) {
                    GreyPixel p = {
                        featureMapImage.getGreyscaleFeatureMap()[i][n].p, 
                    };
                    const unsigned char color[] = {p.p};
                    imgBuf.draw_point(i, n, color);
                }
            }
        }
        imgBuf.display();
    }
};

#endif //_CIMGHELPER_H_