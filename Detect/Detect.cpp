#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree.hpp>

#define SURF_HESSIAN_THRESHOLD 400

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Usage: Detect image_file" << endl;
        return -1;
    }

    string img_file(argv[1]);

    cout << "Reading vocabulary from file..." << endl;
    FileStorage fs;
    if (!fs.open("vocabulary.yml", FileStorage::READ)) {
        cout << "Can't read vocabulary.yml!" << endl;
        return -1;
    }
    Mat vocabulary;
    fs["vocabulary"] >> vocabulary;
    fs.release();
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    matcher->add(vector<Mat>(1, vocabulary));
    int cluster_count = vocabulary.rows;

    cout << "Reading trained neural network..." << endl;
    CvANN_MLP nn;
    nn.load("nn.yml");

    cout << "Reading image..." << endl;
    Mat img;
    img = imread(img_file, 0);
    if (img.empty()) {
        cout << "Can't read " << img_file << endl;
        return -1;
    }

    cout << "Computing image descriptors..." << endl;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Ptr<FeatureDetector> detector(new SurfFeatureDetector(SURF_HESSIAN_THRESHOLD));
    Ptr<DescriptorExtractor> extractor(new SurfDescriptorExtractor);
    detector->detect(img, keypoints);
    extractor->compute(img, keypoints, descriptors);

    Mat keypoints_map(img.size(), CV_16UC1, Scalar::all(0));
    for (int i = 0; i < keypoints.size(); i++)
        keypoints_map.at<ushort>((int)keypoints[i].pt.y, (int)keypoints[i].pt.x) = i;

    cout << "Scanning image with sliding window..." << endl;
    vector<Rect> wins;
    Mat img_rgb(img.size(), CV_8UC3);
    cvtColor(img, img_rgb, cv::COLOR_GRAY2BGR);

    //for (int l = 80; l < img.cols && l < img.rows; l+=3) {
    int l = 120; {
        Rect win(0, 0, l, l);
        for (win.y = 0; win.y <= img.rows - win.height; win.y+=4) {
            for (win.x = 0; win.x <= img.cols - win.width; win.x+=4) {
                Mat subdescriptors;
                SparseMat subkeypoints_map(keypoints_map(win));
                for (SparseMatConstIterator it = subkeypoints_map.begin(); it != subkeypoints_map.end(); it++) {
                    ushort index = it.value<ushort>();
                    KeyPoint *kp = &keypoints[index];
                    if (kp->pt.x - kp->size > win.x && kp->pt.x + kp->size < win.x + win.width
                     && kp->pt.y - kp->size > win.y && kp->pt.y + kp->size < win.y + win.height)
                        subdescriptors.push_back(descriptors.row(index));
                }

                vector<DMatch> matches;
                matcher->match(subdescriptors, matches);

                Mat bowdescriptor(1, cluster_count, CV_32FC1, Scalar::all(0.0));
                float *dptr = (float*)bowdescriptor.data;
                for (int i = 0; i < matches.size(); i++)
                    dptr[matches[i].trainIdx] += 1.f;
                bowdescriptor /= subdescriptors.rows;

                Mat resp;
                nn.predict(bowdescriptor, resp);
                if (resp.at<float>(0, 0) > 0) {
                    wins.push_back(win);
                }
            }
        }
    }

    cout << "Showing detection results..." << endl;
    for (Rect win : wins)
        rectangle(img_rgb, win, cv::Scalar(255, 0, 0), 1);
    groupRectangles(wins, 1, 0.2);
    for (Rect win : wins)
        rectangle(img_rgb, win, cv::Scalar(0, 255, 0), 2);

    namedWindow("Result", cv::WINDOW_AUTOSIZE);
    imshow("Result", img_rgb);
    waitKey(0);

    return 0;
}