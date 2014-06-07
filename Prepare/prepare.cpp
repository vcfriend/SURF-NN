#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree.hpp>

#define SURF_HESSIAN_THRESHOLD 400
#define BOW_CLUSTER_COUNT 100

using namespace std;
using namespace cv;

void progress(int i, size_t size)
{
        if (i != 0) cout << "\b\b\b";
        cout << setw(2) << i * 100 / size << "%";
        if (i == size) cout << endl;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage: Prepare [-p|-n] image_list" << endl;
        return -1;
    }

    int pos_flag;
    if (strcmp(argv[1], "-n") == 0)
        pos_flag = 0;
    else if (strcmp(argv[1], "-p") == 0)
        pos_flag = 1;
    else
        return -1;

    /* get filepaths vector */
    ifstream filelist;
    vector<string> filepaths;
    string filepath;

    filelist.open(argv[2]);
    if (!filelist) {
        cout << "Can't read image_list!" << endl;
        return -1;
    }
    while (getline(filelist, filepath))
        filepaths.push_back(filepath);

    /* get BOW vocabulary */
    Mat img;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat all_descriptors;
    Ptr<FeatureDetector> detector(new SurfFeatureDetector(SURF_HESSIAN_THRESHOLD));
    Ptr<DescriptorExtractor> extractor(new SurfDescriptorExtractor);
    Mat vocabulary;
    FileStorage fs;

    if (!pos_flag)
    {
        cout << "Build negative training set." << endl;
        if (!fs.open("vocabulary.yml", FileStorage::READ)) {
            cout << "Can't read vocabulary file." << endl;
            return -1;
        }
        fs["vocabulary"] >> vocabulary;
        fs.release();
    }
    else
    {
        cout << "Build positive training set." << endl;
        cout << "Building vocabulary..." << endl;
        BOWKMeansTrainer bowtrainer(BOW_CLUSTER_COUNT);

        cout << "Extracting descriptors...";
        for (int i = 0; i < filepaths.size(); i++)
        {
            img = imread(filepaths[i], 0);
            if (img.empty()) {
                cout << "Can't read " << filepaths[i] << "in file list!" << endl;
                return -1;
            }
            detector->detect(img, keypoints);
            extractor->compute(img, keypoints, descriptors);
            all_descriptors.push_back(descriptors);
            progress(i, filepaths.size() - 1);
        }

        cout << "Clustering descriptors..." << endl;
        vocabulary = bowtrainer.cluster(all_descriptors);

        cout << "Writing to vocabulary.yml..." << endl;
        fs.open("vocabulary.yml", FileStorage::WRITE);
        fs << "vocabulary" << vocabulary;

        fs.release();
        all_descriptors.release();
    }

    /* build training set */
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    BOWImgDescriptorExtractor bowextractor(extractor, matcher);
    bowextractor.setVocabulary(vocabulary);
    Mat bowdescriptor;
    int cluster_count = vocabulary.rows;

    cout << "Building training set...";
    if (pos_flag) {
        for (int i = 0; i < filepaths.size(); i++)
        {
            img = imread(filepaths[i], 0);
            detector->detect(img, keypoints);
            bowextractor.compute(img, keypoints, bowdescriptor);
            all_descriptors.push_back(bowdescriptor);
            progress(i, filepaths.size() - 1);
        }
    }
    else {
        for (int i = 0; i < filepaths.size(); i++)
        {
            img = imread(filepaths[i], 0);
            detector->detect(img, keypoints);
            extractor->compute(img, keypoints, descriptors);

            Mat keypoints_map(img.size(), CV_16UC1, Scalar::all(0));
            for (int j = 0; j < keypoints.size(); j++)
                keypoints_map.at<ushort>((int)keypoints[j].pt.y, (int)keypoints[j].pt.x) = j;

            //for (int l = 80; l < img.cols && l < img.rows; l+=10) {
            { int l = 80;
                //cout << l << ' ' << '(' << img.rows << ',' << img.cols << ')' << endl;
                Rect win(0, 0, l, l);
                for (win.y = 0; win.y <= img.rows - win.height; win.y+=win.height) {
                    for (win.x = 0; win.x <= img.cols - win.width; win.x+=win.width) {
                        Mat subdescriptors;
                        SparseMat subkeypoints_map(keypoints_map(win));
                        for (SparseMatConstIterator it = subkeypoints_map.begin(); it != subkeypoints_map.end(); it++) {
                            ushort index = it.value<ushort>();
                            KeyPoint *kp = &keypoints[index];
                            if (kp->pt.x - kp->size > win.x && kp->pt.x + kp->size < win.x + win.width
                             && kp->pt.y - kp->size > win.y && kp->pt.y + kp->size < win.y + win.height)
                                subdescriptors.push_back(descriptors.row(index));
                        }

                        if (subdescriptors.rows > 0) {
                            vector<DMatch> matches;
                            matcher->match(subdescriptors, matches);

                            Mat bowdescriptor(1, cluster_count, CV_32FC1, Scalar::all(0.0));
                            float *dptr = (float*)bowdescriptor.data;
                            for (int k = 0; k < matches.size(); k++)
                                dptr[matches[k].trainIdx] += 1.f;
                            bowdescriptor /= subdescriptors.rows;

                            all_descriptors.push_back(bowdescriptor);
                        }
                    }
                }
            }

            progress(i, filepaths.size() - 1);
        }
    }

    string samples_file;
    if (pos_flag)
        samples_file = "pos";
    else
        samples_file = "neg";

    cout << "Writing to " << samples_file << ".yml..." << endl;
    fs.open(samples_file + ".yml", FileStorage::WRITE);
    fs << samples_file << all_descriptors;
    fs.release();

    cout << "Done." << endl;

    return 0;
}