#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage: Evaluate pos.yml neg.yml" << endl;
        return -1;
    }

    cout << "Reading trained neural network..." << endl;
    CvANN_MLP nn;
    nn.load("nn.yml");

    /* Read training samples */
    cout << "Reading training samples..." << endl;
    string samples_file;
    FileStorage fs;
    Mat samples;
    Mat inputs;
    Mat pos_labels;
    Mat neg_labels;
    Mat labels;

    samples_file = argv[1]; // read positive samples
    fs.open(samples_file, FileStorage::READ);
    fs[samples_file.substr(0, samples_file.rfind('.'))] >> samples;
    inputs.push_back(samples);
    hconcat(Mat::ones(samples.rows, 1, CV_32F), Mat::zeros(samples.rows, 1, CV_32F), pos_labels);

    samples_file = argv[2]; // read negative samples
    fs.open(samples_file, FileStorage::READ);
    fs[samples_file.substr(0, samples_file.rfind('.'))] >> samples;
    inputs.push_back(samples);
    hconcat(Mat::zeros(samples.rows, 1, CV_32F), Mat::ones(samples.rows, 1, CV_32F), neg_labels);

    vconcat(pos_labels, neg_labels, labels);

    samples.release();
    pos_labels.release();
    neg_labels.release();
    fs.release();

    Mat resp;
    nn.predict(inputs, resp);

    fs.open("resp.yml", FileStorage::WRITE);
    fs << "resp" << resp;

    Mat binary_resp;
    threshold(resp, binary_resp, 0, 1, THRESH_BINARY);

    Mat diff;
    compare(labels.col(1), binary_resp.col(1), diff, CMP_NE);
    int nz = countNonZero(diff);

    return 0;
}