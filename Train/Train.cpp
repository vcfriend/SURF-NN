#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Usage: Train pos_samples.yml neg_samples.yml" << endl;
        return -1;
    }

    string pos_samples_file(argv[1]);
    string neg_samples_file(argv[2]);

    FileStorage fs;
    Mat pos_samples;
    Mat neg_samples;

    fs.open(pos_samples_file, FileStorage::READ);
    fs[pos_samples_file.substr(0, pos_samples_file.rfind('.'))] >> pos_samples;
    fs.open(neg_samples_file, FileStorage::READ);
    fs[neg_samples_file.substr(0, neg_samples_file.rfind('.'))] >> neg_samples;
    fs.release();



    return 0;
}
