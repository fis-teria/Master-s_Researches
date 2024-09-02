#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>

// デバッグ版へのリンクを避けるため _DEBUG を一時的に無効化.
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

// 使用する実行環境のPythonインタプリタのパス
// ここで指定するインタプリタパスを利用してpythonモジュールの探索先など実行環境のパスが決まる.
// ここにvenvで作った仮想環境のpythonインタプリタパスを指定した場合、
// 仮想環境下でランタイムが動作する
#define PYTHON_EXE_PATH u8"/home/isight/.pyenv/versions/3.9.19/bin/python3"

PyObject *img_compression(cv::Mat &img)
{
    int mode = 1; // 0 normal 1 compression
    int list_size = img.cols * img.rows;
    PyObject *pValueR, *pValueG, *pValueB, *pList, *pLists;

    if (mode == 1)
        pList = PyList_New(list_size / 8 * 3);
    else
        pList = PyList_New(list_size * 3);

    // compression
    std::vector<std::vector<long long int>> st(106, std::vector<long long int>(480));
    int st_x = 0;
    int st_y = 0;
    int section = img.cols/8 * img.rows;
    long long int b1 = 0;
    long long int b2 = 0;
    long long int g1 = 0;
    long long int g2 = 0;
    long long int r1 = 0;
    long long int r2 = 0;

    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            if (mode == 1)
            {
                b1 *= 256;
                b2 = img.at<cv::Vec3b>(y, x)[0];
                b1 = b1 | b2;
                g1 *= 256;
                g2 = img.at<cv::Vec3b>(y, x)[1];
                g1 = g1 | g2;
                r1 *= 256;
                r2 = img.at<cv::Vec3b>(y, x)[2];
                r1 = r1 | r2;
                if ((x + 1) % 8 == 0)
                {
                    st[st_x][st_y] = r1;
                    pValueB = PyLong_FromLongLong(b1);
                    pValueG = PyLong_FromLongLong(g1);
                    pValueR = PyLong_FromLongLong(r1);
                    if (!pValueR)
                    {
                        fprintf(stderr, "Cannot convert argument\n");
                        return pValueR;
                    }
                    // pValue reference stolen here:
                    if (r1 == PyLong_AsLongLong(pValueR))
                    {
                        //std::cout << img.cols / 8 * st_y + st_x << " " << section + img.cols / 8 * st_y + st_x << " " << section * 2 + img.cols / 8 * st_y + st_x << std::endl;
                        PyList_SetItem(pList, img.cols / 8 * st_y + st_x, pValueB);
                        PyList_SetItem(pList, section + img.cols / 8 * st_y + st_x, pValueG);
                        PyList_SetItem(pList, section * 2 + img.cols / 8 * st_y + st_x, pValueR);
                    }
                    else
                    {
                        std::cout << "cannnot convert" << std::endl;
                        return pValueR;
                    }
                    st_x++;
                    b1 = 0;
                    g1 = 0;
                    r1 = 0;
                }
            }
            else
            {
                // std::cout << img.at<unsigned char>(y, x)<< std::endl;
                pValueG = PyLong_FromLong(img.at<unsigned char>(y, x));
                PyList_SetItem(pList, img.cols * st_y + st_x, pValueG);
            }
        }
        st_x = 0;
        st_y++;
    }

    return pList;
}

int main(int argc, char **argv)
{

    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue, *pList;
    int i;
    // std::vector<std::vector<long long int>> vec(106, std::vector<long long int>(480));

    if (argc < 3)
    {
        fprintf(stderr, "Usage: call pythonfile funcname [args]\n");
        return 1;
    }

    Py_Initialize();
    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyUnicode_DecodeFSDefault("../pycoral/code"));

    pName = PyUnicode_DecodeFSDefault(argv[1]);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL)
    {
        pFunc = PyObject_GetAttrString(pModule, argv[2]);

        // img read
        cv::Mat img = cv::imread("ex_data/color/000030.jpg", 1);
        //cv::resize(img, img, cv::Size(), 0.25, 0.25);

        // img compression
        pList = PyList_New(img.cols / 8 * img.rows);
        pList = img_compression(img);

        pArgs = PyTuple_New(3);
        PyTuple_SetItem(pArgs, 0, pList);
        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(img.cols));
        PyTuple_SetItem(pArgs, 2, PyLong_FromLong(img.rows));
        std::cout << PyTuple_Size(pArgs) << std::endl;
        pValue = PyObject_CallObject(pFunc, pArgs);
        std::cout << std::endl;
    }
    else
    {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
        return 1;
    }
    if (Py_FinalizeEx() < 0)
    {
        return 120;
    }

    return 0;
}
