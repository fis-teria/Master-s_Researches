#define PY_SSIZE_T_CLEAN
#include <iostream>

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

int main(int argc, char** argv)
{
    int result = 0; 
    // このポインタはFinalizeが完了するまで保持する
    wchar_t* program = Py_DecodeLocale(PYTHON_EXE_PATH, NULL);

 
    // 使用したい実行環境のpythonインタプリタのパスを指定
    Py_SetProgramName(program);

    // 初期化
    Py_Initialize();

    // Python実行環境の参照パスを出力してみる
    /*
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("print(sys.path)\n");
    PyRun_SimpleString("print(sys.prefix)\n");
    PyRun_SimpleString("print(sys.exec_prefix)\n");//*/
    /*
    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *path = PyObject_GetAttrString(sys, "path");
    PyList_Append(sys, PyUnicode_FromString("./"));//*/    

    std::cout << "start read .py";
    PyObject *pName = PyUnicode_DecodeFSDefault("../pycoral/code/echo");
    PyObject *pModule = PyImport_Import(pName); 
    std::cout << "\tOK\n";

    std::cout << "install module";
    PyObject *pFunc = PyObject_GetAttrString(pModule, "main");
    std::cout << "\tOK";
    /* pFunc is a new reference */

    if (pFunc && PyCallable_Check(pFunc)) {
    }
    Py_XDECREF(pFunc);
    constexpr int ERROR_RET = -1;
    constexpr int SUCCESS_RET  = 0;

    // 環境終了処理
    int finalize_result = Py_FinalizeEx();

    PyMem_RawFree(program);

    return 0; 
}
