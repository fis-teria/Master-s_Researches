#define PY_SSIZE_T_CLEAN

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
#define PYTHON_EXE_PATH u8"Path to python"

int main(int argc, char** argv)
{
    int result = 0; 
    // このポインタはFinalizeが完了するまで保持する
    wchar_t* program = Py_DecodeLocale(PYTHON_EXE_PATH, NULL);
    if (program == nullptr) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
    }
    else {
 
        // 使用したい実行環境のpythonインタプリタのパスを指定
        Py_SetProgramName(program);

        // 初期化
        Py_Initialize();

        // Python実行環境の参照パスを出力してみる
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("print(sys.path)\n");
        PyRun_SimpleString("print(sys.prefix)\n");
        PyRun_SimpleString("print(sys.exec_prefix)\n");

        constexpr int ERROR_RET      = 1;
        constexpr int SUCCESS_RET  = 0;

        // 環境終了処理
        int finalize_result = Py_FinalizeEx();

        int result = (finalize_result < 0) ? ERROR_RET : SUCCESS_RET;
        PyMem_RawFree(program);
    }
    return result; 
}
