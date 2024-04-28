#include <boost/python.hpp>

// 使用する実行環境のPythonインタプリタのパス
// ここで指定するインタプリタパスを利用してpythonモジュールの探索先など実行環境のパスが決まる.
// ここにvenvで作った仮想環境のpythonインタプリタパスを指定した場合、
// 仮想環境下でランタイムが動作する

int main()
{
    Py_Initialize(); // 最初に呼んでおく必要あり

    try
    {
        // Pythonで「print('Hello World!')」を実行
        boost::python::object global = boost::python::import("__main__").attr("__dict__");
        boost::python::exec("print('Hello World!')", global);
    }
    catch (const boost::python::error_already_set &)
    {
        // Pythonコードの実行中にエラーが発生した場合はエラー内容を表示
        PyErr_Print();
    }

    return 0;
}