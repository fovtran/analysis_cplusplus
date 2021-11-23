#include <iostream>
#include <string>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace boost;

int main(int argc, char** argv)
{
    string text = "token  test\tstring";

    char_separator<char> sep(" \t");
    tokenizer<char_separator<char>> tokens(text, sep);
    for (const string& t : tokens)
    {
        cout << t << "." << endl;
    }
}
