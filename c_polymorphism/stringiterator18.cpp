#include <vector>
#include <string>
using namespace std;

vector<string> split(string data, string token)
{
    vector<string> output;
    size_t pos = string::npos; // size_t to avoid improbable overflow
    do
    {
        pos = data.find(token);
        output.push_back(data.substr(0, pos));
        if (string::npos != pos)
            data = data.substr(pos + token.size());
    } while (string::npos != pos);
    return output;
}
can use any string as delimiter, also can be used with binary data (std::string supports binary data, including nulls)

using:

auto a = split("this!!is!!!example!string", "!!");
