#include <vector>
#include <string>
#include <sstream>

int main()
{
    std::string str("Split me by whitespaces");
    std::string buf;                 // Have a buffer string
    std::stringstream ss(str);       // Insert the string into a stream

    std::vector<std::string> tokens; // Create vector to hold our words

    while (ss >> buf)
        tokens.push_back(buf);

    return 0;
}
