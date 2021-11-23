#include <iostream>
#include <string>

int main()
{
    std::string s("Somewhere down the road");
    std::string::size_type prev_pos = 0, pos = 0;

    while( (pos = s.find(' ', pos)) != std::string::npos )
    {
        std::string substring( s.substr(prev_pos, pos-prev_pos) );

        std::cout << substring << '\n';

        prev_pos = ++pos;
    }

    std::string substring( s.substr(prev_pos, pos-prev_pos) ); // Last word
    std::cout << substring << '\n';

    return 0;
}
