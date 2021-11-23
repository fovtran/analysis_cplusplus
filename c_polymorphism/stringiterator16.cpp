#include <regex.h>
#include <string.h>
#include <vector.h>

using namespace std;

vector<string> split(string s){
    regex r ("\\w+"); //regex matches whole words, (greedy, so no fragment words)
    regex_iterator<string::iterator> rit ( s.begin(), s.end(), r );
    regex_iterator<string::iterator> rend; //iterators to iterate thru words
    vector<string> result<regex_iterator>(rit, rend);
    return result;  //iterates through the matches to fill the vector
}
