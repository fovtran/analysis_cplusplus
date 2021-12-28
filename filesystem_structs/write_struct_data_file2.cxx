struct a
{
  int    x;
  float  f;
  double d;
  char   c;
  char   s[50];
};

// OK


struct b
{
  int    x;
  float  f;
  double d;
  char   c;
  char*  s;
};

// Wrong! 's' has unknown length; only address of pointer will be written.


struct c
{
  int    x;
  float  f;
  double d;
  char   c;

  struct
  {
    char* s;
  } e;
};

// Wrong! 'e' has a char* member.

#include <string>
#include <fstream>

struct s
{
  // Your POD data here
};

void write(const std::string& file_name, s& data)
{
  std::ofstream out(file_name.c_str());
  out.write(reinterpret_cast<char*>(&s), sizeof(s));
}

void read(const std::string& file_name, s& data)
{
  std::ifstream in(file_name.c_str());
  in.read(reinterpret_cast<char*>(&s), sizeof(s));
}

int main()
{
  s myStruct;
  read("test.dat", myStruct);
  write("test.dat", myStruct);
}


template<typename T>
void write_pod(std::ofstream& out, T& t)
{
  out.write(reinterpret_cast<char*>(&t), sizeof(T));
}

template<typename T>
void read_pod(std::ifstream& in, T& t)
{
  in.read(reinterpret_cast<char*>(&t), sizeof(T));
}

#include <vector>
#include <fstream>
#include <algorithm>

int main()
{
  std::vector<s> myStructs;
  std::ofstream out("test.dat");

  // Fill vector here
  std::for_each(myStructs.begin(), myStructs.end(), std::bind1st(write_pod<s>, out));
}
or for an array

Code:
#include <fstream>
#include <algorithm>

int main()
{
  s myStructs[20];
  std::ofstream out("test.dat");

  // Fill array here
  std::for_each(myStructs, myStructs + 20, std::bind1st(write_pod<s>, out));
}

#include <vector>
#include <fstream>
#include <algorithm>

int main()
{
  std::vector<s> myStructs;
  std::ofstream out("test.dat");

  // Fill vector here
  write_pod<long>(out, myStructs.size());
  std::for_each(myStructs.begin(), myStructs.end(), std::bind1st(write_pod<s>, out));
}
Notice, that 'write_pod()' can also be used with integral types since they are POD's themselves. Unfortunately, the STL does not have an algorithm that allows us to easily read from a 'vector' into a file. So, we will just use a loop. Here is the method to read a 'vector' of structs from a file, I will templatize it to begin with:

Code:
#include <vector>
#include <fstream>

template<typename T>
void read_pod_vector(std::ifstream& in, std::vector<T>& vect)
{
  long size;

  read_pod(in, size);
  vect.resize(size);

  for(int i = 0;i < size;++i)
  {
    T t;
    read_pod(in, t);
    vect.push_back(t);
  }
}

#include <vector>
#include <fstream>
#include <algorithm>

template<typename T>
void write_pod_vector(std::ofstream& out, std::vector<T>& vect)
{
  write_pod<long>(out, vect.size());
  std::for_each(vect.begin(), vect.end(), std::bind1st(write_pod<T>, out));
}
So, implementing these functions to read and write a 'vector' we have:

Code:
#include <vector>
#include <fstream>

int main()
{
  std::vector&lts> myStructs;
  std::ofstream out("test.dat");
  std::ifstream in("test.dat");

  // Fill vector here
  write_pod_vector(out, myStructs);
  out.close();
  read_pod_vector(in, myStructs);
}
So, now you know how to write a POD structure, or a set of POD structures to a file. What if your structure is not POD? Well, then it becomes a bit more complicated. The approach you would take is the exact same as we took with 'vector', after all 'vector' is a non-POD type and we are writing it to a file. That is, you would write the size of the data before the data so you know how much data to read. You may have noticed that the second parameter to read/write is the size of the data, so you don't need loop through all of your data. I will show you now examples that use this fact:

Code:
#include <vector>
#include <fstream>

template<typename T>
void write_pod_vector(std::ofstream& out, std::vector<T>& vect)
{
  long size = myStructs.size();
  write_pod<long>(out, size);
  out.write(reinterpret_cast<char*>(vect.front()), size * sizeof(T));
}
This example takes advantage of vector's congenious structure and second parameter to write which allows us to write the entire 'vector' with one call. We can do the same thing to read:

Code:
#include <vector>
#include <fstream>

template<typename T>
void read_pod_vector(std::ifstream& in, std::vector<T>& vect)
{
  long size;
  read_pod(in, size);
  vect.resize(size);
  in.read(reinterpret_cast<char*>(vect.front()), size * sizeof(T));
}
Not only do we cut out several lines, but we make our program more effient. My next example will show you to read/write other pointer data. The following structure is used to store a variable length string:

Code:
struct str
{
  long  size;
  char* s;
};
Now, here are the read an write methods for this structure.

Code:
#include <fstream>

void write_str( std::ofstream& out, str& s )
{
  out.write(reinterpret_cast<char*>(&s.size), sizeof(long));
  out.write(s.s, s.size * sizeof(char));
}

void read_str(std::ifstream& in, str& s)
{
  in.read(reinterpret_cast<char*>(&s.size), sizeof(long));
  s.s = new char[s.size];
  in.read(s.s, s.size * sizeof(char));
}