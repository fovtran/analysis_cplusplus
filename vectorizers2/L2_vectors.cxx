#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

int main()
{
     const int size = 5;
     vector<float> v1{1,1,1,1,1};
     vector<float> v2{1,1,1,1,1};
     vector<float> v3{1,1,1,1,1};
     vector<vector<float>> V;
     V.push_back(v1);
     V.push_back(v2);
     V.push_back(v3);
     vector<float> doc(size,0.0);
     //============================================//
     //STEP 1: Average Pooling
     for(size_t i = 0; i < V.size();i++)
     {
         for(size_t j = 0; j < 5;j++)
         {
             doc[j] += V[i][j];
         }
     }

     for(size_t i = 0; i < doc.size();i++)
     {
        doc[i]= doc[i]/(float) V.size();
     }
    //============================================//
    //STEP 2: L2-Normalization
     float y =  *max_element(std::begin(doc), std::end(doc));
     float m_sum = 0.0;

     for (int k = 0; k < doc.size(); k++)
     {
        m_sum +=  pow(doc[k]/y,2);
     }
     //STEP 3: Divide components by the Norm (m_sum)
     for(size_t i = 0; i < V.size();i++)
     {
        cout << doc[i]/sqrt(m_sum)<<"  ";
     }
     cout << endl;
     return 0;
}
