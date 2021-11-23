// abstract class CPolygon
class CPolygon {
  protected:
    int width, height;
  public:
    void set_values (int a, int b)
      { width=a; height=b; }
    virtual int area () =0;
};


CPolygon poly;


would not be valid for the abstract base class we have just declared, because tries to instantiate an object. Nevertheless, the following pointers:

1
2
CPolygon * ppoly1;
CPolygon * ppoly2;
