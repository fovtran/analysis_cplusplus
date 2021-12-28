class Animal
{
    public:
        void eat() { std::cout << "I'm eating generic food."; }
};

class Cat : public Animal
{
    public:
        void eat() { std::cout << "I'm eating a rat."; }
};
In your main function:

Animal *animal = new Animal;
Cat *cat = new Cat;

animal->eat(); // Outputs: "I'm eating generic food."
cat->eat();    // Outputs: "I'm eating a rat."
So far so good, right? Animals eat generic food, cats eat rats, all without virtual.

Let's change it a little now so that eat() is called via an intermediate function (a trivial function just for this example):

// This can go at the top of the main.cpp file
void func(Animal *xyz) { xyz->eat(); }
Now our main function is:

Animal *animal = new Animal;
Cat *cat = new Cat;

func(animal); // Outputs: "I'm eating generic food."
func(cat);    // Outputs: "I'm eating generic food."
Uh oh... we passed a Cat into func(), but it won't eat rats. Should you overload func() so it takes a Cat*? If you have to derive more animals from Animal they would all need their own func().

The solution is to make eat() from the Animal class a virtual function:

class Animal
{
    public:
        virtual void eat() { std::cout << "I'm eating generic food."; }
};

class Cat : public Animal
{
    public:
        void eat() { std::cout << "I'm eating a rat."; }
};
Main:

func(animal); // Outputs: "I'm eating generic food."
func(cat);    // Outputs: "I'm eating a rat."

class Base
{
  public:
            void Method1 ()  {  std::cout << "Base::Method1" << std::endl;  }
    virtual void Method2 ()  {  std::cout << "Base::Method2" << std::endl;  }
};

class Derived : public Base
{
  public:
    void Method1 ()  {  std::cout << "Derived::Method1" << std::endl;  }
    void Method2 ()  {  std::cout << "Derived::Method2" << std::endl;  }
};

Base* obj = new Derived ();
  //  Note - constructed as Derived, but pointer stored as Base*

obj->Method1 ();  //  Prints "Base::Method1"
obj->Method2 ();  //  Prints "Derived::Method2"


class Animal
{        
    public: 
      // turn the following virtual modifier on/off to see what happens
      //virtual   
      std::string Says() { return "?"; }  
};

class Dog: public Animal
{
    public: std::string Says() { return "Woof"; }
};

void test()
{
    Dog* d = new Dog();
    Animal* a = d;       // refer to Dog instance with Animal pointer

    cout << d->Says();   // always Woof
    cout << a->Says();   // Woof or ?, depends on virtual
	
	
	