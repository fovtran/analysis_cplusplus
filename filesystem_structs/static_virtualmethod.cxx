#include <iostream>
using namespace std;

class Expression
{
public:
    auto value() const
        -> double
    { return 0.0; }         // This should never be invoked, really.
};

class Number
    : public Expression
{
private:
    double  number_;

public:
    auto value() const
        -> double
    { return number_; }     // This is OK.

    Number( double const number )
        : Expression()
        , number_( number )
    {}
};

class Sum
    : public Expression
{
private:
    Expression const*   a_;
    Expression const*   b_;

public:
    auto value() const
        -> double
    { return a_->value() + b_->value(); }       // Uhm, bad! Very bad!

    Sum( Expression const* const a, Expression const* const b )
        : Expression()
        , a_( a )
        , b_( b )
    {}
};

auto main() -> int
{
    Number const    a( 3.14 );
    Number const    b( 2.72 );
    Number const    c( 1.0 );

    Sum const       sum_ab( &a, &b );
    Sum const       sum( &sum_ab, &c );

    cout << sum.value() << endl;
}