#include <iostream>
using namespace std;

class Expression
{
protected:
    typedef auto Value_func( Expression const* ) -> double;

    Value_func* value_func_;

public:
    auto value() const
        -> double
    { return value_func_( this ); }

    Expression(): value_func_( nullptr ) {}     // Like a pure virtual.
};

class Number
    : public Expression
{
private:
    double  number_;

    static
    auto specific_value_func( Expression const* expr )
        -> double
    { return static_cast<Number const*>( expr )->number_; }

public:
    Number( double const number )
        : Expression()
        , number_( number )
    { value_func_ = &Number::specific_value_func; }
};

class Sum
    : public Expression
{
private:
    Expression const*   a_;
    Expression const*   b_;

    static
    auto specific_value_func( Expression const* expr )
        -> double
    {
        auto const p_self  = static_cast<Sum const*>( expr );
        return p_self->a_->value() + p_self->b_->value();
    }

public:
    Sum( Expression const* const a, Expression const* const b )
        : Expression()
        , a_( a )
        , b_( b )
    { value_func_ = &Sum::specific_value_func; }
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