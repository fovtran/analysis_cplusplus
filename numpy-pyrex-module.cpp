%insert("python") %{ import numpy as np %}

/*! Templated function to copy contents of a container to an allocated memory buffer */
%inline %{
//==== ADDED BY numpy.i
#include <algorithm>

template < typename Container_T >
void copy_to_buffer(
           const Container_T& field,
           typename Container_T::value_type* buffer,
           typename Container_T::size_type length
           )
{
//    ValidateUserInput( length == field.size(),
//            "Destination buffer is the wrong size" );
       // put your own assertion here or BAD THINGS CAN HAPPEN
       if (length == field.size()) {
           std::copy( field.begin(), field.end(), buffer );
       }
}
//====
%}

%define TYPEMAP_COPY_TO_BUFFER(CLASS...)
%typemap(in) (CLASS::value_type* buffer, CLASS::size_type length)
(int res = 0, Py_ssize_t size_ = 0, void *buffer_ = 0) {

       res = PyObject_AsWriteBuffer($input, &buffer_, &size_);
       if ( res < 0 ) {
           PyErr_Clear();
           %argument_fail(res, "(CLASS::value_type*, CLASS::size_type length)",
                   $symname, $argnum);
       }
       $1 = ($1_ltype) buffer_;
       $2 = ($2_ltype) (size_/sizeof($*1_type));
}
%enddef

%define ADD_NUMPY_ARRAY_INTERFACE(PYVALUE, PYCLASS, CLASS...)
TYPEMAP_COPY_TO_BUFFER(CLASS)
%template(_copy_to_buffer_ ## PYCLASS) copy_to_buffer< CLASS >;

%extend CLASS {
%insert("python") %{
def __array__(self):
       """Enable access to this data as a numpy array"""
       a = np.ndarray( shape=( len(self), ), dtype=PYVALUE )
       _copy_to_buffer_ ## PYCLASS(self, a)
       return a
%}
}

%enddef

// then you can make a container "Numpy"-able with
%template(DumbVectorFloat) DumbVector<double>;
ADD_NUMPY_ARRAY_INTERFACE(float, DumbVectorFloat, DumbVector<double>);

// Then in Python, just do:
import numpy as np
# dvf is an instance of DumbVectorFloat
my_numpy_array = np.asarray( dvf )
