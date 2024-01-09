# tl;dr

Numba is a "just in time" (jit) compiler for python.  It is designed to optimize floating point operations and loops which makes it ideal for scientific code. The simplest way to use it is to decorate you function as shown below, which instructs Numba to compile the function into fast machine code the first time it is called.

~~~ python
from numba import jit

@jit(nopython=True)
def dotproduct(v1, v2):
	isum = 0.0
	for i in range(len(v1)):
		isum += v1[i]*v2[i]
	return isum
~~~

Numba provides a quick, [5 minute overview](https://numba.readthedocs.io/en/stable/user/5minguide.html) of the basic functionality which I highly recommend reading.  Here, I will summarize and add elements from their documentation most relevant to writing functions and classes in [packages](/tutorials/git_python_workflow/).

[JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) is a similar tool that "is NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research."  This includes a jit compiler, and other features, but the main features are that:
1. Code can be easily offloaded to accelerators
2. Code can be automatically differentiated

JAX is mostly aimed at machine learning applications, but these features are widely applicable to many scientific domains and problems.

The tutorial below focuses just on numba - JAX is easy to use if you understand this, and there are important caveats the documentation discusses in detail which is better covered therein.

# Overview

From [Numba's documentation](https://numba.pydata.org/):

> "Numba translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN. You don't need to replace the Python interpreter, run a separate compilation step, or even have a C/C++ compiler installed. Just apply one of the Numba decorators to your Python function, and Numba does the rest."

Essentially, Numba is built for scientific computing.  While libraries like [sklearn](https://scikit-learn.org/stable/), [numpy](https://numpy.org/), and [scipy](https://www.scipy.org/) are highly optimized, it can be a tedious task to write and optimize your own custom functions.  [Cython](https://cython.readthedocs.io/en/latest/) is a great alternative for getting to near-C-speed with python code, but it is more cumbersome.  Numba often works nearly as well and requires nothing more than a simple function decoration!

> "Numba is designed to be used with NumPy arrays and functions. Numba generates specialized code for different array data types and layouts to optimize performance. Special decorators can create universal functions that broadcast over NumPy arrays just like NumPy functions do."

Notably, [Pandas](https://pandas.pydata.org/) is not understood by Numba so it just runs operations involving DataFrames, etc. using the python interpreter which is slow.  It is generally advantageous to convert data to numpy arrays to work with if the operations required cannot be handled by Pandas natively.  New DataFrames can always be instantiated after a lengthy series of operations.

The jit behavior means that Numba will compile a function or piece of code, appropriately decorated, the first time it sees it.  So the first time a function is called, it will be much slower than subsequent times since the compiled version is cached.  A new version has to be compiled each time new datatypes are provided, however; this means if you send vectors of integers to the `dotproduct()` function example above, and floats another time, this has to be compiled for each the first time it sees those types of vectors (but only the first time).

# Installation

Numba is available via anaconda or pip;

~~~ bash
$ conda activate my_environment; conda install numba # OR
$ pip install numba
~~~

At the time of writing, Numba is compatible with Python >=3.6, and Numpy versions 1.15 or later.  See their [compatibility guide](https://numba.readthedocs.io/en/stable/user/installing.html#compatibility) for more up to date details.

> "Intel provides a short vector math library (SVML) that contains a large number of optimised transcendental functions available for use as compiler intrinsics. If the icc_rt package is present in the environment (or the SVML libraries are simply locatable!) then Numba automatically configures the LLVM back end to use the SVML intrinsic functions where ever possible."

~~~ bash
$ conda activate my_environment
$ conda install -c numba icc_rt
$ conda install numba
~~~

The SVML can speed up Numba-compiled code even more!  One [final note](https://numba.readthedocs.io/en/stable/user/performance-tips.html#performance-tips) about linear algebra functions:

> "Numba supports most of numpy.linalg in no Python mode. The internal implementation relies on a LAPACK and BLAS library to do the numerical work and it obtains the bindings for the necessary functions from SciPy. Therefore, to achieve good performance in numpy.linalg functions with Numba it is necessary to use a SciPy built against a well optimised LAPACK/BLAS library. In the case of the Anaconda distribution SciPy is built against Intel’s MKL which is highly optimised and as a result Numba makes use of this performance."

# Testing Speed

Since the function is compiled the first time it is called it is not fair to test the speed by simply calling the function.  You actually need to call it once to get it to compile, then time the second (or average subsequent) call(s).  Here is an example file, named tets.py, in which `timeit` is used to measure the performance.  Note that the first time `go_fast` is not included in the setup so is reporting the first call; while the second example correctly calls the code once during the setup.

~~~
from numba import jit
import numpy as np
import timeit

@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

if __name__ == '__main__':
	# Tested the first time (wrong)
	print(timeit.timeit("go_fast(x)", setup="from __main__ import go_fast; import numpy as np; x = np.arange(100).reshape(10, 10)", number=1))

	# Tested the second time (correct)
	print(timeit.timeit("go_fast(x)", setup="from __main__ import go_fast; import numpy as np; x = np.arange(100).reshape(10, 10); go_fast(x)", number=1))
~~~

~~~ bash
$ python test.py
0.35856989584863186
2.9169023036956787e-06
~~~

# The Most Common Decorators

~~~ python
@jit(nopython=True, 
     fastmath=False, 
     parallel=True, 
     nogil=True, 
     cache=True, 
     boundscheck=False)
def my_function(x, y, z):
	...
	return result
~~~

In the above example 6 different options are specified for [jit](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#jit-decorator-cache).  This decoration method is actually all it takes to use Numba! The @jit decorator operates in [two compilation modes](https://numba.readthedocs.io/en/stable/user/5minguide.html): (1) nopython mode and (2) object mode. The former essentially compiles the function so that it can run with having to invoke the python interpreter. **This is best, recommended practice.**  In object mode, the fallback if this fails, loops and things that it can compile will be, whereas everything it cannot compile (like pandas) will continue to run via the interpreter.

1. nopython=True; specifying this to be True forces the function to be compiled in nopython mode. If this is not possible, it will raise an error.  Specifying False (default) will try to compile, but if it fails no error will be raised.
2. fastmath=False; if set to True, this enables the otherwise unsafe floating point transforms described by [LLVM](https://llvm.org/docs/LangRef.html#fast-math-flags). If Intel SVML is installed, faster but less accurate versions of some math intrinsics are used.  For example, loops don't necessarily need to accumulate in [strict order](https://numba.readthedocs.io/en/stable/user/performance-tips.html#performance-tips).
3. parallel=True; if the code contains any parallelizable operations, they will be run in multiple native threads free of the global interpreter lock (GIL).  More fine-grained details are available [here](https://numba.readthedocs.io/en/stable/user/parallel.html), but essentially all atomic operations (multiplication, addition, etc.) are supported.  You can see how well this performs by calling `my_function.test.parallel_diagnostics(level=4)` which provides [diagnotic information](https://numba.readthedocs.io/en/stable/user/parallel.html) about how parallelization is achieved.  See that documentation for a breakdown of the meanings of the different sections of the report.
4. nogil=True; this tries to release the GIL inside the compiled function. This only happens if nopython mode can be achieved, otherwise a compilation warning will be printed.
5. cache=True; this enables file-based caching to reduce compilation times when the function was already compiled previously. According to the documentation: "The cache is maintained in the __pycache__ subdirectory of the directory containing the source file; if the current user is not allowed to write to it, though, it falls back to a platform-specific user-wide cache directory (such as $HOME/.cache/numba on Unix platforms)."  Note that not all functions can be cached - when a function cannot be cached, a warning is raised.
6. boundscheck=False; this disables bounds checking for array indices. Out of bounds accesses will raise IndexError if set to True.  Default behavior is to not do this, in which case segfaults and incorrect results are possible. 

# Vectorizing functions

[Vectorizing](https://numba.readthedocs.io/en/stable/user/vectorize.html#vectorize) a function means you can write a simple function that accepts scalars, then automatically create a version of that operates quickly on arrays the way numpy does.

> "Numba’s vectorize allows Python functions taking scalar input arguments to be used as NumPy ufuncs. Creating a traditional NumPy ufunc is not the most straightforward process and involves writing some C code. Numba makes this easy. Using the vectorize() decorator, Numba can compile a pure Python function into a ufunc that operates over NumPy arrays as fast as traditional ufuncs written in C."

~~~ python
@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)])
def f(x, y):
    return x + y
~~~

In this example, the output type followed by the input types in parentheses are given as its [signature](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#jit-decorator-fastmath) (the same can be done with normal @jit function).

~~~ python
>>> a = np.arange(6)
>>> f(a, a)
array([ 0,  2,  4,  6,  8, 10])
>>> a = np.linspace(0, 1, 6)
>>> f(a, a)
array([ 0. ,  0.4,  0.8,  1.2,  1.6,  2. ])
~~~

However, this will only work for the specified types; trying operate on arrays of different types raises a TypeError.

# Working with Classes

This is a new feature and still under development at the time of writing.  The [@jitclass](https://numba.readthedocs.io/en/stable/user/jitclass.html) decorator can be used to mark a class for optimization.

> "Numba supports code generation for classes via the numba.jitclass() decorator. A class can be marked for optimization using this decorator along with a specification of the types of each field. We call the resulting class object a jitclass. All methods of a jitclass are compiled into nopython functions. The data of a jitclass instance is allocated on the heap as a C-compatible structure so that any compiled functions can have direct access to the underlying data, bypassing the interpreter."

This is an example class from Numba's documentation.

~~~ python
import numpy as np
from numba import int32, float32    # import the types
from numba.experimental import jitclass

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

@jitclass(spec)
class Bag(object):
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] += val
        return self.array

    @staticmethod
    def add(x, y):
        return x + y

n = 21
mybag = Bag(n)
~~~

More examples can be found in their documentation, but this is an experimental feature subject to change.  So far, the following operations are [supported](https://numba.readthedocs.io/en/stable/user/jitclass.html#support-operations):

* calling the jitclass class object to construct a new instance (e.g. mybag = Bag(123))
* read/write access to attributes and properties (e.g. mybag.value)
* calling methods (e.g. mybag.increment(3))
* calling static methods as instance attributes (e.g. mybag.add(1, 1))
* calling static methods as class attributes (e.g. Bag.add(1, 2))

