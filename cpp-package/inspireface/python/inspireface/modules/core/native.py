__docformat__ = "restructuredtext"

# Begin preamble for Python

import ctypes
import sys
from ctypes import *  # noqa: F401, F403

from ._library_path import get_lib_path

_LIBRARY_FILENAME = get_lib_path()

_int_types = (ctypes.c_int16, ctypes.c_int32)
if hasattr(ctypes, "c_int64"):
    # Some builds of ctypes apparently do not have ctypes.c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (ctypes.c_int64,)
for t in _int_types:
    if ctypes.sizeof(t) == ctypes.sizeof(ctypes.c_size_t):
        c_ptrdiff_t = t
del t
del _int_types



class UserString:
    def __init__(self, seq):
        if isinstance(seq, bytes):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq).encode()

    def __bytes__(self):
        return self.data

    def __str__(self):
        return self.data.decode()

    def __repr__(self):
        return repr(self.data)

    def __int__(self):
        return int(self.data.decode())

    def __long__(self):
        return int(self.data.decode())

    def __float__(self):
        return float(self.data.decode())

    def __complex__(self):
        return complex(self.data.decode())

    def __hash__(self):
        return hash(self.data)

    def __le__(self, string):
        if isinstance(string, UserString):
            return self.data <= string.data
        else:
            return self.data <= string

    def __lt__(self, string):
        if isinstance(string, UserString):
            return self.data < string.data
        else:
            return self.data < string

    def __ge__(self, string):
        if isinstance(string, UserString):
            return self.data >= string.data
        else:
            return self.data >= string

    def __gt__(self, string):
        if isinstance(string, UserString):
            return self.data > string.data
        else:
            return self.data > string

    def __eq__(self, string):
        if isinstance(string, UserString):
            return self.data == string.data
        else:
            return self.data == string

    def __ne__(self, string):
        if isinstance(string, UserString):
            return self.data != string.data
        else:
            return self.data != string

    def __contains__(self, char):
        return char in self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__class__(self.data[index])

    def __getslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, bytes):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other).encode())

    def __radd__(self, other):
        if isinstance(other, bytes):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other).encode() + self.data)

    def __mul__(self, n):
        return self.__class__(self.data * n)

    __rmul__ = __mul__

    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self):
        return self.__class__(self.data.capitalize())

    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))

    def count(self, sub, start=0, end=sys.maxsize):
        return self.data.count(sub, start, end)

    def decode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())

    def encode(self, encoding=None, errors=None):  # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())

    def endswith(self, suffix, start=0, end=sys.maxsize):
        return self.data.endswith(suffix, start, end)

    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))

    def find(self, sub, start=0, end=sys.maxsize):
        return self.data.find(sub, start, end)

    def index(self, sub, start=0, end=sys.maxsize):
        return self.data.index(sub, start, end)

    def isalpha(self):
        return self.data.isalpha()

    def isalnum(self):
        return self.data.isalnum()

    def isdecimal(self):
        return self.data.isdecimal()

    def isdigit(self):
        return self.data.isdigit()

    def islower(self):
        return self.data.islower()

    def isnumeric(self):
        return self.data.isnumeric()

    def isspace(self):
        return self.data.isspace()

    def istitle(self):
        return self.data.istitle()

    def isupper(self):
        return self.data.isupper()

    def join(self, seq):
        return self.data.join(seq)

    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))

    def lower(self):
        return self.__class__(self.data.lower())

    def lstrip(self, chars=None):
        return self.__class__(self.data.lstrip(chars))

    def partition(self, sep):
        return self.data.partition(sep)

    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))

    def rfind(self, sub, start=0, end=sys.maxsize):
        return self.data.rfind(sub, start, end)

    def rindex(self, sub, start=0, end=sys.maxsize):
        return self.data.rindex(sub, start, end)

    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))

    def rpartition(self, sep):
        return self.data.rpartition(sep)

    def rstrip(self, chars=None):
        return self.__class__(self.data.rstrip(chars))

    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)

    def splitlines(self, keepends=0):
        return self.data.splitlines(keepends)

    def startswith(self, prefix, start=0, end=sys.maxsize):
        return self.data.startswith(prefix, start, end)

    def strip(self, chars=None):
        return self.__class__(self.data.strip(chars))

    def swapcase(self):
        return self.__class__(self.data.swapcase())

    def title(self):
        return self.__class__(self.data.title())

    def translate(self, *args):
        return self.__class__(self.data.translate(*args))

    def upper(self):
        return self.__class__(self.data.upper())

    def zfill(self, width):
        return self.__class__(self.data.zfill(width))


class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""

    def __init__(self, string=""):
        self.data = string

    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")

    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data):
            raise IndexError
        self.data = self.data[:index] + sub + self.data[index + 1 :]

    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data):
            raise IndexError
        self.data = self.data[:index] + self.data[index + 1 :]

    def __setslice__(self, start, end, sub):
        start = max(start, 0)
        end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start] + sub.data + self.data[end:]
        elif isinstance(sub, bytes):
            self.data = self.data[:start] + sub + self.data[end:]
        else:
            self.data = self.data[:start] + str(sub).encode() + self.data[end:]

    def __delslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]

    def immutable(self):
        return UserString(self.data)

    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, bytes):
            self.data += other
        else:
            self.data += str(other).encode()
        return self

    def __imul__(self, n):
        self.data *= n
        return self


class String(MutableString, ctypes.Union):

    _fields_ = [("raw", ctypes.POINTER(ctypes.c_char)), ("data", ctypes.c_char_p)]

    def __init__(self, obj=b""):
        if isinstance(obj, (bytes, UserString)):
            self.data = bytes(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(ctypes.POINTER(ctypes.c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from bytes
        elif isinstance(obj, bytes):
            return cls(obj)

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj.encode())

        # Convert from c_char_p
        elif isinstance(obj, ctypes.c_char_p):
            return obj

        # Convert from POINTER(ctypes.c_char)
        elif isinstance(obj, ctypes.POINTER(ctypes.c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(ctypes.cast(obj, ctypes.POINTER(ctypes.c_char)))

        # Convert from ctypes.c_char array
        elif isinstance(obj, ctypes.c_char * len(obj)):
            return obj

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)

    from_param = classmethod(from_param)


def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)


# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to ctypes.c_void_p.
def UNCHECKED(type):
    if hasattr(type, "_type_") and isinstance(type._type_, str) and type._type_ != "P":
        return type
    else:
        return ctypes.c_void_p


# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self, func, restype, argtypes, errcheck):
        self.func = func
        self.func.restype = restype
        self.argtypes = argtypes
        if errcheck:
            self.func.errcheck = errcheck

    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func

    def __call__(self, *args):
        fixed_args = []
        i = 0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i += 1
        return self.func(*fixed_args + list(args[i:]))


def ord_if_char(value):
    """
    Simple helper used for casts to simple builtin types:  if the argument is a
    string type, it will be converted to it's ordinal value.

    This function will raise an exception if the argument is string with more
    than one characters.
    """
    return ord(value) if (isinstance(value, bytes) or isinstance(value, str)) else value

# End preamble

_libs = {}
_libdirs = []

from ._native_loader import (
    DarwinLibraryLoader,
    LibraryLoader,
    PosixLibraryLoader,
    WindowsLibraryLoader,
    add_library_search_dirs,
    load_library,
)

add_library_search_dirs([])

# Begin libraries
_libs[_LIBRARY_FILENAME] = load_library(_LIBRARY_FILENAME)
# 1 libraries
# End libraries

# No modules

HFImageStream = POINTER(None)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 11

PHFImageStream = POINTER(POINTER(None))# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 12

HFSession = POINTER(None)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 13

PHFSession = POINTER(POINTER(None))# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 14

HFImageBitmap = POINTER(None)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 15

PHFImageBitmap = POINTER(POINTER(None))# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 16

HFFaceResultSnapshot = POINTER(None)

PHFFaceResultSnapshot = POINTER(POINTER(None))

HFFaceCaptureSession = POINTER(None)

PHFFaceCaptureSession = POINTER(POINTER(None))

HPVoid = POINTER(None)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 17

HFloat = c_float# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 19

HPFloat = POINTER(c_float)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 20

HPUInt8 = POINTER(c_ubyte)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 23

HInt32 = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 24

HOption = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 25

HPInt32 = POINTER(c_int)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 26

HFStatus = c_int32

HFUInt32 = c_uint32

HFUInt64 = c_uint64

HFaceId = c_int64# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 27

HPFaceId = POINTER(c_int64)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 28

HResult = c_long# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 29

HString = String# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 30

HPath = String# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 31

HFormat = String# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 32

HChar = c_char# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 34

HPBuffer = String# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 35

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 45
class struct_HFaceRect(Structure):
    pass

struct_HFaceRect.__slots__ = [
    'x',
    'y',
    'width',
    'height',
]
struct_HFaceRect._fields_ = [
    ('x', HInt32),
    ('y', HInt32),
    ('width', HInt32),
    ('height', HInt32),
]

HFaceRect = struct_HFaceRect# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 45

PHFaceRect = POINTER(struct_HFaceRect)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 45

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 50
class struct_HPoint2f(Structure):
    pass

struct_HPoint2f.__slots__ = [
    'x',
    'y',
]
struct_HPoint2f._fields_ = [
    ('x', HFloat),
    ('y', HFloat),
]

HPoint2f = struct_HPoint2f# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 50

PHPoint2f = POINTER(struct_HPoint2f)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 50

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 55
class struct_HPoint2i(Structure):
    pass

struct_HPoint2i.__slots__ = [
    'x',
    'y',
]
struct_HPoint2i._fields_ = [
    ('x', HInt32),
    ('y', HInt32),
]

HPoint2i = struct_HPoint2i# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 55

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 61
class struct_HColor(Structure):
    pass

struct_HColor.__slots__ = [
    'r',
    'g',
    'b',
]
struct_HColor._fields_ = [
    ('r', HFloat),
    ('g', HFloat),
    ('b', HFloat),
]

HColor = struct_HColor# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/intypedef.h: 61

enum_HFImageFormat = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

HF_STREAM_RGB = 0# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

HF_STREAM_BGR = 1# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

HF_STREAM_RGBA = 2# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

HF_STREAM_BGRA = 3# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

HF_STREAM_YUV_NV12 = 4# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

HF_STREAM_YUV_NV21 = 5# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

HF_STREAM_I420 = 6# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

HF_STREAM_GRAY = 7# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

HFImageFormat = enum_HFImageFormat# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 100

enum_HFRotation = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 111

HF_CAMERA_ROTATION_0 = 0# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 111

HF_CAMERA_ROTATION_90 = 1# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 111

HF_CAMERA_ROTATION_180 = 2# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 111

HF_CAMERA_ROTATION_270 = 3# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 111

HFRotation = enum_HFRotation# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 111

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 123
class struct_HFImageData(Structure):
    pass

struct_HFImageData.__slots__ = [
    'data',
    'width',
    'height',
    'format',
    'rotation',
]
struct_HFImageData._fields_ = [
    ('data', HPUInt8),
    ('width', HInt32),
    ('height', HInt32),
    ('format', HFImageFormat),
    ('rotation', HFRotation),
]

HFImageData = struct_HFImageData# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 123

PHFImageData = POINTER(struct_HFImageData)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 123

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 134
if _libs[_LIBRARY_FILENAME].has("HFCreateImageStream", "cdecl"):
    HFCreateImageStream = _libs[_LIBRARY_FILENAME].get("HFCreateImageStream", "cdecl")
    HFCreateImageStream.argtypes = [PHFImageData, PHFImageStream]
    HFCreateImageStream.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 144
if _libs[_LIBRARY_FILENAME].has("HFCreateImageStreamEmpty", "cdecl"):
    HFCreateImageStreamEmpty = _libs[_LIBRARY_FILENAME].get("HFCreateImageStreamEmpty", "cdecl")
    HFCreateImageStreamEmpty.argtypes = [PHFImageStream]
    HFCreateImageStreamEmpty.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 155
if _libs[_LIBRARY_FILENAME].has("HFImageStreamSetBuffer", "cdecl"):
    HFImageStreamSetBuffer = _libs[_LIBRARY_FILENAME].get("HFImageStreamSetBuffer", "cdecl")
    HFImageStreamSetBuffer.argtypes = [HFImageStream, HPUInt8, HInt32, HInt32]
    HFImageStreamSetBuffer.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 164
if _libs[_LIBRARY_FILENAME].has("HFImageStreamSetRotation", "cdecl"):
    HFImageStreamSetRotation = _libs[_LIBRARY_FILENAME].get("HFImageStreamSetRotation", "cdecl")
    HFImageStreamSetRotation.argtypes = [HFImageStream, HFRotation]
    HFImageStreamSetRotation.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 173
if _libs[_LIBRARY_FILENAME].has("HFImageStreamSetFormat", "cdecl"):
    HFImageStreamSetFormat = _libs[_LIBRARY_FILENAME].get("HFImageStreamSetFormat", "cdecl")
    HFImageStreamSetFormat.argtypes = [HFImageStream, HFImageFormat]
    HFImageStreamSetFormat.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 183
if _libs[_LIBRARY_FILENAME].has("HFReleaseImageStream", "cdecl"):
    HFReleaseImageStream = _libs[_LIBRARY_FILENAME].get("HFReleaseImageStream", "cdecl")
    HFReleaseImageStream.argtypes = [HFImageStream]
    HFReleaseImageStream.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 200
class struct_HFImageBitmapData(Structure):
    pass

struct_HFImageBitmapData.__slots__ = [
    'data',
    'width',
    'height',
    'channels',
]
struct_HFImageBitmapData._fields_ = [
    ('data', HPUInt8),
    ('width', HInt32),
    ('height', HInt32),
    ('channels', HInt32),
]

HFImageBitmapData = struct_HFImageBitmapData# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 200

PHFImageBitmapData = POINTER(struct_HFImageBitmapData)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 200

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 209
if _libs[_LIBRARY_FILENAME].has("HFCreateImageBitmap", "cdecl"):
    HFCreateImageBitmap = _libs[_LIBRARY_FILENAME].get("HFCreateImageBitmap", "cdecl")
    HFCreateImageBitmap.argtypes = [PHFImageBitmapData, PHFImageBitmap]
    HFCreateImageBitmap.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 219
if _libs[_LIBRARY_FILENAME].has("HFCreateImageBitmapFromFilePath", "cdecl"):
    HFCreateImageBitmapFromFilePath = _libs[_LIBRARY_FILENAME].get("HFCreateImageBitmapFromFilePath", "cdecl")
    HFCreateImageBitmapFromFilePath.argtypes = [HPath, HInt32, PHFImageBitmap]
    HFCreateImageBitmapFromFilePath.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 228
if _libs[_LIBRARY_FILENAME].has("HFImageBitmapCopy", "cdecl"):
    HFImageBitmapCopy = _libs[_LIBRARY_FILENAME].get("HFImageBitmapCopy", "cdecl")
    HFImageBitmapCopy.argtypes = [HFImageBitmap, PHFImageBitmap]
    HFImageBitmapCopy.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 236
if _libs[_LIBRARY_FILENAME].has("HFReleaseImageBitmap", "cdecl"):
    HFReleaseImageBitmap = _libs[_LIBRARY_FILENAME].get("HFReleaseImageBitmap", "cdecl")
    HFReleaseImageBitmap.argtypes = [HFImageBitmap]
    HFReleaseImageBitmap.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 246
if _libs[_LIBRARY_FILENAME].has("HFCreateImageStreamFromImageBitmap", "cdecl"):
    HFCreateImageStreamFromImageBitmap = _libs[_LIBRARY_FILENAME].get("HFCreateImageStreamFromImageBitmap", "cdecl")
    HFCreateImageStreamFromImageBitmap.argtypes = [HFImageBitmap, HFRotation, PHFImageStream]
    HFCreateImageStreamFromImageBitmap.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 257
if _libs[_LIBRARY_FILENAME].has("HFCreateImageBitmapFromImageStreamProcess", "cdecl"):
    HFCreateImageBitmapFromImageStreamProcess = _libs[_LIBRARY_FILENAME].get("HFCreateImageBitmapFromImageStreamProcess", "cdecl")
    HFCreateImageBitmapFromImageStreamProcess.argtypes = [HFImageStream, PHFImageBitmap, HInt32, HFloat]
    HFCreateImageBitmapFromImageStreamProcess.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 267
if _libs[_LIBRARY_FILENAME].has("HFImageBitmapWriteToFile", "cdecl"):
    HFImageBitmapWriteToFile = _libs[_LIBRARY_FILENAME].get("HFImageBitmapWriteToFile", "cdecl")
    HFImageBitmapWriteToFile.argtypes = [HFImageBitmap, HPath]
    HFImageBitmapWriteToFile.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 278
if _libs[_LIBRARY_FILENAME].has("HFImageBitmapDrawRect", "cdecl"):
    HFImageBitmapDrawRect = _libs[_LIBRARY_FILENAME].get("HFImageBitmapDrawRect", "cdecl")
    HFImageBitmapDrawRect.argtypes = [HFImageBitmap, HFaceRect, HColor, HInt32]
    HFImageBitmapDrawRect.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 290
if _libs[_LIBRARY_FILENAME].has("HFImageBitmapDrawCircleF", "cdecl"):
    HFImageBitmapDrawCircleF = _libs[_LIBRARY_FILENAME].get("HFImageBitmapDrawCircleF", "cdecl")
    HFImageBitmapDrawCircleF.argtypes = [HFImageBitmap, HPoint2f, HInt32, HColor, HInt32]
    HFImageBitmapDrawCircleF.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 291
if _libs[_LIBRARY_FILENAME].has("HFImageBitmapDrawCircle", "cdecl"):
    HFImageBitmapDrawCircle = _libs[_LIBRARY_FILENAME].get("HFImageBitmapDrawCircle", "cdecl")
    HFImageBitmapDrawCircle.argtypes = [HFImageBitmap, HPoint2i, HInt32, HColor, HInt32]
    HFImageBitmapDrawCircle.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 300
if _libs[_LIBRARY_FILENAME].has("HFImageBitmapGetData", "cdecl"):
    HFImageBitmapGetData = _libs[_LIBRARY_FILENAME].get("HFImageBitmapGetData", "cdecl")
    HFImageBitmapGetData.argtypes = [HFImageBitmap, PHFImageBitmapData]
    HFImageBitmapGetData.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 310
if _libs[_LIBRARY_FILENAME].has("HFImageBitmapShow", "cdecl"):
    HFImageBitmapShow = _libs[_LIBRARY_FILENAME].get("HFImageBitmapShow", "cdecl")
    HFImageBitmapShow.argtypes = [HFImageBitmap, HString, HInt32]
    HFImageBitmapShow.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 327
class struct_HFResourcePackInfo(Structure):
    pass

struct_HFResourcePackInfo.__slots__ = [
    'structSize',
    'structVersion',
    'archiveFileCount',
    'modelCount',
    'tag',
    'version',
    'major',
    'releaseDate',
    'reserved',
]
struct_HFResourcePackInfo._fields_ = [
    ('structSize', HFUInt32),
    ('structVersion', HFUInt32),
    ('archiveFileCount', HFUInt32),
    ('modelCount', HFUInt32),
    ('tag', HChar * int(64)),
    ('version', HChar * int(64)),
    ('major', HChar * int(64)),
    ('releaseDate', HChar * int(64)),
    ('reserved', HFUInt64 * int(8)),
]

HFResourcePackInfo = struct_HFResourcePackInfo
PHFResourcePackInfo = POINTER(struct_HFResourcePackInfo)

if _libs[_LIBRARY_FILENAME].has("HFValidateResourcePack", "cdecl"):
    HFValidateResourcePack = _libs[_LIBRARY_FILENAME].get("HFValidateResourcePack", "cdecl")
    HFValidateResourcePack.argtypes = [HPath, PHFResourcePackInfo]
    HFValidateResourcePack.restype = HFStatus

if _libs[_LIBRARY_FILENAME].has("HFLaunchInspireFace", "cdecl"):
    HFLaunchInspireFace = _libs[_LIBRARY_FILENAME].get("HFLaunchInspireFace", "cdecl")
    HFLaunchInspireFace.argtypes = [HPath]
    HFLaunchInspireFace.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 335
if _libs[_LIBRARY_FILENAME].has("HFReloadInspireFace", "cdecl"):
    HFReloadInspireFace = _libs[_LIBRARY_FILENAME].get("HFReloadInspireFace", "cdecl")
    HFReloadInspireFace.argtypes = [HPath]
    HFReloadInspireFace.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 343
if _libs[_LIBRARY_FILENAME].has("HFTerminateInspireFace", "cdecl"):
    HFTerminateInspireFace = _libs[_LIBRARY_FILENAME].get("HFTerminateInspireFace", "cdecl")
    HFTerminateInspireFace.argtypes = []
    HFTerminateInspireFace.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 351
if _libs[_LIBRARY_FILENAME].has("HFQueryInspireFaceLaunchStatus", "cdecl"):
    HFQueryInspireFaceLaunchStatus = _libs[_LIBRARY_FILENAME].get("HFQueryInspireFaceLaunchStatus", "cdecl")
    HFQueryInspireFaceLaunchStatus.argtypes = [HPInt32]
    HFQueryInspireFaceLaunchStatus.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 367
if _libs[_LIBRARY_FILENAME].has("HFQueryExpansiveHardwareRGACompileOption", "cdecl"):
    HFQueryExpansiveHardwareRGACompileOption = _libs[_LIBRARY_FILENAME].get("HFQueryExpansiveHardwareRGACompileOption", "cdecl")
    HFQueryExpansiveHardwareRGACompileOption.argtypes = [HPInt32]
    HFQueryExpansiveHardwareRGACompileOption.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 376
if _libs[_LIBRARY_FILENAME].has("HFSetExpansiveHardwareRockchipDmaHeapPath", "cdecl"):
    HFSetExpansiveHardwareRockchipDmaHeapPath = _libs[_LIBRARY_FILENAME].get("HFSetExpansiveHardwareRockchipDmaHeapPath", "cdecl")
    HFSetExpansiveHardwareRockchipDmaHeapPath.argtypes = [HPath]
    HFSetExpansiveHardwareRockchipDmaHeapPath.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 384
if _libs[_LIBRARY_FILENAME].has("HFQueryExpansiveHardwareRockchipDmaHeapPath", "cdecl"):
    HFQueryExpansiveHardwareRockchipDmaHeapPath = _libs[_LIBRARY_FILENAME].get("HFQueryExpansiveHardwareRockchipDmaHeapPath", "cdecl")
    HFQueryExpansiveHardwareRockchipDmaHeapPath.argtypes = [HString]
    HFQueryExpansiveHardwareRockchipDmaHeapPath.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize", "cdecl"):
    HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize = _libs[_LIBRARY_FILENAME].get("HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize", "cdecl")
    HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize.argtypes = [HString, HInt32]
    HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize.restype = HResult

enum_HFImageProcessingBackend = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 392

HF_IMAGE_PROCESSING_CPU = 0# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 392

HF_IMAGE_PROCESSING_RGA = 1# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 392

HFImageProcessingBackend = enum_HFImageProcessingBackend# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 392

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 399
if _libs[_LIBRARY_FILENAME].has("HFSwitchImageProcessingBackend", "cdecl"):
    HFSwitchImageProcessingBackend = _libs[_LIBRARY_FILENAME].get("HFSwitchImageProcessingBackend", "cdecl")
    HFSwitchImageProcessingBackend.argtypes = [HFImageProcessingBackend]
    HFSwitchImageProcessingBackend.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 406
if _libs[_LIBRARY_FILENAME].has("HFSetImageProcessAlignedWidth", "cdecl"):
    HFSetImageProcessAlignedWidth = _libs[_LIBRARY_FILENAME].get("HFSetImageProcessAlignedWidth", "cdecl")
    HFSetImageProcessAlignedWidth.argtypes = [HInt32]
    HFSetImageProcessAlignedWidth.restype = HResult

enum_HFAppleCoreMLInferenceMode = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 415

HF_APPLE_COREML_INFERENCE_MODE_CPU = 0# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 415

HF_APPLE_COREML_INFERENCE_MODE_GPU = 1# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 415

HF_APPLE_COREML_INFERENCE_MODE_ANE = 2# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 415

HFAppleCoreMLInferenceMode = enum_HFAppleCoreMLInferenceMode# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 415

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 422
if _libs[_LIBRARY_FILENAME].has("HFSetAppleCoreMLInferenceMode", "cdecl"):
    HFSetAppleCoreMLInferenceMode = _libs[_LIBRARY_FILENAME].get("HFSetAppleCoreMLInferenceMode", "cdecl")
    HFSetAppleCoreMLInferenceMode.argtypes = [HFAppleCoreMLInferenceMode]
    HFSetAppleCoreMLInferenceMode.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 429
if _libs[_LIBRARY_FILENAME].has("HFSetCudaDeviceId", "cdecl"):
    HFSetCudaDeviceId = _libs[_LIBRARY_FILENAME].get("HFSetCudaDeviceId", "cdecl")
    HFSetCudaDeviceId.argtypes = [HInt32]
    HFSetCudaDeviceId.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 436
if _libs[_LIBRARY_FILENAME].has("HFGetCudaDeviceId", "cdecl"):
    HFGetCudaDeviceId = _libs[_LIBRARY_FILENAME].get("HFGetCudaDeviceId", "cdecl")
    HFGetCudaDeviceId.argtypes = [HPInt32]
    HFGetCudaDeviceId.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 442
if _libs[_LIBRARY_FILENAME].has("HFPrintCudaDeviceInfo", "cdecl"):
    HFPrintCudaDeviceInfo = _libs[_LIBRARY_FILENAME].get("HFPrintCudaDeviceInfo", "cdecl")
    HFPrintCudaDeviceInfo.argtypes = []
    HFPrintCudaDeviceInfo.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 449
if _libs[_LIBRARY_FILENAME].has("HFGetNumCudaDevices", "cdecl"):
    HFGetNumCudaDevices = _libs[_LIBRARY_FILENAME].get("HFGetNumCudaDevices", "cdecl")
    HFGetNumCudaDevices.argtypes = [HPInt32]
    HFGetNumCudaDevices.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 456
if _libs[_LIBRARY_FILENAME].has("HFCheckCudaDeviceSupport", "cdecl"):
    HFCheckCudaDeviceSupport = _libs[_LIBRARY_FILENAME].get("HFCheckCudaDeviceSupport", "cdecl")
    HFCheckCudaDeviceSupport.argtypes = [HPInt32]
    HFCheckCudaDeviceSupport.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 485
class struct_HFSessionCustomParameter(Structure):
    pass

struct_HFSessionCustomParameter.__slots__ = [
    'enable_recognition',
    'enable_liveness',
    'enable_ir_liveness',
    'enable_mask_detect',
    'enable_face_quality',
    'enable_face_attribute',
    'enable_interaction_liveness',
    'enable_detect_mode_landmark',
    'enable_face_pose',
    'enable_face_emotion',
]
struct_HFSessionCustomParameter._fields_ = [
    ('enable_recognition', HInt32),
    ('enable_liveness', HInt32),
    ('enable_ir_liveness', HInt32),
    ('enable_mask_detect', HInt32),
    ('enable_face_quality', HInt32),
    ('enable_face_attribute', HInt32),
    ('enable_interaction_liveness', HInt32),
    ('enable_detect_mode_landmark', HInt32),
    ('enable_face_pose', HInt32),
    ('enable_face_emotion', HInt32),
]

HFSessionCustomParameter = struct_HFSessionCustomParameter# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 485

PHFSessionCustomParameter = POINTER(struct_HFSessionCustomParameter)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 485

enum_HFDetectMode = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 498

HF_DETECT_MODE_ALWAYS_DETECT = 0# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 498

HF_DETECT_MODE_LIGHT_TRACK = (HF_DETECT_MODE_ALWAYS_DETECT + 1)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 498

HF_DETECT_MODE_TRACK_BY_DETECTION = (HF_DETECT_MODE_LIGHT_TRACK + 1)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 498

HFDetectMode = enum_HFDetectMode# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 498

class struct_HFSessionConfigV2(Structure):
    pass

struct_HFSessionConfigV2.__slots__ = [
    'structSize',
    'structVersion',
    'featureMask',
    'detectMode',
    'maxDetectFaceNum',
    'detectPixelLevel',
    'trackByDetectModeFPS',
    'reserved',
]
struct_HFSessionConfigV2._fields_ = [
    ('structSize', HFUInt32),
    ('structVersion', HFUInt32),
    ('featureMask', HFUInt64),
    ('detectMode', HInt32),
    ('maxDetectFaceNum', HInt32),
    ('detectPixelLevel', HInt32),
    ('trackByDetectModeFPS', HInt32),
    ('reserved', HFUInt32 * int(8)),
]

HFSessionConfigV2 = struct_HFSessionConfigV2

PHFSessionConfigV2 = POINTER(struct_HFSessionConfigV2)

enum_HFSessionLandmarkEngine = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 507

HF_LANDMARK_HYPLMV2_0_25 = 0# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 507

HF_LANDMARK_HYPLMV2_0_50 = 1# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 507

HF_LANDMARK_INSIGHTFACE_2D106_TRACK = 2# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 507

HFSessionLandmarkEngine = enum_HFSessionLandmarkEngine# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 507

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 515
if _libs[_LIBRARY_FILENAME].has("HFSwitchLandmarkEngine", "cdecl"):
    HFSwitchLandmarkEngine = _libs[_LIBRARY_FILENAME].get("HFSwitchLandmarkEngine", "cdecl")
    HFSwitchLandmarkEngine.argtypes = [HFSessionLandmarkEngine]
    HFSwitchLandmarkEngine.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 523
class struct_HFFaceDetectPixelList(Structure):
    pass

struct_HFFaceDetectPixelList.__slots__ = [
    'pixel_level',
    'size',
]
struct_HFFaceDetectPixelList._fields_ = [
    ('pixel_level', HInt32 * int(20)),
    ('size', HInt32),
]

HFFaceDetectPixelList = struct_HFFaceDetectPixelList# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 523

PHFFaceDetectPixelList = POINTER(struct_HFFaceDetectPixelList)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 523

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 530
if _libs[_LIBRARY_FILENAME].has("HFQuerySupportedPixelLevelsForFaceDetection", "cdecl"):
    HFQuerySupportedPixelLevelsForFaceDetection = _libs[_LIBRARY_FILENAME].get("HFQuerySupportedPixelLevelsForFaceDetection", "cdecl")
    HFQuerySupportedPixelLevelsForFaceDetection.argtypes = [PHFFaceDetectPixelList]
    HFQuerySupportedPixelLevelsForFaceDetection.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 546
if _libs[_LIBRARY_FILENAME].has("HFCreateInspireFaceSession", "cdecl"):
    HFCreateInspireFaceSession = _libs[_LIBRARY_FILENAME].get("HFCreateInspireFaceSession", "cdecl")
    HFCreateInspireFaceSession.argtypes = [HFSessionCustomParameter, HFDetectMode, HInt32, HInt32, HInt32, PHFSession]
    HFCreateInspireFaceSession.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 563
if _libs[_LIBRARY_FILENAME].has("HFCreateInspireFaceSessionOptional", "cdecl"):
    HFCreateInspireFaceSessionOptional = _libs[_LIBRARY_FILENAME].get("HFCreateInspireFaceSessionOptional", "cdecl")
    HFCreateInspireFaceSessionOptional.argtypes = [HOption, HFDetectMode, HInt32, HInt32, HInt32, PHFSession]
    HFCreateInspireFaceSessionOptional.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFCreateInspireFaceSessionV2", "cdecl"):
    HFCreateInspireFaceSessionV2 = _libs[_LIBRARY_FILENAME].get("HFCreateInspireFaceSessionV2", "cdecl")
    HFCreateInspireFaceSessionV2.argtypes = [PHFSessionConfigV2, PHFSession]
    HFCreateInspireFaceSessionV2.restype = HFStatus

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 572
if _libs[_LIBRARY_FILENAME].has("HFReleaseInspireFaceSession", "cdecl"):
    HFReleaseInspireFaceSession = _libs[_LIBRARY_FILENAME].get("HFReleaseInspireFaceSession", "cdecl")
    HFReleaseInspireFaceSession.argtypes = [HFSession]
    HFReleaseInspireFaceSession.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 589
class struct_HFFaceBasicToken(Structure):
    pass

struct_HFFaceBasicToken.__slots__ = [
    'size',
    'data',
]
struct_HFFaceBasicToken._fields_ = [
    ('size', HInt32),
    ('data', HPVoid),
]

HFFaceBasicToken = struct_HFFaceBasicToken# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 589

PHFFaceBasicToken = POINTER(struct_HFFaceBasicToken)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 589

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 600
class struct_HFFaceEulerAngle(Structure):
    pass

struct_HFFaceEulerAngle.__slots__ = [
    'roll',
    'yaw',
    'pitch',
]
struct_HFFaceEulerAngle._fields_ = [
    ('roll', HPFloat),
    ('yaw', HPFloat),
    ('pitch', HPFloat),
]

HFFaceEulerAngle = struct_HFFaceEulerAngle# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 600

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 616
class struct_HFMultipleFaceData(Structure):
    pass

struct_HFMultipleFaceData.__slots__ = [
    'detectedNum',
    'rects',
    'trackIds',
    'trackCounts',
    'detConfidence',
    'angles',
    'tokens',
]
struct_HFMultipleFaceData._fields_ = [
    ('detectedNum', HInt32),
    ('rects', PHFaceRect),
    ('trackIds', HPInt32),
    ('trackCounts', HPInt32),
    ('detConfidence', HPFloat),
    ('angles', HFFaceEulerAngle),
    ('tokens', PHFFaceBasicToken),
]

HFMultipleFaceData = struct_HFMultipleFaceData# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 616

PHFMultipleFaceData = POINTER(struct_HFMultipleFaceData)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 616

HFFaceCaptureState = c_int


class struct_HFFaceCaptureConfig(Structure):
    pass


struct_HFFaceCaptureConfig.__slots__ = [
    'structSize',
    'structVersion',
    'filterMask',
    'outputCount',
    'minTrackCount',
    'stableDurationMs',
    'collectDurationMs',
    'maxCollectDurationMs',
    'trackLostGraceMs',
    'minCandidateIntervalMs',
    'minFaceWidthRatio',
    'maxFaceWidthRatio',
    'maxCenterOffsetX',
    'maxCenterOffsetY',
    'boundaryMarginRatio',
    'maxCenterMotionRatio',
    'maxSizeChangeRatio',
    'maxAbsYaw',
    'maxAbsPitch',
    'maxAbsRoll',
    'minQualityScore',
    'minSharpnessScore',
    'minBrightnessScore',
    'maxBrightnessScore',
    'reserved',
]
struct_HFFaceCaptureConfig._fields_ = [
    ('structSize', HFUInt32),
    ('structVersion', HFUInt32),
    ('filterMask', HFUInt64),
    ('outputCount', HFUInt32),
    ('minTrackCount', HFUInt32),
    ('stableDurationMs', HFUInt64),
    ('collectDurationMs', HFUInt64),
    ('maxCollectDurationMs', HFUInt64),
    ('trackLostGraceMs', HFUInt64),
    ('minCandidateIntervalMs', HFUInt64),
    ('minFaceWidthRatio', HFloat),
    ('maxFaceWidthRatio', HFloat),
    ('maxCenterOffsetX', HFloat),
    ('maxCenterOffsetY', HFloat),
    ('boundaryMarginRatio', HFloat),
    ('maxCenterMotionRatio', HFloat),
    ('maxSizeChangeRatio', HFloat),
    ('maxAbsYaw', HFloat),
    ('maxAbsPitch', HFloat),
    ('maxAbsRoll', HFloat),
    ('minQualityScore', HFloat),
    ('minSharpnessScore', HFloat),
    ('minBrightnessScore', HFloat),
    ('maxBrightnessScore', HFloat),
    ('reserved', HFUInt32 * int(8)),
]

HFFaceCaptureConfig = struct_HFFaceCaptureConfig
PHFFaceCaptureConfig = POINTER(struct_HFFaceCaptureConfig)


class struct_HFFaceCaptureMetrics(Structure):
    pass


struct_HFFaceCaptureMetrics.__slots__ = [
    'availableMetrics',
    'faceWidthRatio',
    'centerOffsetX',
    'centerOffsetY',
    'stabilityScore',
    'poseScore',
    'qualityScore',
    'sharpnessScore',
    'brightnessScore',
]
struct_HFFaceCaptureMetrics._fields_ = [
    ('availableMetrics', HFUInt64),
    ('faceWidthRatio', HFloat),
    ('centerOffsetX', HFloat),
    ('centerOffsetY', HFloat),
    ('stabilityScore', HFloat),
    ('poseScore', HFloat),
    ('qualityScore', HFloat),
    ('sharpnessScore', HFloat),
    ('brightnessScore', HFloat),
]

HFFaceCaptureMetrics = struct_HFFaceCaptureMetrics
PHFFaceCaptureMetrics = POINTER(struct_HFFaceCaptureMetrics)


class struct_HFFaceCaptureProgress(Structure):
    pass


struct_HFFaceCaptureProgress.__slots__ = [
    'state',
    'candidateCount',
    'frameId',
    'timestampMs',
    'trackId',
    'trackCount',
    'evaluatedFilters',
    'rejectReasons',
    'progress',
    'currentScore',
    'metrics',
]
struct_HFFaceCaptureProgress._fields_ = [
    ('state', HInt32),
    ('candidateCount', HFUInt32),
    ('frameId', HFUInt64),
    ('timestampMs', HFUInt64),
    ('trackId', HInt32),
    ('trackCount', HInt32),
    ('evaluatedFilters', HFUInt64),
    ('rejectReasons', HFUInt64),
    ('progress', HFloat),
    ('currentScore', HFloat),
    ('metrics', HFFaceCaptureMetrics),
]

HFFaceCaptureProgress = struct_HFFaceCaptureProgress
PHFFaceCaptureProgress = POINTER(struct_HFFaceCaptureProgress)


class struct_HFFaceCaptureResult(Structure):
    pass


struct_HFFaceCaptureResult.__slots__ = [
    'frameId',
    'timestampMs',
    'trackId',
    'trackCount',
    'score',
    'rect',
    'roll',
    'yaw',
    'pitch',
    'token',
    'metrics',
]
struct_HFFaceCaptureResult._fields_ = [
    ('frameId', HFUInt64),
    ('timestampMs', HFUInt64),
    ('trackId', HInt32),
    ('trackCount', HInt32),
    ('score', HFloat),
    ('rect', HFaceRect),
    ('roll', HFloat),
    ('yaw', HFloat),
    ('pitch', HFloat),
    ('token', HFFaceBasicToken),
    ('metrics', HFFaceCaptureMetrics),
]

HFFaceCaptureResult = struct_HFFaceCaptureResult
PHFFaceCaptureResult = POINTER(struct_HFFaceCaptureResult)

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 623
if _libs[_LIBRARY_FILENAME].has("HFSessionClearTrackingFace", "cdecl"):
    HFSessionClearTrackingFace = _libs[_LIBRARY_FILENAME].get("HFSessionClearTrackingFace", "cdecl")
    HFSessionClearTrackingFace.argtypes = [HFSession]
    HFSessionClearTrackingFace.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 631
if _libs[_LIBRARY_FILENAME].has("HFSessionSetTrackLostRecoveryMode", "cdecl"):
    HFSessionSetTrackLostRecoveryMode = _libs[_LIBRARY_FILENAME].get("HFSessionSetTrackLostRecoveryMode", "cdecl")
    HFSessionSetTrackLostRecoveryMode.argtypes = [HFSession, HInt32]
    HFSessionSetTrackLostRecoveryMode.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 639
if _libs[_LIBRARY_FILENAME].has("HFSessionSetLightTrackConfidenceThreshold", "cdecl"):
    HFSessionSetLightTrackConfidenceThreshold = _libs[_LIBRARY_FILENAME].get("HFSessionSetLightTrackConfidenceThreshold", "cdecl")
    HFSessionSetLightTrackConfidenceThreshold.argtypes = [HFSession, HFloat]
    HFSessionSetLightTrackConfidenceThreshold.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 649
if _libs[_LIBRARY_FILENAME].has("HFSessionSetTrackPreviewSize", "cdecl"):
    HFSessionSetTrackPreviewSize = _libs[_LIBRARY_FILENAME].get("HFSessionSetTrackPreviewSize", "cdecl")
    HFSessionSetTrackPreviewSize.argtypes = [HFSession, HInt32]
    HFSessionSetTrackPreviewSize.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 657
if _libs[_LIBRARY_FILENAME].has("HFSessionGetTrackPreviewSize", "cdecl"):
    HFSessionGetTrackPreviewSize = _libs[_LIBRARY_FILENAME].get("HFSessionGetTrackPreviewSize", "cdecl")
    HFSessionGetTrackPreviewSize.argtypes = [HFSession, HPInt32]
    HFSessionGetTrackPreviewSize.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 667
if _libs[_LIBRARY_FILENAME].has("HFSessionSetFilterMinimumFacePixelSize", "cdecl"):
    HFSessionSetFilterMinimumFacePixelSize = _libs[_LIBRARY_FILENAME].get("HFSessionSetFilterMinimumFacePixelSize", "cdecl")
    HFSessionSetFilterMinimumFacePixelSize.argtypes = [HFSession, HInt32]
    HFSessionSetFilterMinimumFacePixelSize.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 676
if _libs[_LIBRARY_FILENAME].has("HFSessionSetFaceDetectThreshold", "cdecl"):
    HFSessionSetFaceDetectThreshold = _libs[_LIBRARY_FILENAME].get("HFSessionSetFaceDetectThreshold", "cdecl")
    HFSessionSetFaceDetectThreshold.argtypes = [HFSession, HFloat]
    HFSessionSetFaceDetectThreshold.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 685
if _libs[_LIBRARY_FILENAME].has("HFSessionSetTrackModeSmoothRatio", "cdecl"):
    HFSessionSetTrackModeSmoothRatio = _libs[_LIBRARY_FILENAME].get("HFSessionSetTrackModeSmoothRatio", "cdecl")
    HFSessionSetTrackModeSmoothRatio.argtypes = [HFSession, HFloat]
    HFSessionSetTrackModeSmoothRatio.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 694
if _libs[_LIBRARY_FILENAME].has("HFSessionSetTrackModeNumSmoothCacheFrame", "cdecl"):
    HFSessionSetTrackModeNumSmoothCacheFrame = _libs[_LIBRARY_FILENAME].get("HFSessionSetTrackModeNumSmoothCacheFrame", "cdecl")
    HFSessionSetTrackModeNumSmoothCacheFrame.argtypes = [HFSession, HInt32]
    HFSessionSetTrackModeNumSmoothCacheFrame.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 703
if _libs[_LIBRARY_FILENAME].has("HFSessionSetTrackModeDetectInterval", "cdecl"):
    HFSessionSetTrackModeDetectInterval = _libs[_LIBRARY_FILENAME].get("HFSessionSetTrackModeDetectInterval", "cdecl")
    HFSessionSetTrackModeDetectInterval.argtypes = [HFSession, HInt32]
    HFSessionSetTrackModeDetectInterval.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h
if _libs[_LIBRARY_FILENAME].has("HFSessionSetLandmarkAugmentationNum", "cdecl"):
    HFSessionSetLandmarkAugmentationNum = _libs[_LIBRARY_FILENAME].get("HFSessionSetLandmarkAugmentationNum", "cdecl")
    HFSessionSetLandmarkAugmentationNum.argtypes = [HFSession, HInt32]
    HFSessionSetLandmarkAugmentationNum.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 713
if _libs[_LIBRARY_FILENAME].has("HFExecuteFaceTrack", "cdecl"):
    HFExecuteFaceTrack = _libs[_LIBRARY_FILENAME].get("HFExecuteFaceTrack", "cdecl")
    HFExecuteFaceTrack.argtypes = [HFSession, HFImageStream, PHFMultipleFaceData]
    HFExecuteFaceTrack.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFExecuteFaceTrackSnapshot", "cdecl"):
    HFExecuteFaceTrackSnapshot = _libs[_LIBRARY_FILENAME].get("HFExecuteFaceTrackSnapshot", "cdecl")
    HFExecuteFaceTrackSnapshot.argtypes = [HFSession, HFImageStream, PHFFaceResultSnapshot]
    HFExecuteFaceTrackSnapshot.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFGetFaceResultSnapshotData", "cdecl"):
    HFGetFaceResultSnapshotData = _libs[_LIBRARY_FILENAME].get("HFGetFaceResultSnapshotData", "cdecl")
    HFGetFaceResultSnapshotData.argtypes = [HFFaceResultSnapshot, PHFMultipleFaceData]
    HFGetFaceResultSnapshotData.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFReleaseFaceResultSnapshot", "cdecl"):
    HFReleaseFaceResultSnapshot = _libs[_LIBRARY_FILENAME].get("HFReleaseFaceResultSnapshot", "cdecl")
    HFReleaseFaceResultSnapshot.argtypes = [HFFaceResultSnapshot]
    HFReleaseFaceResultSnapshot.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFGetDefaultFaceCaptureConfig", "cdecl"):
    HFGetDefaultFaceCaptureConfig = _libs[_LIBRARY_FILENAME].get("HFGetDefaultFaceCaptureConfig", "cdecl")
    HFGetDefaultFaceCaptureConfig.argtypes = [PHFFaceCaptureConfig]
    HFGetDefaultFaceCaptureConfig.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFCreateFaceCaptureSession", "cdecl"):
    HFCreateFaceCaptureSession = _libs[_LIBRARY_FILENAME].get("HFCreateFaceCaptureSession", "cdecl")
    HFCreateFaceCaptureSession.argtypes = [HFSession, PHFFaceCaptureConfig, PHFFaceCaptureSession]
    HFCreateFaceCaptureSession.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFUpdateFaceCaptureSession", "cdecl"):
    HFUpdateFaceCaptureSession = _libs[_LIBRARY_FILENAME].get("HFUpdateFaceCaptureSession", "cdecl")
    HFUpdateFaceCaptureSession.argtypes = [
        HFFaceCaptureSession,
        HFImageStream,
        HFUInt64,
        HFUInt64,
        PHFFaceCaptureProgress,
    ]
    HFUpdateFaceCaptureSession.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFUpdateFaceCaptureSessionWithSnapshot", "cdecl"):
    HFUpdateFaceCaptureSessionWithSnapshot = _libs[_LIBRARY_FILENAME].get(
        "HFUpdateFaceCaptureSessionWithSnapshot", "cdecl"
    )
    HFUpdateFaceCaptureSessionWithSnapshot.argtypes = [
        HFFaceCaptureSession,
        HFImageStream,
        HFFaceResultSnapshot,
        HFUInt64,
        HFUInt64,
        PHFFaceCaptureProgress,
    ]
    HFUpdateFaceCaptureSessionWithSnapshot.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFGetFaceCaptureResults", "cdecl"):
    HFGetFaceCaptureResults = _libs[_LIBRARY_FILENAME].get("HFGetFaceCaptureResults", "cdecl")
    HFGetFaceCaptureResults.argtypes = [
        HFFaceCaptureSession,
        PHFFaceCaptureResult,
        HFUInt32,
        POINTER(HFUInt32),
    ]
    HFGetFaceCaptureResults.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFFinishFaceCaptureSession", "cdecl"):
    HFFinishFaceCaptureSession = _libs[_LIBRARY_FILENAME].get("HFFinishFaceCaptureSession", "cdecl")
    HFFinishFaceCaptureSession.argtypes = [HFFaceCaptureSession, PHFFaceCaptureProgress]
    HFFinishFaceCaptureSession.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFResetFaceCaptureSession", "cdecl"):
    HFResetFaceCaptureSession = _libs[_LIBRARY_FILENAME].get("HFResetFaceCaptureSession", "cdecl")
    HFResetFaceCaptureSession.argtypes = [HFFaceCaptureSession]
    HFResetFaceCaptureSession.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFReleaseFaceCaptureSession", "cdecl"):
    HFReleaseFaceCaptureSession = _libs[_LIBRARY_FILENAME].get("HFReleaseFaceCaptureSession", "cdecl")
    HFReleaseFaceCaptureSession.argtypes = [HFFaceCaptureSession]
    HFReleaseFaceCaptureSession.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 721
if _libs[_LIBRARY_FILENAME].has("HFSessionLastFaceDetectionGetDebugPreviewImageSize", "cdecl"):
    HFSessionLastFaceDetectionGetDebugPreviewImageSize = _libs[_LIBRARY_FILENAME].get("HFSessionLastFaceDetectionGetDebugPreviewImageSize", "cdecl")
    HFSessionLastFaceDetectionGetDebugPreviewImageSize.argtypes = [HFSession, HPInt32]
    HFSessionLastFaceDetectionGetDebugPreviewImageSize.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 738
if _libs[_LIBRARY_FILENAME].has("HFCopyFaceBasicToken", "cdecl"):
    HFCopyFaceBasicToken = _libs[_LIBRARY_FILENAME].get("HFCopyFaceBasicToken", "cdecl")
    HFCopyFaceBasicToken.argtypes = [HFFaceBasicToken, HPBuffer, HInt32]
    HFCopyFaceBasicToken.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 752
if _libs[_LIBRARY_FILENAME].has("HFGetFaceBasicTokenSize", "cdecl"):
    HFGetFaceBasicTokenSize = _libs[_LIBRARY_FILENAME].get("HFGetFaceBasicTokenSize", "cdecl")
    HFGetFaceBasicTokenSize.argtypes = [HPInt32]
    HFGetFaceBasicTokenSize.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 759
if _libs[_LIBRARY_FILENAME].has("HFGetNumOfFaceDenseLandmark", "cdecl"):
    HFGetNumOfFaceDenseLandmark = _libs[_LIBRARY_FILENAME].get("HFGetNumOfFaceDenseLandmark", "cdecl")
    HFGetNumOfFaceDenseLandmark.argtypes = [HPInt32]
    HFGetNumOfFaceDenseLandmark.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 769
if _libs[_LIBRARY_FILENAME].has("HFGetFaceDenseLandmarkFromFaceToken", "cdecl"):
    HFGetFaceDenseLandmarkFromFaceToken = _libs[_LIBRARY_FILENAME].get("HFGetFaceDenseLandmarkFromFaceToken", "cdecl")
    HFGetFaceDenseLandmarkFromFaceToken.argtypes = [HFFaceBasicToken, PHPoint2f, HInt32]
    HFGetFaceDenseLandmarkFromFaceToken.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 778
if _libs[_LIBRARY_FILENAME].has("HFGetFaceFiveKeyPointsFromFaceToken", "cdecl"):
    HFGetFaceFiveKeyPointsFromFaceToken = _libs[_LIBRARY_FILENAME].get("HFGetFaceFiveKeyPointsFromFaceToken", "cdecl")
    HFGetFaceFiveKeyPointsFromFaceToken.argtypes = [HFFaceBasicToken, PHPoint2f, HInt32]
    HFGetFaceFiveKeyPointsFromFaceToken.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 785
if _libs[_LIBRARY_FILENAME].has("HFSessionSetEnableTrackCostSpend", "cdecl"):
    HFSessionSetEnableTrackCostSpend = _libs[_LIBRARY_FILENAME].get("HFSessionSetEnableTrackCostSpend", "cdecl")
    HFSessionSetEnableTrackCostSpend.argtypes = [HFSession, HInt32]
    HFSessionSetEnableTrackCostSpend.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 792
if _libs[_LIBRARY_FILENAME].has("HFSessionPrintTrackCostSpend", "cdecl"):
    HFSessionPrintTrackCostSpend = _libs[_LIBRARY_FILENAME].get("HFSessionPrintTrackCostSpend", "cdecl")
    HFSessionPrintTrackCostSpend.argtypes = [HFSession]
    HFSessionPrintTrackCostSpend.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 809
class struct_HFFaceFeature(Structure):
    pass

struct_HFFaceFeature.__slots__ = [
    'size',
    'data',
]
struct_HFFaceFeature._fields_ = [
    ('size', HInt32),
    ('data', HPFloat),
]

HFFaceFeature = struct_HFFaceFeature# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 809

PHFFaceFeature = POINTER(struct_HFFaceFeature)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 809

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 820
if _libs[_LIBRARY_FILENAME].has("HFFaceFeatureExtract", "cdecl"):
    HFFaceFeatureExtract = _libs[_LIBRARY_FILENAME].get("HFFaceFeatureExtract", "cdecl")
    HFFaceFeatureExtract.argtypes = [HFSession, HFImageStream, HFFaceBasicToken, PHFFaceFeature]
    HFFaceFeatureExtract.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 831
if _libs[_LIBRARY_FILENAME].has("HFFaceFeatureExtractTo", "cdecl"):
    HFFaceFeatureExtractTo = _libs[_LIBRARY_FILENAME].get("HFFaceFeatureExtractTo", "cdecl")
    HFFaceFeatureExtractTo.argtypes = [HFSession, HFImageStream, HFFaceBasicToken, HFFaceFeature]
    HFFaceFeatureExtractTo.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 843
if _libs[_LIBRARY_FILENAME].has("HFFaceFeatureExtractCpy", "cdecl"):
    HFFaceFeatureExtractCpy = _libs[_LIBRARY_FILENAME].get("HFFaceFeatureExtractCpy", "cdecl")
    HFFaceFeatureExtractCpy.argtypes = [HFSession, HFImageStream, HFFaceBasicToken, HPFloat]
    HFFaceFeatureExtractCpy.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 850
if _libs[_LIBRARY_FILENAME].has("HFCreateFaceFeature", "cdecl"):
    HFCreateFaceFeature = _libs[_LIBRARY_FILENAME].get("HFCreateFaceFeature", "cdecl")
    HFCreateFaceFeature.argtypes = [PHFFaceFeature]
    HFCreateFaceFeature.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 857
if _libs[_LIBRARY_FILENAME].has("HFReleaseFaceFeature", "cdecl"):
    HFReleaseFaceFeature = _libs[_LIBRARY_FILENAME].get("HFReleaseFaceFeature", "cdecl")
    HFReleaseFaceFeature.argtypes = [PHFFaceFeature]
    HFReleaseFaceFeature.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 867
if _libs[_LIBRARY_FILENAME].has("HFFaceGetFaceAlignmentImage", "cdecl"):
    HFFaceGetFaceAlignmentImage = _libs[_LIBRARY_FILENAME].get("HFFaceGetFaceAlignmentImage", "cdecl")
    HFFaceGetFaceAlignmentImage.argtypes = [HFSession, HFImageStream, HFFaceBasicToken, PHFImageBitmap]
    HFFaceGetFaceAlignmentImage.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 877
if _libs[_LIBRARY_FILENAME].has("HFFaceFeatureExtractWithAlignmentImage", "cdecl"):
    HFFaceFeatureExtractWithAlignmentImage = _libs[_LIBRARY_FILENAME].get("HFFaceFeatureExtractWithAlignmentImage", "cdecl")
    HFFaceFeatureExtractWithAlignmentImage.argtypes = [HFSession, HFImageStream, HFFaceFeature]
    HFFaceFeatureExtractWithAlignmentImage.restype = HResult

enum_HFSearchMode = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 895

HF_SEARCH_MODE_EAGER = 0# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 895

HF_SEARCH_MODE_EXHAUSTIVE = (HF_SEARCH_MODE_EAGER + 1)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 895

HFSearchMode = enum_HFSearchMode# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 895

enum_HFPKMode = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 903

HF_PK_AUTO_INCREMENT = 0# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 903

HF_PK_MANUAL_INPUT = (HF_PK_AUTO_INCREMENT + 1)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 903

HFPKMode = enum_HFPKMode# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 903

HF_INVALID_FACE_ID = -1

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 916
class struct_HFFeatureHubConfiguration(Structure):
    pass

struct_HFFeatureHubConfiguration.__slots__ = [
    'primaryKeyMode',
    'enablePersistence',
    'persistenceDbPath',
    'searchThreshold',
    'searchMode',
]
struct_HFFeatureHubConfiguration._fields_ = [
    ('primaryKeyMode', HFPKMode),
    ('enablePersistence', HInt32),
    ('persistenceDbPath', HString),
    ('searchThreshold', HFloat),
    ('searchMode', HFSearchMode),
]

HFFeatureHubConfiguration = struct_HFFeatureHubConfiguration# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 916

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 928
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubDataEnable", "cdecl"):
    HFFeatureHubDataEnable = _libs[_LIBRARY_FILENAME].get("HFFeatureHubDataEnable", "cdecl")
    HFFeatureHubDataEnable.argtypes = [HFFeatureHubConfiguration]
    HFFeatureHubDataEnable.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 934
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubDataDisable", "cdecl"):
    HFFeatureHubDataDisable = _libs[_LIBRARY_FILENAME].get("HFFeatureHubDataDisable", "cdecl")
    HFFeatureHubDataDisable.argtypes = []
    HFFeatureHubDataDisable.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 945
class struct_HFFaceFeatureIdentity(Structure):
    pass

struct_HFFaceFeatureIdentity.__slots__ = [
    'id',
    'feature',
]
struct_HFFaceFeatureIdentity._fields_ = [
    ('id', HFaceId),
    ('feature', PHFFaceFeature),
]

HFFaceFeatureIdentity = struct_HFFaceFeatureIdentity# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 945

PHFFaceFeatureIdentity = POINTER(struct_HFFaceFeatureIdentity)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 945

class struct_HFFeatureHubSearchResultV2(Structure):
    pass

struct_HFFeatureHubSearchResultV2.__slots__ = [
    'found',
    'id',
    'confidence',
    'feature',
]
struct_HFFeatureHubSearchResultV2._fields_ = [
    ('found', HInt32),
    ('id', HFaceId),
    ('confidence', HFloat),
    ('feature', HFFaceFeature),
]

HFFeatureHubSearchResultV2 = struct_HFFeatureHubSearchResultV2
PHFFeatureHubSearchResultV2 = POINTER(struct_HFFeatureHubSearchResultV2)

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 954
class struct_HFSearchTopKResults(Structure):
    pass

struct_HFSearchTopKResults.__slots__ = [
    'size',
    'confidence',
    'ids',
]
struct_HFSearchTopKResults._fields_ = [
    ('size', HInt32),
    ('confidence', HPFloat),
    ('ids', HPFaceId),
]

HFSearchTopKResults = struct_HFSearchTopKResults# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 954

PHFSearchTopKResults = POINTER(struct_HFSearchTopKResults)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 954

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 966
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubFaceSearchThresholdSetting", "cdecl"):
    HFFeatureHubFaceSearchThresholdSetting = _libs[_LIBRARY_FILENAME].get("HFFeatureHubFaceSearchThresholdSetting", "cdecl")
    HFFeatureHubFaceSearchThresholdSetting.argtypes = [HFloat]
    HFFeatureHubFaceSearchThresholdSetting.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 981
if _libs[_LIBRARY_FILENAME].has("HFFaceComparison", "cdecl"):
    HFFaceComparison = _libs[_LIBRARY_FILENAME].get("HFFaceComparison", "cdecl")
    HFFaceComparison.argtypes = [HFFaceFeature, HFFaceFeature, HPFloat]
    HFFaceComparison.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 988
if _libs[_LIBRARY_FILENAME].has("HFGetRecommendedCosineThreshold", "cdecl"):
    HFGetRecommendedCosineThreshold = _libs[_LIBRARY_FILENAME].get("HFGetRecommendedCosineThreshold", "cdecl")
    HFGetRecommendedCosineThreshold.argtypes = [HPFloat]
    HFGetRecommendedCosineThreshold.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1001
if _libs[_LIBRARY_FILENAME].has("HFCosineSimilarityConvertToPercentage", "cdecl"):
    HFCosineSimilarityConvertToPercentage = _libs[_LIBRARY_FILENAME].get("HFCosineSimilarityConvertToPercentage", "cdecl")
    HFCosineSimilarityConvertToPercentage.argtypes = [HFloat, HPFloat]
    HFCosineSimilarityConvertToPercentage.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1014
class struct_HFSimilarityConverterConfig(Structure):
    pass

struct_HFSimilarityConverterConfig.__slots__ = [
    'threshold',
    'middleScore',
    'steepness',
    'outputMin',
    'outputMax',
]
struct_HFSimilarityConverterConfig._fields_ = [
    ('threshold', HFloat),
    ('middleScore', HFloat),
    ('steepness', HFloat),
    ('outputMin', HFloat),
    ('outputMax', HFloat),
]

HFSimilarityConverterConfig = struct_HFSimilarityConverterConfig# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1014

PHFSimilarityConverterConfig = POINTER(struct_HFSimilarityConverterConfig)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1014

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1023
if _libs[_LIBRARY_FILENAME].has("HFUpdateCosineSimilarityConverter", "cdecl"):
    HFUpdateCosineSimilarityConverter = _libs[_LIBRARY_FILENAME].get("HFUpdateCosineSimilarityConverter", "cdecl")
    HFUpdateCosineSimilarityConverter.argtypes = [HFSimilarityConverterConfig]
    HFUpdateCosineSimilarityConverter.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1030
if _libs[_LIBRARY_FILENAME].has("HFGetCosineSimilarityConverter", "cdecl"):
    HFGetCosineSimilarityConverter = _libs[_LIBRARY_FILENAME].get("HFGetCosineSimilarityConverter", "cdecl")
    HFGetCosineSimilarityConverter.argtypes = [PHFSimilarityConverterConfig]
    HFGetCosineSimilarityConverter.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1038
if _libs[_LIBRARY_FILENAME].has("HFGetFeatureLength", "cdecl"):
    HFGetFeatureLength = _libs[_LIBRARY_FILENAME].get("HFGetFeatureLength", "cdecl")
    HFGetFeatureLength.argtypes = [HPInt32]
    HFGetFeatureLength.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1046
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubInsertFeature", "cdecl"):
    HFFeatureHubInsertFeature = _libs[_LIBRARY_FILENAME].get("HFFeatureHubInsertFeature", "cdecl")
    HFFeatureHubInsertFeature.argtypes = [HFFaceFeatureIdentity, HPFaceId]
    HFFeatureHubInsertFeature.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1057
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubFaceSearch", "cdecl"):
    HFFeatureHubFaceSearch = _libs[_LIBRARY_FILENAME].get("HFFeatureHubFaceSearch", "cdecl")
    HFFeatureHubFaceSearch.argtypes = [HFFaceFeature, HPFloat, PHFFaceFeatureIdentity]
    HFFeatureHubFaceSearch.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFFeatureHubFaceSearchV2", "cdecl"):
    HFFeatureHubFaceSearchV2 = _libs[_LIBRARY_FILENAME].get("HFFeatureHubFaceSearchV2", "cdecl")
    HFFeatureHubFaceSearchV2.argtypes = [HFFaceFeature, PHFFeatureHubSearchResultV2]
    HFFeatureHubFaceSearchV2.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1067
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubFaceSearchTopK", "cdecl"):
    HFFeatureHubFaceSearchTopK = _libs[_LIBRARY_FILENAME].get("HFFeatureHubFaceSearchTopK", "cdecl")
    HFFeatureHubFaceSearchTopK.argtypes = [HFFaceFeature, HInt32, PHFSearchTopKResults]
    HFFeatureHubFaceSearchTopK.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1075
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubFaceRemove", "cdecl"):
    HFFeatureHubFaceRemove = _libs[_LIBRARY_FILENAME].get("HFFeatureHubFaceRemove", "cdecl")
    HFFeatureHubFaceRemove.argtypes = [HFaceId]
    HFFeatureHubFaceRemove.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1083
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubFaceUpdate", "cdecl"):
    HFFeatureHubFaceUpdate = _libs[_LIBRARY_FILENAME].get("HFFeatureHubFaceUpdate", "cdecl")
    HFFeatureHubFaceUpdate.argtypes = [HFFaceFeatureIdentity]
    HFFeatureHubFaceUpdate.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1092
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubGetFaceIdentity", "cdecl"):
    HFFeatureHubGetFaceIdentity = _libs[_LIBRARY_FILENAME].get("HFFeatureHubGetFaceIdentity", "cdecl")
    HFFeatureHubGetFaceIdentity.argtypes = [HFaceId, PHFFaceFeatureIdentity]
    HFFeatureHubGetFaceIdentity.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1100
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubGetFaceCount", "cdecl"):
    HFFeatureHubGetFaceCount = _libs[_LIBRARY_FILENAME].get("HFFeatureHubGetFaceCount", "cdecl")
    HFFeatureHubGetFaceCount.argtypes = [HPInt32]
    HFFeatureHubGetFaceCount.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1107
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubViewDBTable", "cdecl"):
    HFFeatureHubViewDBTable = _libs[_LIBRARY_FILENAME].get("HFFeatureHubViewDBTable", "cdecl")
    HFFeatureHubViewDBTable.argtypes = []
    HFFeatureHubViewDBTable.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1115
class struct_HFFeatureHubExistingIds(Structure):
    pass

struct_HFFeatureHubExistingIds.__slots__ = [
    'size',
    'ids',
]
struct_HFFeatureHubExistingIds._fields_ = [
    ('size', HInt32),
    ('ids', HPFaceId),
]

HFFeatureHubExistingIds = struct_HFFeatureHubExistingIds# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1115

PHFFeatureHubExistingIds = POINTER(struct_HFFeatureHubExistingIds)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1115

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1122
if _libs[_LIBRARY_FILENAME].has("HFFeatureHubGetExistingIds", "cdecl"):
    HFFeatureHubGetExistingIds = _libs[_LIBRARY_FILENAME].get("HFFeatureHubGetExistingIds", "cdecl")
    HFFeatureHubGetExistingIds.argtypes = [PHFFeatureHubExistingIds]
    HFFeatureHubGetExistingIds.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1143
if _libs[_LIBRARY_FILENAME].has("HFMultipleFacePipelineProcess", "cdecl"):
    HFMultipleFacePipelineProcess = _libs[_LIBRARY_FILENAME].get("HFMultipleFacePipelineProcess", "cdecl")
    HFMultipleFacePipelineProcess.argtypes = [HFSession, HFImageStream, PHFMultipleFaceData, HFSessionCustomParameter]
    HFMultipleFacePipelineProcess.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1158
if _libs[_LIBRARY_FILENAME].has("HFMultipleFacePipelineProcessOptional", "cdecl"):
    HFMultipleFacePipelineProcessOptional = _libs[_LIBRARY_FILENAME].get("HFMultipleFacePipelineProcessOptional", "cdecl")
    HFMultipleFacePipelineProcessOptional.argtypes = [HFSession, HFImageStream, PHFMultipleFaceData, HInt32]
    HFMultipleFacePipelineProcessOptional.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1170
class struct_HFRGBLivenessConfidence(Structure):
    pass

struct_HFRGBLivenessConfidence.__slots__ = [
    'num',
    'confidence',
]
struct_HFRGBLivenessConfidence._fields_ = [
    ('num', HInt32),
    ('confidence', HPFloat),
]

HFRGBLivenessConfidence = struct_HFRGBLivenessConfidence# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1170

PHFRGBLivenessConfidence = POINTER(struct_HFRGBLivenessConfidence)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1170

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1182
if _libs[_LIBRARY_FILENAME].has("HFGetRGBLivenessConfidence", "cdecl"):
    HFGetRGBLivenessConfidence = _libs[_LIBRARY_FILENAME].get("HFGetRGBLivenessConfidence", "cdecl")
    HFGetRGBLivenessConfidence.argtypes = [HFSession, PHFRGBLivenessConfidence]
    HFGetRGBLivenessConfidence.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1193
class struct_HFFaceMaskConfidence(Structure):
    pass

struct_HFFaceMaskConfidence.__slots__ = [
    'num',
    'confidence',
]
struct_HFFaceMaskConfidence._fields_ = [
    ('num', HInt32),
    ('confidence', HPFloat),
]

HFFaceMaskConfidence = struct_HFFaceMaskConfidence# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1193

PHFFaceMaskConfidence = POINTER(struct_HFFaceMaskConfidence)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1193

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1205
if _libs[_LIBRARY_FILENAME].has("HFGetFaceMaskConfidence", "cdecl"):
    HFGetFaceMaskConfidence = _libs[_LIBRARY_FILENAME].get("HFGetFaceMaskConfidence", "cdecl")
    HFGetFaceMaskConfidence.argtypes = [HFSession, PHFFaceMaskConfidence]
    HFGetFaceMaskConfidence.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1216
class struct_HFFaceQualityConfidence(Structure):
    pass

struct_HFFaceQualityConfidence.__slots__ = [
    'num',
    'confidence',
]
struct_HFFaceQualityConfidence._fields_ = [
    ('num', HInt32),
    ('confidence', HPFloat),
]

HFFaceQualityConfidence = struct_HFFaceQualityConfidence# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1216

PHFFaceQualityConfidence = POINTER(struct_HFFaceQualityConfidence)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1216

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1228
if _libs[_LIBRARY_FILENAME].has("HFGetFaceQualityConfidence", "cdecl"):
    HFGetFaceQualityConfidence = _libs[_LIBRARY_FILENAME].get("HFGetFaceQualityConfidence", "cdecl")
    HFGetFaceQualityConfidence.argtypes = [HFSession, PHFFaceQualityConfidence]
    HFGetFaceQualityConfidence.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1240
if _libs[_LIBRARY_FILENAME].has("HFFaceQualityDetect", "cdecl"):
    HFFaceQualityDetect = _libs[_LIBRARY_FILENAME].get("HFFaceQualityDetect", "cdecl")
    HFFaceQualityDetect.argtypes = [HFSession, HFFaceBasicToken, HPFloat]
    HFFaceQualityDetect.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1251
class struct_HFFaceInteractionState(Structure):
    pass

struct_HFFaceInteractionState.__slots__ = [
    'num',
    'leftEyeStatusConfidence',
    'rightEyeStatusConfidence',
]
struct_HFFaceInteractionState._fields_ = [
    ('num', HInt32),
    ('leftEyeStatusConfidence', HPFloat),
    ('rightEyeStatusConfidence', HPFloat),
]

HFFaceInteractionState = struct_HFFaceInteractionState# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1251

PHFFaceInteractionState = POINTER(struct_HFFaceInteractionState)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1251

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1258
if _libs[_LIBRARY_FILENAME].has("HFGetFaceInteractionStateResult", "cdecl"):
    HFGetFaceInteractionStateResult = _libs[_LIBRARY_FILENAME].get("HFGetFaceInteractionStateResult", "cdecl")
    HFGetFaceInteractionStateResult.argtypes = [HFSession, PHFFaceInteractionState]
    HFGetFaceInteractionStateResult.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1270
class struct_HFFaceInteractionsActions(Structure):
    pass

struct_HFFaceInteractionsActions.__slots__ = [
    'num',
    'normal',
    'shake',
    'jawOpen',
    'headRaise',
    'blink',
]
struct_HFFaceInteractionsActions._fields_ = [
    ('num', HInt32),
    ('normal', HPInt32),
    ('shake', HPInt32),
    ('jawOpen', HPInt32),
    ('headRaise', HPInt32),
    ('blink', HPInt32),
]

HFFaceInteractionsActions = struct_HFFaceInteractionsActions# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1270

PHFFaceInteractionsActions = POINTER(struct_HFFaceInteractionsActions)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1270

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1278
if _libs[_LIBRARY_FILENAME].has("HFGetFaceInteractionActionsResult", "cdecl"):
    HFGetFaceInteractionActionsResult = _libs[_LIBRARY_FILENAME].get("HFGetFaceInteractionActionsResult", "cdecl")
    HFGetFaceInteractionActionsResult.argtypes = [HFSession, PHFFaceInteractionsActions]
    HFGetFaceInteractionActionsResult.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1306
class struct_HFFaceAttributeResult(Structure):
    pass

struct_HFFaceAttributeResult.__slots__ = [
    'num',
    'race',
    'gender',
    'ageBracket',
]
struct_HFFaceAttributeResult._fields_ = [
    ('num', HInt32),
    ('race', HPInt32),
    ('gender', HPInt32),
    ('ageBracket', HPInt32),
]

HFFaceAttributeResult = struct_HFFaceAttributeResult# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1306

PHFFaceAttributeResult = POINTER(struct_HFFaceAttributeResult)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1306

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1318
if _libs[_LIBRARY_FILENAME].has("HFGetFaceAttributeResult", "cdecl"):
    HFGetFaceAttributeResult = _libs[_LIBRARY_FILENAME].get("HFGetFaceAttributeResult", "cdecl")
    HFGetFaceAttributeResult.argtypes = [HFSession, PHFFaceAttributeResult]
    HFGetFaceAttributeResult.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1333
class struct_HFFaceEmotionResult(Structure):
    pass

struct_HFFaceEmotionResult.__slots__ = [
    'num',
    'emotion',
]
struct_HFFaceEmotionResult._fields_ = [
    ('num', HInt32),
    ('emotion', HPInt32),
]

HFFaceEmotionResult = struct_HFFaceEmotionResult# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1333

PHFFaceEmotionResult = POINTER(struct_HFFaceEmotionResult)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1333

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1341
if _libs[_LIBRARY_FILENAME].has("HFGetFaceEmotionResult", "cdecl"):
    HFGetFaceEmotionResult = _libs[_LIBRARY_FILENAME].get("HFGetFaceEmotionResult", "cdecl")
    HFGetFaceEmotionResult.argtypes = [HFSession, PHFFaceEmotionResult]
    HFGetFaceEmotionResult.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1354
class struct_HFInspireFaceVersion(Structure):
    pass

struct_HFInspireFaceVersion.__slots__ = [
    'major',
    'minor',
    'patch',
]
struct_HFInspireFaceVersion._fields_ = [
    ('major', HInt32),
    ('minor', HInt32),
    ('patch', HInt32),
]

HFInspireFaceVersion = struct_HFInspireFaceVersion# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1354

PHFInspireFaceVersion = POINTER(struct_HFInspireFaceVersion)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1354

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1364
if _libs[_LIBRARY_FILENAME].has("HFQueryInspireFaceVersion", "cdecl"):
    HFQueryInspireFaceVersion = _libs[_LIBRARY_FILENAME].get("HFQueryInspireFaceVersion", "cdecl")
    HFQueryInspireFaceVersion.argtypes = [PHFInspireFaceVersion]
    HFQueryInspireFaceVersion.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFQueryCAPILevel", "cdecl"):
    HFQueryCAPILevel = _libs[_LIBRARY_FILENAME].get("HFQueryCAPILevel", "cdecl")
    HFQueryCAPILevel.argtypes = [POINTER(HFUInt32)]
    HFQueryCAPILevel.restype = HFStatus

enum_HFComponentType = c_int

HF_COMPONENT_MNN = 0
HF_COMPONENT_INSPIRECV = 1
HF_COMPONENT_EIGEN = 2
HF_COMPONENT_SQLITE = 3
HF_COMPONENT_SQLITE_VEC = 4
HF_COMPONENT_NLOHMANN_JSON = 5
HF_COMPONENT_OPENCV = 6
HF_COMPONENT_TENSORRT = 7
HF_COMPONENT_CUDA = 8
HF_COMPONENT_RKNN = 9
HF_COMPONENT_RGA = 10
HF_COMPONENT_COREML = 11
HF_COMPONENT_COUNT = 12

HFComponentType = enum_HFComponentType

enum_HFComponentVersionState = c_int

HF_COMPONENT_VERSION_DISABLED = 0
HF_COMPONENT_VERSION_KNOWN = 1
HF_COMPONENT_VERSION_UNKNOWN = 2

HFComponentVersionState = enum_HFComponentVersionState

class struct_HFComponentVersion(Structure):
    pass

struct_HFComponentVersion.__slots__ = [
    'major',
    'minor',
    'patch',
    'state',
]
struct_HFComponentVersion._fields_ = [
    ('major', HInt32),
    ('minor', HInt32),
    ('patch', HInt32),
    ('state', HFComponentVersionState),
]

HFComponentVersion = struct_HFComponentVersion
PHFComponentVersion = POINTER(struct_HFComponentVersion)

if _libs[_LIBRARY_FILENAME].has("HFQueryInspireFaceComponentVersion", "cdecl"):
    HFQueryInspireFaceComponentVersion = _libs[_LIBRARY_FILENAME].get("HFQueryInspireFaceComponentVersion", "cdecl")
    HFQueryInspireFaceComponentVersion.argtypes = [HFComponentType, PHFComponentVersion]
    HFQueryInspireFaceComponentVersion.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFQueryInspireFaceComponentVersions", "cdecl"):
    HFQueryInspireFaceComponentVersions = _libs[_LIBRARY_FILENAME].get("HFQueryInspireFaceComponentVersions", "cdecl")
    HFQueryInspireFaceComponentVersions.argtypes = [HString, HInt32, HPInt32]
    HFQueryInspireFaceComponentVersions.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFQueryInspireFaceDiagnosticInformation", "cdecl"):
    HFQueryInspireFaceDiagnosticInformation = _libs[_LIBRARY_FILENAME].get("HFQueryInspireFaceDiagnosticInformation", "cdecl")
    HFQueryInspireFaceDiagnosticInformation.argtypes = [HString, HInt32, HPInt32]
    HFQueryInspireFaceDiagnosticInformation.restype = HResult

if _libs[_LIBRARY_FILENAME].has("HFGetErrorMessage", "cdecl"):
    HFGetErrorMessage = _libs[_LIBRARY_FILENAME].get("HFGetErrorMessage", "cdecl")
    HFGetErrorMessage.argtypes = [HResult, HString, HInt32, HPInt32]
    HFGetErrorMessage.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1372
class struct_HFInspireFaceExtendedInformation(Structure):
    pass

struct_HFInspireFaceExtendedInformation.__slots__ = [
    'information',
]
struct_HFInspireFaceExtendedInformation._fields_ = [
    ('information', HChar * int(256)),
]

HFInspireFaceExtendedInformation = struct_HFInspireFaceExtendedInformation# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1372

PHFInspireFaceExtendedInformation = POINTER(struct_HFInspireFaceExtendedInformation)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1372

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1379
if _libs[_LIBRARY_FILENAME].has("HFQueryInspireFaceExtendedInformation", "cdecl"):
    HFQueryInspireFaceExtendedInformation = _libs[_LIBRARY_FILENAME].get("HFQueryInspireFaceExtendedInformation", "cdecl")
    HFQueryInspireFaceExtendedInformation.argtypes = [PHFInspireFaceExtendedInformation]
    HFQueryInspireFaceExtendedInformation.restype = HResult

enum_HFLogLevel = c_int# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1391

HF_LOG_NONE = 0# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1391

HF_LOG_DEBUG = (HF_LOG_NONE + 1)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1391

HF_LOG_INFO = (HF_LOG_DEBUG + 1)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1391

HF_LOG_WARN = (HF_LOG_INFO + 1)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1391

HF_LOG_ERROR = (HF_LOG_WARN + 1)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1391

HF_LOG_FATAL = (HF_LOG_ERROR + 1)# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1391

HFLogLevel = enum_HFLogLevel# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1391

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1396
if _libs[_LIBRARY_FILENAME].has("HFSetLogLevel", "cdecl"):
    HFSetLogLevel = _libs[_LIBRARY_FILENAME].get("HFSetLogLevel", "cdecl")
    HFSetLogLevel.argtypes = [HFLogLevel]
    HFSetLogLevel.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1401
if _libs[_LIBRARY_FILENAME].has("HFLogDisable", "cdecl"):
    HFLogDisable = _libs[_LIBRARY_FILENAME].get("HFLogDisable", "cdecl")
    HFLogDisable.argtypes = []
    HFLogDisable.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1411
if _libs[_LIBRARY_FILENAME].has("HFLogPrint", "cdecl"):
    _func = _libs[_LIBRARY_FILENAME].get("HFLogPrint", "cdecl")
    _restype = HResult
    _errcheck = None
    _argtypes = [HFLogLevel, HFormat]
    HFLogPrint = _variadic_function(_func,_restype,_argtypes,_errcheck)

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1424
if _libs[_LIBRARY_FILENAME].has("HFDeBugImageStreamImShow", "cdecl"):
    HFDeBugImageStreamImShow = _libs[_LIBRARY_FILENAME].get("HFDeBugImageStreamImShow", "cdecl")
    HFDeBugImageStreamImShow.argtypes = [HFImageStream]
    HFDeBugImageStreamImShow.restype = None

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1436
if _libs[_LIBRARY_FILENAME].has("HFDeBugImageStreamDecodeSave", "cdecl"):
    HFDeBugImageStreamDecodeSave = _libs[_LIBRARY_FILENAME].get("HFDeBugImageStreamDecodeSave", "cdecl")
    HFDeBugImageStreamDecodeSave.argtypes = [HFImageStream, HPath]
    HFDeBugImageStreamDecodeSave.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1451
if _libs[_LIBRARY_FILENAME].has("HFDeBugShowResourceStatistics", "cdecl"):
    HFDeBugShowResourceStatistics = _libs[_LIBRARY_FILENAME].get("HFDeBugShowResourceStatistics", "cdecl")
    HFDeBugShowResourceStatistics.argtypes = []
    HFDeBugShowResourceStatistics.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1461
if _libs[_LIBRARY_FILENAME].has("HFDeBugGetUnreleasedSessionsCount", "cdecl"):
    HFDeBugGetUnreleasedSessionsCount = _libs[_LIBRARY_FILENAME].get("HFDeBugGetUnreleasedSessionsCount", "cdecl")
    HFDeBugGetUnreleasedSessionsCount.argtypes = [HPInt32]
    HFDeBugGetUnreleasedSessionsCount.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1472
if _libs[_LIBRARY_FILENAME].has("HFDeBugGetUnreleasedSessions", "cdecl"):
    HFDeBugGetUnreleasedSessions = _libs[_LIBRARY_FILENAME].get("HFDeBugGetUnreleasedSessions", "cdecl")
    HFDeBugGetUnreleasedSessions.argtypes = [PHFSession, HInt32]
    HFDeBugGetUnreleasedSessions.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1482
if _libs[_LIBRARY_FILENAME].has("HFDeBugGetUnreleasedStreamsCount", "cdecl"):
    HFDeBugGetUnreleasedStreamsCount = _libs[_LIBRARY_FILENAME].get("HFDeBugGetUnreleasedStreamsCount", "cdecl")
    HFDeBugGetUnreleasedStreamsCount.argtypes = [HPInt32]
    HFDeBugGetUnreleasedStreamsCount.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1493
if _libs[_LIBRARY_FILENAME].has("HFDeBugGetUnreleasedStreams", "cdecl"):
    HFDeBugGetUnreleasedStreams = _libs[_LIBRARY_FILENAME].get("HFDeBugGetUnreleasedStreams", "cdecl")
    HFDeBugGetUnreleasedStreams.argtypes = [PHFImageStream, HInt32]
    HFDeBugGetUnreleasedStreams.restype = HResult

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 27
HF_STATUS_ENABLE = 1

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 28
HF_STATUS_DISABLE = 0

HF_C_API_LEVEL = 2

HF_SESSION_CONFIG_V2_VERSION = 1

HF_FACE_CAPTURE_CONFIG_VERSION = 1

HF_FACE_CAPTURE_MAX_RESULTS = 8

HF_RESOURCE_PACK_INFO_VERSION = 1

HF_RESOURCE_PACK_TAG_CAPACITY = 64

HF_RESOURCE_PACK_VERSION_CAPACITY = 64

HF_RESOURCE_PACK_MAJOR_CAPACITY = 64

HF_RESOURCE_PACK_RELEASE_CAPACITY = 64

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 30
HF_ENABLE_NONE = 0x00000000

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 31
HF_ENABLE_FACE_RECOGNITION = 0x00000002

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 32
HF_ENABLE_LIVENESS = 0x00000004

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 33
HF_ENABLE_IR_LIVENESS = 0x00000008

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 34
HF_ENABLE_MASK_DETECT = 0x00000010

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 35
HF_ENABLE_FACE_ATTRIBUTE = 0x00000020

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 36
HF_ENABLE_PLACEHOLDER_ = 0x00000040

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 37
HF_ENABLE_QUALITY = 0x00000080

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 38
HF_ENABLE_INTERACTION = 0x00000100

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 39
HF_ENABLE_FACE_POSE = 0x00000200

# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 40
HF_ENABLE_FACE_EMOTION = 0x00000400

HF_CAPTURE_FILTER_NONE = 0
HF_CAPTURE_FILTER_FACE_COUNT = 1 << 0
HF_CAPTURE_FILTER_FACE_SIZE = 1 << 1
HF_CAPTURE_FILTER_FACE_POSITION = 1 << 2
HF_CAPTURE_FILTER_FACE_BOUNDARY = 1 << 3
HF_CAPTURE_FILTER_STABILITY = 1 << 4
HF_CAPTURE_FILTER_POSE = 1 << 5
HF_CAPTURE_FILTER_QUALITY = 1 << 6
HF_CAPTURE_FILTER_SHARPNESS = 1 << 7
HF_CAPTURE_FILTER_BRIGHTNESS = 1 << 8
HF_CAPTURE_FILTER_TRACK_COUNT = 1 << 9

HF_CAPTURE_REJECT_NONE = 0
HF_CAPTURE_REJECT_NO_FACE = 1 << 0
HF_CAPTURE_REJECT_MULTIPLE_FACES = 1 << 1
HF_CAPTURE_REJECT_FACE_TOO_SMALL = 1 << 2
HF_CAPTURE_REJECT_FACE_TOO_LARGE = 1 << 3
HF_CAPTURE_REJECT_FACE_OFF_CENTER = 1 << 4
HF_CAPTURE_REJECT_FACE_OUT_OF_BOUNDS = 1 << 5
HF_CAPTURE_REJECT_UNSTABLE = 1 << 6
HF_CAPTURE_REJECT_POSE = 1 << 7
HF_CAPTURE_REJECT_QUALITY = 1 << 8
HF_CAPTURE_REJECT_SHARPNESS = 1 << 9
HF_CAPTURE_REJECT_BRIGHTNESS = 1 << 10
HF_CAPTURE_REJECT_TRACK_COUNT_TOO_LOW = 1 << 11

HF_CAPTURE_STATE_IDLE = 0
HF_CAPTURE_STATE_STABILIZING = 1
HF_CAPTURE_STATE_COLLECTING = 2
HF_CAPTURE_STATE_READY = 3
HF_CAPTURE_STATE_FINISHED = 4
HF_CAPTURE_STATE_TRACK_LOST = 5

HFImageData = struct_HFImageData# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 123

HFImageBitmapData = struct_HFImageBitmapData# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 200

HFSessionCustomParameter = struct_HFSessionCustomParameter# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 485

HFSessionConfigV2 = struct_HFSessionConfigV2

HFResourcePackInfo = struct_HFResourcePackInfo

HFFaceDetectPixelList = struct_HFFaceDetectPixelList# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 523

HFFaceBasicToken = struct_HFFaceBasicToken# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 589

HFFaceEulerAngle = struct_HFFaceEulerAngle# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 600

HFMultipleFaceData = struct_HFMultipleFaceData# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 616

HFFaceCaptureConfig = struct_HFFaceCaptureConfig

HFFaceCaptureMetrics = struct_HFFaceCaptureMetrics

HFFaceCaptureProgress = struct_HFFaceCaptureProgress

HFFaceCaptureResult = struct_HFFaceCaptureResult

HFFaceFeature = struct_HFFaceFeature# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 809

HFFeatureHubConfiguration = struct_HFFeatureHubConfiguration# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 916

HFFaceFeatureIdentity = struct_HFFaceFeatureIdentity# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 945

HFFeatureHubSearchResultV2 = struct_HFFeatureHubSearchResultV2

HFSearchTopKResults = struct_HFSearchTopKResults# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 954

HFSimilarityConverterConfig = struct_HFSimilarityConverterConfig# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1014

HFFeatureHubExistingIds = struct_HFFeatureHubExistingIds# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1115

HFRGBLivenessConfidence = struct_HFRGBLivenessConfidence# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1170

HFFaceMaskConfidence = struct_HFFaceMaskConfidence# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1193

HFFaceQualityConfidence = struct_HFFaceQualityConfidence# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1216

HFFaceInteractionState = struct_HFFaceInteractionState# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1251

HFFaceInteractionsActions = struct_HFFaceInteractionsActions# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1270

HFFaceAttributeResult = struct_HFFaceAttributeResult# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1306

HFFaceEmotionResult = struct_HFFaceEmotionResult# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1333

HFInspireFaceVersion = struct_HFInspireFaceVersion# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1354

HFComponentVersion = struct_HFComponentVersion

HFInspireFaceExtendedInformation = struct_HFInspireFaceExtendedInformation# /Users/tunm/work/InspireFace/cpp/inspireface/c_api/inspireface.h: 1372

# No inserted files

# No prefix-stripping
