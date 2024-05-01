/*
   Copyright 2017-2023 Leon Merten Lohse

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#ifndef NPY_HPP_
#define NPY_HPP_

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

namespace npy {

/* Compile-time test for byte order.
If your compiler does not define these per default, you may want to define
one of these constants manually.
Defaults to little endian order. */
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN || defined(__BIG_ENDIAN__) || defined(__ARMEB__) || \
defined(__THUMBEB__) || defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
const bool big_endian = true;
#else
const bool big_endian = false;
#endif

const size_t magic_string_length = 6;
const std::array<char, magic_string_length> magic_string = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

const char little_endian_char = '<';
const char big_endian_char = '>';
const char no_endian_char = '|';

constexpr std::array<char, 3> endian_chars = {little_endian_char, big_endian_char, no_endian_char};
constexpr std::array<char, 4> numtype_chars = {'f', 'i', 'u', 'c'};

constexpr char host_endian_char = (big_endian ? big_endian_char : little_endian_char);

/* npy array length */
using ndarray_len_t = unsigned long int;
using shape_t = std::vector<ndarray_len_t>;

using version_t = std::pair<char, char>;

struct dtype_t {
    char byteorder;
    char kind;
    unsigned int itemsize;

    inline std::string str() const {
        std::stringstream ss;
        ss << byteorder << kind << itemsize;
        return ss.str();
    }

    inline std::tuple<const char, const char, const unsigned int> tie() const {
        return std::tie(byteorder, kind, itemsize);
    }
};

struct header_t {
    dtype_t dtype;
    bool fortran_order;
    shape_t shape;
};

inline void write_magic(std::ostream &ostream, version_t version) {
    ostream.write(magic_string.data(), magic_string_length);
    ostream.put(version.first);
    ostream.put(version.second);
}

inline version_t read_magic(std::istream &istream) {
    std::array<char, magic_string_length + 2> buf{};
    istream.read(buf.data(), sizeof(buf));

    if (!istream) {
        throw std::runtime_error("io error: failed reading file");
    }

    if (!std::equal(magic_string.begin(), magic_string.end(), buf.begin()))
        throw std::runtime_error("this file does not have a valid npy format.");

    version_t version;
    version.first = buf[magic_string_length];
    version.second = buf[magic_string_length + 1];

    return version;
}

const std::unordered_map<std::type_index, dtype_t> dtype_map = {
        {std::type_index(typeid(float)), {host_endian_char, 'f', sizeof(float)}},
        {std::type_index(typeid(double)), {host_endian_char, 'f', sizeof(double)}},
        {std::type_index(typeid(long double)), {host_endian_char, 'f', sizeof(long double)}},
        {std::type_index(typeid(char)), {no_endian_char, 'i', sizeof(char)}},
        {std::type_index(typeid(signed char)), {no_endian_char, 'i', sizeof(signed char)}},
        {std::type_index(typeid(short)), {host_endian_char, 'i', sizeof(short)}},
        {std::type_index(typeid(int)), {host_endian_char, 'i', sizeof(int)}},
        {std::type_index(typeid(long)), {host_endian_char, 'i', sizeof(long)}},
        {std::type_index(typeid(long long)), {host_endian_char, 'i', sizeof(long long)}},
        {std::type_index(typeid(unsigned char)), {no_endian_char, 'u', sizeof(unsigned char)}},
        {std::type_index(typeid(unsigned short)), {host_endian_char, 'u', sizeof(unsigned short)}},
        {std::type_index(typeid(unsigned int)), {host_endian_char, 'u', sizeof(unsigned int)}},
        {std::type_index(typeid(unsigned long)), {host_endian_char, 'u', sizeof(unsigned long)}},
        {std::type_index(typeid(unsigned long long)), {host_endian_char, 'u', sizeof(unsigned long long)}},
        {std::type_index(typeid(std::complex<float>)), {host_endian_char, 'c', sizeof(std::complex<float>)}},
        {std::type_index(typeid(std::complex<double>)), {host_endian_char, 'c', sizeof(std::complex<double>)}},
        {std::type_index(typeid(std::complex<long double>)), {host_endian_char, 'c', sizeof(std::complex<long double>)}}};

// helpers
inline bool is_digits(const std::string &str) { return std::all_of(str.begin(), str.end(), ::isdigit); }

template <typename T, size_t N>
inline bool in_array(T val, const std::array<T, N> &arr) {
    return std::find(std::begin(arr), std::end(arr), val) != std::end(arr);
}

inline dtype_t parse_descr(std::string typestring) {
    if (typestring.length() < 3) {
        throw std::runtime_error("invalid typestring (length)");
    }

    char byteorder_c = typestring.at(0);
    char kind_c = typestring.at(1);
    std::string itemsize_s = typestring.substr(2);

    if (!in_array(byteorder_c, endian_chars)) {
        throw std::runtime_error("invalid typestring (byteorder)");
    }

    if (!in_array(kind_c, numtype_chars)) {
        throw std::runtime_error("invalid typestring (kind)");
    }

    if (!is_digits(itemsize_s)) {
        throw std::runtime_error("invalid typestring (itemsize)");
    }
    unsigned int itemsize = std::stoul(itemsize_s);

    return {byteorder_c, kind_c, itemsize};
}

namespace pyparse {

/**
Removes leading and trailing whitespaces
*/
    inline std::string trim(const std::string &str) {
        const std::string whitespace = " \t";
        auto begin = str.find_first_not_of(whitespace);

        if (begin == std::string::npos) return "";

        auto end = str.find_last_not_of(whitespace);

        return str.substr(begin, end - begin + 1);
    }

    inline std::string get_value_from_map(const std::string &mapstr) {
        size_t sep_pos = mapstr.find_first_of(":");
        if (sep_pos == std::string::npos) return "";

        std::string tmp = mapstr.substr(sep_pos + 1);
        return trim(tmp);
    }

/**
Parses the string representation of a Python dict

The keys need to be known and may not appear anywhere else in the data.
*/
    inline std::unordered_map<std::string, std::string> parse_dict(std::string in, const std::vector<std::string> &keys) {
        std::unordered_map<std::string, std::string> map;

        if (keys.size() == 0) return map;

        in = trim(in);

        // unwrap dictionary
        if ((in.front() == '{') && (in.back() == '}'))
            in = in.substr(1, in.length() - 2);
        else
            throw std::runtime_error("Not a Python dictionary.");

        std::vector<std::pair<size_t, std::string>> positions;

        for (auto const &value : keys) {
            size_t pos = in.find("'" + value + "'");

            if (pos == std::string::npos) throw std::runtime_error("Missing '" + value + "' key.");

            std::pair<size_t, std::string> position_pair{pos, value};
            positions.push_back(position_pair);
        }

        // sort by position in dict
        std::sort(positions.begin(), positions.end());

        for (size_t i = 0; i < positions.size(); ++i) {
            std::string raw_value;
            size_t begin{positions[i].first};
            size_t end{std::string::npos};

            std::string key = positions[i].second;

            if (i + 1 < positions.size()) end = positions[i + 1].first;

            raw_value = in.substr(begin, end - begin);

            raw_value = trim(raw_value);

            if (raw_value.back() == ',') raw_value.pop_back();

            map[key] = get_value_from_map(raw_value);
        }

        return map;
    }

/**
Parses the string representation of a Python boolean
*/
    inline bool parse_bool(const std::string &in) {
        if (in == "True") return true;
        if (in == "False") return false;

        throw std::runtime_error("Invalid python boolan.");
    }

/**
Parses the string representation of a Python str
*/
    inline std::string parse_str(const std::string &in) {
        if ((in.front() == '\'') && (in.back() == '\'')) return in.substr(1, in.length() - 2);

        throw std::runtime_error("Invalid python string.");
    }

/**
Parses the string represenatation of a Python tuple into a vector of its items
*/
    inline std::vector<std::string> parse_tuple(std::string in) {
        std::vector<std::string> v;
        const char seperator = ',';

        in = trim(in);

        if ((in.front() == '(') && (in.back() == ')'))
            in = in.substr(1, in.length() - 2);
        else
            throw std::runtime_error("Invalid Python tuple.");

        std::istringstream iss(in);

        for (std::string token; std::getline(iss, token, seperator);) {
            v.push_back(token);
        }

        return v;
    }

    template <typename T>
    inline std::string write_tuple(const std::vector<T> &v) {
        if (v.size() == 0) return "()";

        std::ostringstream ss;
        ss.imbue(std::locale("C"));

        if (v.size() == 1) {
            ss << "(" << v.front() << ",)";
        } else {
            const std::string delimiter = ", ";
            // v.size() > 1
            ss << "(";
            std::copy(v.begin(), v.end() - 1, std::ostream_iterator<T>(ss, delimiter.c_str()));
            ss << v.back();
            ss << ")";
        }

        return ss.str();
    }

    inline std::string write_boolean(bool b) {
        if (b)
            return "True";
        else
            return "False";
    }

}  // namespace pyparse

inline header_t parse_header(std::string header) {
    /*
       The first 6 bytes are a magic string: exactly "x93NUMPY".
       The next 1 byte is an unsigned byte: the major version number of the file
       format, e.g. x01. The next 1 byte is an unsigned byte: the minor version
       number of the file format, e.g. x00. Note: the version of the file format
       is not tied to the version of the numpy package. The next 2 bytes form a
       little-endian unsigned short int: the length of the header data HEADER_LEN.
       The next HEADER_LEN bytes form the header data describing the array's
       format. It is an ASCII string which contains a Python literal expression of
       a dictionary. It is terminated by a newline ('n') and padded with spaces
       ('x20') to make the total length of the magic string + 4 + HEADER_LEN be
       evenly divisible by 16 for alignment purposes. The dictionary contains
       three keys:

       "descr" : dtype.descr
       An object that can be passed as an argument to the numpy.dtype()
       constructor to create the array's dtype. "fortran_order" : bool Whether the
       array data is Fortran-contiguous or not. Since Fortran-contiguous arrays
       are a common form of non-C-contiguity, we allow them to be written directly
       to disk for efficiency. "shape" : tuple of int The shape of the array. For
       repeatability and readability, this dictionary is formatted using
       pprint.pformat() so the keys are in alphabetic order.
     */

    // remove trailing newline
    if (header.back() != '\n') throw std::runtime_error("invalid header");
    header.pop_back();

    // parse the dictionary
    std::vector<std::string> keys{"descr", "fortran_order", "shape"};
    auto dict_map = npy::pyparse::parse_dict(header, keys);

    if (dict_map.size() == 0) throw std::runtime_error("invalid dictionary in header");

    std::string descr_s = dict_map["descr"];
    std::string fortran_s = dict_map["fortran_order"];
    std::string shape_s = dict_map["shape"];

    std::string descr = npy::pyparse::parse_str(descr_s);
    dtype_t dtype = parse_descr(descr);

    // convert literal Python bool to C++ bool
    bool fortran_order = npy::pyparse::parse_bool(fortran_s);

    // parse the shape tuple
    auto shape_v = npy::pyparse::parse_tuple(shape_s);

    shape_t shape;
    for (auto item : shape_v) {
        auto dim = static_cast<ndarray_len_t>(std::stoul(item));
        shape.push_back(dim);
    }

    return {dtype, fortran_order, shape};
}

inline std::string write_header_dict(const std::string &descr, bool fortran_order, const shape_t &shape) {
    std::string s_fortran_order = npy::pyparse::write_boolean(fortran_order);
    std::string shape_s = npy::pyparse::write_tuple(shape);

    return "{'descr': '" + descr + "', 'fortran_order': " + s_fortran_order + ", 'shape': " + shape_s + ", }";
}

inline void write_header(std::ostream &out, const header_t &header) {
    std::string header_dict = write_header_dict(header.dtype.str(), header.fortran_order, header.shape);

    size_t length = magic_string_length + 2 + 2 + header_dict.length() + 1;

    version_t version{1, 0};
    if (length >= 255 * 255) {
        length = magic_string_length + 2 + 4 + header_dict.length() + 1;
        version = {2, 0};
    }
    size_t padding_len = 16 - length % 16;
    std::string padding(padding_len, ' ');

    // write magic
    write_magic(out, version);

    // write header length
    if (version == version_t{1, 0}) {
        auto header_len = static_cast<uint16_t>(header_dict.length() + padding.length() + 1);

        std::array<uint8_t, 2> header_len_le16{static_cast<uint8_t>((header_len >> 0) & 0xff),
                                               static_cast<uint8_t>((header_len >> 8) & 0xff)};
        out.write(reinterpret_cast<char *>(header_len_le16.data()), 2);
    } else {
        auto header_len = static_cast<uint32_t>(header_dict.length() + padding.length() + 1);

        std::array<uint8_t, 4> header_len_le32{
                static_cast<uint8_t>((header_len >> 0) & 0xff), static_cast<uint8_t>((header_len >> 8) & 0xff),
                static_cast<uint8_t>((header_len >> 16) & 0xff), static_cast<uint8_t>((header_len >> 24) & 0xff)};
        out.write(reinterpret_cast<char *>(header_len_le32.data()), 4);
    }

    out << header_dict << padding << '\n';
}

inline std::string read_header(std::istream &istream) {
    // check magic bytes an version number
    version_t version = read_magic(istream);

    uint32_t header_length = 0;
    if (version == version_t{1, 0}) {
        std::array<uint8_t, 2> header_len_le16{};
        istream.read(reinterpret_cast<char *>(header_len_le16.data()), 2);
        header_length = (header_len_le16[0] << 0) | (header_len_le16[1] << 8);

        if ((magic_string_length + 2 + 2 + header_length) % 16 != 0) {
            // TODO(llohse): display warning
        }
    } else if (version == version_t{2, 0}) {
        std::array<uint8_t, 4> header_len_le32{};
        istream.read(reinterpret_cast<char *>(header_len_le32.data()), 4);

        header_length =
                (header_len_le32[0] << 0) | (header_len_le32[1] << 8) | (header_len_le32[2] << 16) | (header_len_le32[3] << 24);

        if ((magic_string_length + 2 + 4 + header_length) % 16 != 0) {
            // TODO(llohse): display warning
        }
    } else {
        throw std::runtime_error("unsupported file format version");
    }

    auto buf_v = std::vector<char>(header_length);
    istream.read(buf_v.data(), header_length);
    std::string header(buf_v.data(), header_length);

    return header;
}

inline ndarray_len_t comp_size(const shape_t &shape) {
    ndarray_len_t size = 1;
    for (ndarray_len_t i : shape) size *= i;

    return size;
}

template <typename Scalar>
struct npy_data {
    std::vector<Scalar> data = {};
    shape_t shape = {};
    bool fortran_order = false;
};

template <typename Scalar>
struct npy_data_ptr {
    const Scalar *data_ptr = nullptr;
    shape_t shape = {};
    bool fortran_order = false;
};

template <typename Scalar>
inline npy_data<Scalar> read_npy(std::istream &in) {
    std::string header_s = read_header(in);

    // parse header
    header_t header = parse_header(header_s);

    // check if the typestring matches the given one
    const dtype_t dtype = dtype_map.at(std::type_index(typeid(Scalar)));

    if (header.dtype.tie() != dtype.tie()) {
        throw std::runtime_error("formatting error: typestrings not matching");
    }

    // compute the data size based on the shape
    auto size = static_cast<size_t>(comp_size(header.shape));

    npy_data<Scalar> data;

    data.shape = header.shape;
    data.fortran_order = header.fortran_order;

    data.data.resize(size);

    // read the data
    in.read(reinterpret_cast<char *>(data.data.data()), sizeof(Scalar) * size);

    return data;
}

template <typename Scalar>
inline npy_data<Scalar> read_npy(const std::string &filename) {
    std::ifstream stream(filename, std::ifstream::binary);
    if (!stream) {
        throw std::runtime_error("io error: failed to open a file.");
    }

    return read_npy<Scalar>(stream);
}

template <typename Scalar>
inline void write_npy(std::ostream &out, const npy_data<Scalar> &data) {
    //  static_assert(has_typestring<Scalar>::value, "scalar type not
    //  understood");
    const dtype_t dtype = dtype_map.at(std::type_index(typeid(Scalar)));

    header_t header{dtype, data.fortran_order, data.shape};
    write_header(out, header);

    auto size = static_cast<size_t>(comp_size(data.shape));

    out.write(reinterpret_cast<const char *>(data.data.data()), sizeof(Scalar) * size);
}

template <typename Scalar>
inline void write_npy(const std::string &filename, const npy_data<Scalar> &data) {
    std::ofstream stream(filename, std::ofstream::binary);
    if (!stream) {
        throw std::runtime_error("io error: failed to open a file.");
    }

    write_npy<Scalar>(stream, data);
}

template <typename Scalar>
inline void write_npy(std::ostream &out, const npy_data_ptr<Scalar> &data_ptr) {
    const dtype_t dtype = dtype_map.at(std::type_index(typeid(Scalar)));

    header_t header{dtype, data_ptr.fortran_order, data_ptr.shape};
    write_header(out, header);

    auto size = static_cast<size_t>(comp_size(data_ptr.shape));

    out.write(reinterpret_cast<const char *>(data_ptr.data_ptr), sizeof(Scalar) * size);
}

template <typename Scalar>
inline void write_npy(const std::string &filename, const npy_data_ptr<Scalar> &data_ptr) {
    std::ofstream stream(filename, std::ofstream::binary);
    if (!stream) {
        throw std::runtime_error("io error: failed to open a file.");
    }

    write_npy<Scalar>(stream, data_ptr);
}

// old interface

// NOLINTBEGIN(*-avoid-c-arrays)
template <typename Scalar>
inline void SaveArrayAsNumpy(const std::string &filename, bool fortran_order, unsigned int n_dims,
                             const unsigned long shape[], const Scalar *data) {
    const npy_data_ptr<Scalar> ptr{data, {shape, shape + n_dims}, fortran_order};

    write_npy<Scalar>(filename, ptr);
}

template <typename Scalar>
inline void SaveArrayAsNumpy(const std::string &filename, bool fortran_order, unsigned int n_dims,
                             const unsigned long shape[], const std::vector<Scalar> &data) {
    SaveArrayAsNumpy(filename, fortran_order, n_dims, shape, data.data());
}

template <typename Scalar>
inline void LoadArrayFromNumpy(const std::string &filename, std::vector<unsigned long> &shape, bool &fortran_order,
                               std::vector<Scalar> &data) {
    const npy_data<Scalar> n_data = read_npy<Scalar>(filename);

    shape = n_data.shape;
    fortran_order = n_data.fortran_order;

    std::copy(n_data.data.begin(), n_data.data.end(), std::back_inserter(data));
}

template <typename Scalar>
inline void LoadArrayFromNumpy(const std::string &filename, std::vector<unsigned long> &shape,
                               std::vector<Scalar> &data) {
    bool fortran_order = false;
    LoadArrayFromNumpy<Scalar>(filename, shape, fortran_order, data);
}
// NOLINTEND(*-avoid-c-arrays)

}  // namespace npy

#endif  // NPY_HPP_