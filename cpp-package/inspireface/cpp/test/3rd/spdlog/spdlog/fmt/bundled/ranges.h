// Formatting library for C++ - experimental range support
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.
//
// Copyright (c) 2018 - present, Remotion (Igor Schulz)
// All Rights Reserved
// {fmt} support for ranges, containers and types tuple interface.

#ifndef FMT_RANGES_H_
#define FMT_RANGES_H_

#include <initializer_list>
#include <tuple>
#include <type_traits>

#include "format.h"

FMT_BEGIN_NAMESPACE

namespace detail {

template <typename RangeT, typename OutputIterator>
OutputIterator copy(const RangeT& range, OutputIterator out) {
  for (auto it = range.begin(), end = range.end(); it != end; ++it)
    *out++ = *it;
  return out;
}

template <typename OutputIterator>
OutputIterator copy(const char* str, OutputIterator out) {
  while (*str) *out++ = *str++;
  return out;
}

template <typename OutputIterator>
OutputIterator copy(char ch, OutputIterator out) {
  *out++ = ch;
  return out;
}

template <typename OutputIterator>
OutputIterator copy(wchar_t ch, OutputIterator out) {
  *out++ = ch;
  return out;
}

// Returns true if T has a std::string-like interface, like std::string_view.
template <typename T> class is_std_string_like {
  template <typename U>
  static auto check(U* p)
      -> decltype((void)p->find('a'), p->length(), (void)p->data(), int());
  template <typename> static void check(...);

 public:
  static FMT_CONSTEXPR_DECL const bool value =
      is_string<T>::value ||
      std::is_convertible<T, std_string_view<char>>::value ||
      !std::is_void<decltype(check<T>(nullptr))>::value;
};

template <typename Char>
struct is_std_string_like<fmt::basic_string_view<Char>> : std::true_type {};

template <typename T> class is_map {
  template <typename U> static auto check(U*) -> typename U::mapped_type;
  template <typename> static void check(...);

 public:
#ifdef FMT_FORMAT_MAP_AS_LIST
  static FMT_CONSTEXPR_DECL const bool value = false;
#else
  static FMT_CONSTEXPR_DECL const bool value =
      !std::is_void<decltype(check<T>(nullptr))>::value;
#endif
};

template <typename T> class is_set {
  template <typename U> static auto check(U*) -> typename U::key_type;
  template <typename> static void check(...);

 public:
#ifdef FMT_FORMAT_SET_AS_LIST
  static FMT_CONSTEXPR_DECL const bool value = false;
#else
  static FMT_CONSTEXPR_DECL const bool value =
      !std::is_void<decltype(check<T>(nullptr))>::value && !is_map<T>::value;
#endif
};

template <typename... Ts> struct conditional_helper {};

template <typename T, typename _ = void> struct is_range_ : std::false_type {};

#if !FMT_MSC_VER || FMT_MSC_VER > 1800

#  define FMT_DECLTYPE_RETURN(val)  \
    ->decltype(val) { return val; } \
    static_assert(                  \
        true, "")  // This makes it so that a semicolon is required after the
                   // macro, which helps clang-format handle the formatting.

// C array overload
template <typename T, std::size_t N>
auto range_begin(const T (&arr)[N]) -> const T* {
  return arr;
}
template <typename T, std::size_t N>
auto range_end(const T (&arr)[N]) -> const T* {
  return arr + N;
}

template <typename T, typename Enable = void>
struct has_member_fn_begin_end_t : std::false_type {};

template <typename T>
struct has_member_fn_begin_end_t<T, void_t<decltype(std::declval<T>().begin()),
                                           decltype(std::declval<T>().end())>>
    : std::true_type {};

// Member function overload
template <typename T>
auto range_begin(T&& rng) FMT_DECLTYPE_RETURN(static_cast<T&&>(rng).begin());
template <typename T>
auto range_end(T&& rng) FMT_DECLTYPE_RETURN(static_cast<T&&>(rng).end());

// ADL overload. Only participates in overload resolution if member functions
// are not found.
template <typename T>
auto range_begin(T&& rng)
    -> enable_if_t<!has_member_fn_begin_end_t<T&&>::value,
                   decltype(begin(static_cast<T&&>(rng)))> {
  return begin(static_cast<T&&>(rng));
}
template <typename T>
auto range_end(T&& rng) -> enable_if_t<!has_member_fn_begin_end_t<T&&>::value,
                                       decltype(end(static_cast<T&&>(rng)))> {
  return end(static_cast<T&&>(rng));
}

template <typename T, typename Enable = void>
struct has_const_begin_end : std::false_type {};
template <typename T, typename Enable = void>
struct has_mutable_begin_end : std::false_type {};

template <typename T>
struct has_const_begin_end<
    T,
    void_t<
        decltype(detail::range_begin(std::declval<const remove_cvref_t<T>&>())),
        decltype(detail::range_end(std::declval<const remove_cvref_t<T>&>()))>>
    : std::true_type {};

template <typename T>
struct has_mutable_begin_end<
    T, void_t<decltype(detail::range_begin(std::declval<T>())),
              decltype(detail::range_end(std::declval<T>())),
              enable_if_t<std::is_copy_constructible<T>::value>>>
    : std::true_type {};

template <typename T>
struct is_range_<T, void>
    : std::integral_constant<bool, (has_const_begin_end<T>::value ||
                                    has_mutable_begin_end<T>::value)> {};
#  undef FMT_DECLTYPE_RETURN
#endif

// tuple_size and tuple_element check.
template <typename T> class is_tuple_like_ {
  template <typename U>
  static auto check(U* p) -> decltype(std::tuple_size<U>::value, int());
  template <typename> static void check(...);

 public:
  static FMT_CONSTEXPR_DECL const bool value =
      !std::is_void<decltype(check<T>(nullptr))>::value;
};

// Check for integer_sequence
#if defined(__cpp_lib_integer_sequence) || FMT_MSC_VER >= 1900
template <typename T, T... N>
using integer_sequence = std::integer_sequence<T, N...>;
template <size_t... N> using index_sequence = std::index_sequence<N...>;
template <size_t N> using make_index_sequence = std::make_index_sequence<N>;
#else
template <typename T, T... N> struct integer_sequence {
  using value_type = T;

  static FMT_CONSTEXPR size_t size() { return sizeof...(N); }
};

template <size_t... N> using index_sequence = integer_sequence<size_t, N...>;

template <typename T, size_t N, T... Ns>
struct make_integer_sequence : make_integer_sequence<T, N - 1, N - 1, Ns...> {};
template <typename T, T... Ns>
struct make_integer_sequence<T, 0, Ns...> : integer_sequence<T, Ns...> {};

template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;
#endif

template <class Tuple, class F, size_t... Is>
void for_each(index_sequence<Is...>, Tuple&& tup, F&& f) FMT_NOEXCEPT {
  using std::get;
  // using free function get<I>(T) now.
  const int _[] = {0, ((void)f(get<Is>(tup)), 0)...};
  (void)_;  // blocks warnings
}

template <class T>
FMT_CONSTEXPR make_index_sequence<std::tuple_size<T>::value> get_indexes(
    T const&) {
  return {};
}

template <class Tuple, class F> void for_each(Tuple&& tup, F&& f) {
  const auto indexes = get_indexes(tup);
  for_each(indexes, std::forward<Tuple>(tup), std::forward<F>(f));
}

template <typename Range>
using value_type =
    remove_cvref_t<decltype(*detail::range_begin(std::declval<Range>()))>;

template <typename OutputIt> OutputIt write_delimiter(OutputIt out) {
  *out++ = ',';
  *out++ = ' ';
  return out;
}

struct singleton {
  unsigned char upper;
  unsigned char lower_count;
};

inline auto is_printable(uint16_t x, const singleton* singletons,
                         size_t singletons_size,
                         const unsigned char* singleton_lowers,
                         const unsigned char* normal, size_t normal_size)
    -> bool {
  auto upper = x >> 8;
  auto lower_start = 0;
  for (size_t i = 0; i < singletons_size; ++i) {
    auto s = singletons[i];
    auto lower_end = lower_start + s.lower_count;
    if (upper < s.upper) break;
    if (upper == s.upper) {
      for (auto j = lower_start; j < lower_end; ++j) {
        if (singleton_lowers[j] == (x & 0xff)) return false;
      }
    }
    lower_start = lower_end;
  }

  auto xsigned = static_cast<int>(x);
  auto current = true;
  for (size_t i = 0; i < normal_size; ++i) {
    auto v = static_cast<int>(normal[i]);
    auto len = (v & 0x80) != 0 ? (v & 0x7f) << 8 | normal[++i] : v;
    xsigned -= len;
    if (xsigned < 0) break;
    current = !current;
  }
  return current;
}

// Returns true iff the code point cp is printable.
// This code is generated by support/printable.py.
inline auto is_printable(uint32_t cp) -> bool {
  static constexpr singleton singletons0[] = {
      {0x00, 1},  {0x03, 5},  {0x05, 6},  {0x06, 3},  {0x07, 6},  {0x08, 8},
      {0x09, 17}, {0x0a, 28}, {0x0b, 25}, {0x0c, 20}, {0x0d, 16}, {0x0e, 13},
      {0x0f, 4},  {0x10, 3},  {0x12, 18}, {0x13, 9},  {0x16, 1},  {0x17, 5},
      {0x18, 2},  {0x19, 3},  {0x1a, 7},  {0x1c, 2},  {0x1d, 1},  {0x1f, 22},
      {0x20, 3},  {0x2b, 3},  {0x2c, 2},  {0x2d, 11}, {0x2e, 1},  {0x30, 3},
      {0x31, 2},  {0x32, 1},  {0xa7, 2},  {0xa9, 2},  {0xaa, 4},  {0xab, 8},
      {0xfa, 2},  {0xfb, 5},  {0xfd, 4},  {0xfe, 3},  {0xff, 9},
  };
  static constexpr unsigned char singletons0_lower[] = {
      0xad, 0x78, 0x79, 0x8b, 0x8d, 0xa2, 0x30, 0x57, 0x58, 0x8b, 0x8c, 0x90,
      0x1c, 0x1d, 0xdd, 0x0e, 0x0f, 0x4b, 0x4c, 0xfb, 0xfc, 0x2e, 0x2f, 0x3f,
      0x5c, 0x5d, 0x5f, 0xb5, 0xe2, 0x84, 0x8d, 0x8e, 0x91, 0x92, 0xa9, 0xb1,
      0xba, 0xbb, 0xc5, 0xc6, 0xc9, 0xca, 0xde, 0xe4, 0xe5, 0xff, 0x00, 0x04,
      0x11, 0x12, 0x29, 0x31, 0x34, 0x37, 0x3a, 0x3b, 0x3d, 0x49, 0x4a, 0x5d,
      0x84, 0x8e, 0x92, 0xa9, 0xb1, 0xb4, 0xba, 0xbb, 0xc6, 0xca, 0xce, 0xcf,
      0xe4, 0xe5, 0x00, 0x04, 0x0d, 0x0e, 0x11, 0x12, 0x29, 0x31, 0x34, 0x3a,
      0x3b, 0x45, 0x46, 0x49, 0x4a, 0x5e, 0x64, 0x65, 0x84, 0x91, 0x9b, 0x9d,
      0xc9, 0xce, 0xcf, 0x0d, 0x11, 0x29, 0x45, 0x49, 0x57, 0x64, 0x65, 0x8d,
      0x91, 0xa9, 0xb4, 0xba, 0xbb, 0xc5, 0xc9, 0xdf, 0xe4, 0xe5, 0xf0, 0x0d,
      0x11, 0x45, 0x49, 0x64, 0x65, 0x80, 0x84, 0xb2, 0xbc, 0xbe, 0xbf, 0xd5,
      0xd7, 0xf0, 0xf1, 0x83, 0x85, 0x8b, 0xa4, 0xa6, 0xbe, 0xbf, 0xc5, 0xc7,
      0xce, 0xcf, 0xda, 0xdb, 0x48, 0x98, 0xbd, 0xcd, 0xc6, 0xce, 0xcf, 0x49,
      0x4e, 0x4f, 0x57, 0x59, 0x5e, 0x5f, 0x89, 0x8e, 0x8f, 0xb1, 0xb6, 0xb7,
      0xbf, 0xc1, 0xc6, 0xc7, 0xd7, 0x11, 0x16, 0x17, 0x5b, 0x5c, 0xf6, 0xf7,
      0xfe, 0xff, 0x80, 0x0d, 0x6d, 0x71, 0xde, 0xdf, 0x0e, 0x0f, 0x1f, 0x6e,
      0x6f, 0x1c, 0x1d, 0x5f, 0x7d, 0x7e, 0xae, 0xaf, 0xbb, 0xbc, 0xfa, 0x16,
      0x17, 0x1e, 0x1f, 0x46, 0x47, 0x4e, 0x4f, 0x58, 0x5a, 0x5c, 0x5e, 0x7e,
      0x7f, 0xb5, 0xc5, 0xd4, 0xd5, 0xdc, 0xf0, 0xf1, 0xf5, 0x72, 0x73, 0x8f,
      0x74, 0x75, 0x96, 0x2f, 0x5f, 0x26, 0x2e, 0x2f, 0xa7, 0xaf, 0xb7, 0xbf,
      0xc7, 0xcf, 0xd7, 0xdf, 0x9a, 0x40, 0x97, 0x98, 0x30, 0x8f, 0x1f, 0xc0,
      0xc1, 0xce, 0xff, 0x4e, 0x4f, 0x5a, 0x5b, 0x07, 0x08, 0x0f, 0x10, 0x27,
      0x2f, 0xee, 0xef, 0x6e, 0x6f, 0x37, 0x3d, 0x3f, 0x42, 0x45, 0x90, 0x91,
      0xfe, 0xff, 0x53, 0x67, 0x75, 0xc8, 0xc9, 0xd0, 0xd1, 0xd8, 0xd9, 0xe7,
      0xfe, 0xff,
  };
  static constexpr singleton singletons1[] = {
      {0x00, 6},  {0x01, 1}, {0x03, 1},  {0x04, 2}, {0x08, 8},  {0x09, 2},
      {0x0a, 5},  {0x0b, 2}, {0x0e, 4},  {0x10, 1}, {0x11, 2},  {0x12, 5},
      {0x13, 17}, {0x14, 1}, {0x15, 2},  {0x17, 2}, {0x19, 13}, {0x1c, 5},
      {0x1d, 8},  {0x24, 1}, {0x6a, 3},  {0x6b, 2}, {0xbc, 2},  {0xd1, 2},
      {0xd4, 12}, {0xd5, 9}, {0xd6, 2},  {0xd7, 2}, {0xda, 1},  {0xe0, 5},
      {0xe1, 2},  {0xe8, 2}, {0xee, 32}, {0xf0, 4}, {0xf8, 2},  {0xf9, 2},
      {0xfa, 2},  {0xfb, 1},
  };
  static constexpr unsigned char singletons1_lower[] = {
      0x0c, 0x27, 0x3b, 0x3e, 0x4e, 0x4f, 0x8f, 0x9e, 0x9e, 0x9f, 0x06, 0x07,
      0x09, 0x36, 0x3d, 0x3e, 0x56, 0xf3, 0xd0, 0xd1, 0x04, 0x14, 0x18, 0x36,
      0x37, 0x56, 0x57, 0x7f, 0xaa, 0xae, 0xaf, 0xbd, 0x35, 0xe0, 0x12, 0x87,
      0x89, 0x8e, 0x9e, 0x04, 0x0d, 0x0e, 0x11, 0x12, 0x29, 0x31, 0x34, 0x3a,
      0x45, 0x46, 0x49, 0x4a, 0x4e, 0x4f, 0x64, 0x65, 0x5c, 0xb6, 0xb7, 0x1b,
      0x1c, 0x07, 0x08, 0x0a, 0x0b, 0x14, 0x17, 0x36, 0x39, 0x3a, 0xa8, 0xa9,
      0xd8, 0xd9, 0x09, 0x37, 0x90, 0x91, 0xa8, 0x07, 0x0a, 0x3b, 0x3e, 0x66,
      0x69, 0x8f, 0x92, 0x6f, 0x5f, 0xee, 0xef, 0x5a, 0x62, 0x9a, 0x9b, 0x27,
      0x28, 0x55, 0x9d, 0xa0, 0xa1, 0xa3, 0xa4, 0xa7, 0xa8, 0xad, 0xba, 0xbc,
      0xc4, 0x06, 0x0b, 0x0c, 0x15, 0x1d, 0x3a, 0x3f, 0x45, 0x51, 0xa6, 0xa7,
      0xcc, 0xcd, 0xa0, 0x07, 0x19, 0x1a, 0x22, 0x25, 0x3e, 0x3f, 0xc5, 0xc6,
      0x04, 0x20, 0x23, 0x25, 0x26, 0x28, 0x33, 0x38, 0x3a, 0x48, 0x4a, 0x4c,
      0x50, 0x53, 0x55, 0x56, 0x58, 0x5a, 0x5c, 0x5e, 0x60, 0x63, 0x65, 0x66,
      0x6b, 0x73, 0x78, 0x7d, 0x7f, 0x8a, 0xa4, 0xaa, 0xaf, 0xb0, 0xc0, 0xd0,
      0xae, 0xaf, 0x79, 0xcc, 0x6e, 0x6f, 0x93,
  };
  static constexpr unsigned char normal0[] = {
      0x00, 0x20, 0x5f, 0x22, 0x82, 0xdf, 0x04, 0x82, 0x44, 0x08, 0x1b, 0x04,
      0x06, 0x11, 0x81, 0xac, 0x0e, 0x80, 0xab, 0x35, 0x28, 0x0b, 0x80, 0xe0,
      0x03, 0x19, 0x08, 0x01, 0x04, 0x2f, 0x04, 0x34, 0x04, 0x07, 0x03, 0x01,
      0x07, 0x06, 0x07, 0x11, 0x0a, 0x50, 0x0f, 0x12, 0x07, 0x55, 0x07, 0x03,
      0x04, 0x1c, 0x0a, 0x09, 0x03, 0x08, 0x03, 0x07, 0x03, 0x02, 0x03, 0x03,
      0x03, 0x0c, 0x04, 0x05, 0x03, 0x0b, 0x06, 0x01, 0x0e, 0x15, 0x05, 0x3a,
      0x03, 0x11, 0x07, 0x06, 0x05, 0x10, 0x07, 0x57, 0x07, 0x02, 0x07, 0x15,
      0x0d, 0x50, 0x04, 0x43, 0x03, 0x2d, 0x03, 0x01, 0x04, 0x11, 0x06, 0x0f,
      0x0c, 0x3a, 0x04, 0x1d, 0x25, 0x5f, 0x20, 0x6d, 0x04, 0x6a, 0x25, 0x80,
      0xc8, 0x05, 0x82, 0xb0, 0x03, 0x1a, 0x06, 0x82, 0xfd, 0x03, 0x59, 0x07,
      0x15, 0x0b, 0x17, 0x09, 0x14, 0x0c, 0x14, 0x0c, 0x6a, 0x06, 0x0a, 0x06,
      0x1a, 0x06, 0x59, 0x07, 0x2b, 0x05, 0x46, 0x0a, 0x2c, 0x04, 0x0c, 0x04,
      0x01, 0x03, 0x31, 0x0b, 0x2c, 0x04, 0x1a, 0x06, 0x0b, 0x03, 0x80, 0xac,
      0x06, 0x0a, 0x06, 0x21, 0x3f, 0x4c, 0x04, 0x2d, 0x03, 0x74, 0x08, 0x3c,
      0x03, 0x0f, 0x03, 0x3c, 0x07, 0x38, 0x08, 0x2b, 0x05, 0x82, 0xff, 0x11,
      0x18, 0x08, 0x2f, 0x11, 0x2d, 0x03, 0x20, 0x10, 0x21, 0x0f, 0x80, 0x8c,
      0x04, 0x82, 0x97, 0x19, 0x0b, 0x15, 0x88, 0x94, 0x05, 0x2f, 0x05, 0x3b,
      0x07, 0x02, 0x0e, 0x18, 0x09, 0x80, 0xb3, 0x2d, 0x74, 0x0c, 0x80, 0xd6,
      0x1a, 0x0c, 0x05, 0x80, 0xff, 0x05, 0x80, 0xdf, 0x0c, 0xee, 0x0d, 0x03,
      0x84, 0x8d, 0x03, 0x37, 0x09, 0x81, 0x5c, 0x14, 0x80, 0xb8, 0x08, 0x80,
      0xcb, 0x2a, 0x38, 0x03, 0x0a, 0x06, 0x38, 0x08, 0x46, 0x08, 0x0c, 0x06,
      0x74, 0x0b, 0x1e, 0x03, 0x5a, 0x04, 0x59, 0x09, 0x80, 0x83, 0x18, 0x1c,
      0x0a, 0x16, 0x09, 0x4c, 0x04, 0x80, 0x8a, 0x06, 0xab, 0xa4, 0x0c, 0x17,
      0x04, 0x31, 0xa1, 0x04, 0x81, 0xda, 0x26, 0x07, 0x0c, 0x05, 0x05, 0x80,
      0xa5, 0x11, 0x81, 0x6d, 0x10, 0x78, 0x28, 0x2a, 0x06, 0x4c, 0x04, 0x80,
      0x8d, 0x04, 0x80, 0xbe, 0x03, 0x1b, 0x03, 0x0f, 0x0d,
  };
  static constexpr unsigned char normal1[] = {
      0x5e, 0x22, 0x7b, 0x05, 0x03, 0x04, 0x2d, 0x03, 0x66, 0x03, 0x01, 0x2f,
      0x2e, 0x80, 0x82, 0x1d, 0x03, 0x31, 0x0f, 0x1c, 0x04, 0x24, 0x09, 0x1e,
      0x05, 0x2b, 0x05, 0x44, 0x04, 0x0e, 0x2a, 0x80, 0xaa, 0x06, 0x24, 0x04,
      0x24, 0x04, 0x28, 0x08, 0x34, 0x0b, 0x01, 0x80, 0x90, 0x81, 0x37, 0x09,
      0x16, 0x0a, 0x08, 0x80, 0x98, 0x39, 0x03, 0x63, 0x08, 0x09, 0x30, 0x16,
      0x05, 0x21, 0x03, 0x1b, 0x05, 0x01, 0x40, 0x38, 0x04, 0x4b, 0x05, 0x2f,
      0x04, 0x0a, 0x07, 0x09, 0x07, 0x40, 0x20, 0x27, 0x04, 0x0c, 0x09, 0x36,
      0x03, 0x3a, 0x05, 0x1a, 0x07, 0x04, 0x0c, 0x07, 0x50, 0x49, 0x37, 0x33,
      0x0d, 0x33, 0x07, 0x2e, 0x08, 0x0a, 0x81, 0x26, 0x52, 0x4e, 0x28, 0x08,
      0x2a, 0x56, 0x1c, 0x14, 0x17, 0x09, 0x4e, 0x04, 0x1e, 0x0f, 0x43, 0x0e,
      0x19, 0x07, 0x0a, 0x06, 0x48, 0x08, 0x27, 0x09, 0x75, 0x0b, 0x3f, 0x41,
      0x2a, 0x06, 0x3b, 0x05, 0x0a, 0x06, 0x51, 0x06, 0x01, 0x05, 0x10, 0x03,
      0x05, 0x80, 0x8b, 0x62, 0x1e, 0x48, 0x08, 0x0a, 0x80, 0xa6, 0x5e, 0x22,
      0x45, 0x0b, 0x0a, 0x06, 0x0d, 0x13, 0x39, 0x07, 0x0a, 0x36, 0x2c, 0x04,
      0x10, 0x80, 0xc0, 0x3c, 0x64, 0x53, 0x0c, 0x48, 0x09, 0x0a, 0x46, 0x45,
      0x1b, 0x48, 0x08, 0x53, 0x1d, 0x39, 0x81, 0x07, 0x46, 0x0a, 0x1d, 0x03,
      0x47, 0x49, 0x37, 0x03, 0x0e, 0x08, 0x0a, 0x06, 0x39, 0x07, 0x0a, 0x81,
      0x36, 0x19, 0x80, 0xb7, 0x01, 0x0f, 0x32, 0x0d, 0x83, 0x9b, 0x66, 0x75,
      0x0b, 0x80, 0xc4, 0x8a, 0xbc, 0x84, 0x2f, 0x8f, 0xd1, 0x82, 0x47, 0xa1,
      0xb9, 0x82, 0x39, 0x07, 0x2a, 0x04, 0x02, 0x60, 0x26, 0x0a, 0x46, 0x0a,
      0x28, 0x05, 0x13, 0x82, 0xb0, 0x5b, 0x65, 0x4b, 0x04, 0x39, 0x07, 0x11,
      0x40, 0x05, 0x0b, 0x02, 0x0e, 0x97, 0xf8, 0x08, 0x84, 0xd6, 0x2a, 0x09,
      0xa2, 0xf7, 0x81, 0x1f, 0x31, 0x03, 0x11, 0x04, 0x08, 0x81, 0x8c, 0x89,
      0x04, 0x6b, 0x05, 0x0d, 0x03, 0x09, 0x07, 0x10, 0x93, 0x60, 0x80, 0xf6,
      0x0a, 0x73, 0x08, 0x6e, 0x17, 0x46, 0x80, 0x9a, 0x14, 0x0c, 0x57, 0x09,
      0x19, 0x80, 0x87, 0x81, 0x47, 0x03, 0x85, 0x42, 0x0f, 0x15, 0x85, 0x50,
      0x2b, 0x80, 0xd5, 0x2d, 0x03, 0x1a, 0x04, 0x02, 0x81, 0x70, 0x3a, 0x05,
      0x01, 0x85, 0x00, 0x80, 0xd7, 0x29, 0x4c, 0x04, 0x0a, 0x04, 0x02, 0x83,
      0x11, 0x44, 0x4c, 0x3d, 0x80, 0xc2, 0x3c, 0x06, 0x01, 0x04, 0x55, 0x05,
      0x1b, 0x34, 0x02, 0x81, 0x0e, 0x2c, 0x04, 0x64, 0x0c, 0x56, 0x0a, 0x80,
      0xae, 0x38, 0x1d, 0x0d, 0x2c, 0x04, 0x09, 0x07, 0x02, 0x0e, 0x06, 0x80,
      0x9a, 0x83, 0xd8, 0x08, 0x0d, 0x03, 0x0d, 0x03, 0x74, 0x0c, 0x59, 0x07,
      0x0c, 0x14, 0x0c, 0x04, 0x38, 0x08, 0x0a, 0x06, 0x28, 0x08, 0x22, 0x4e,
      0x81, 0x54, 0x0c, 0x15, 0x03, 0x03, 0x05, 0x07, 0x09, 0x19, 0x07, 0x07,
      0x09, 0x03, 0x0d, 0x07, 0x29, 0x80, 0xcb, 0x25, 0x0a, 0x84, 0x06,
  };
  auto lower = static_cast<uint16_t>(cp);
  if (cp < 0x10000) {
    return is_printable(lower, singletons0,
                        sizeof(singletons0) / sizeof(*singletons0),
                        singletons0_lower, normal0, sizeof(normal0));
  }
  if (cp < 0x20000) {
    return is_printable(lower, singletons1,
                        sizeof(singletons1) / sizeof(*singletons1),
                        singletons1_lower, normal1, sizeof(normal1));
  }
  if (0x2a6de <= cp && cp < 0x2a700) return false;
  if (0x2b735 <= cp && cp < 0x2b740) return false;
  if (0x2b81e <= cp && cp < 0x2b820) return false;
  if (0x2cea2 <= cp && cp < 0x2ceb0) return false;
  if (0x2ebe1 <= cp && cp < 0x2f800) return false;
  if (0x2fa1e <= cp && cp < 0x30000) return false;
  if (0x3134b <= cp && cp < 0xe0100) return false;
  if (0xe01f0 <= cp && cp < 0x110000) return false;
  return cp < 0x110000;
}

inline auto needs_escape(uint32_t cp) -> bool {
  return cp < 0x20 || cp == 0x7f || cp == '"' || cp == '\\' ||
         !is_printable(cp);
}

template <typename Char> struct find_escape_result {
  const Char* begin;
  const Char* end;
  uint32_t cp;
};

template <typename Char>
auto find_escape(const Char* begin, const Char* end)
    -> find_escape_result<Char> {
  for (; begin != end; ++begin) {
    auto cp = static_cast<typename std::make_unsigned<Char>::type>(*begin);
    if (sizeof(Char) == 1 && cp >= 0x80) continue;
    if (needs_escape(cp)) return {begin, begin + 1, cp};
  }
  return {begin, nullptr, 0};
}

inline auto find_escape(const char* begin, const char* end)
    -> find_escape_result<char> {
  if (!is_utf8()) return find_escape<char>(begin, end);
  auto result = find_escape_result<char>{end, nullptr, 0};
  for_each_codepoint(string_view(begin, to_unsigned(end - begin)),
                     [&](uint32_t cp, string_view sv) {
                       if (needs_escape(cp)) {
                         result = {sv.begin(), sv.end(), cp};
                         return false;
                       }
                       return true;
                     });
  return result;
}

template <typename Char, typename OutputIt>
auto write_range_entry(OutputIt out, basic_string_view<Char> str) -> OutputIt {
  *out++ = '"';
  auto begin = str.begin(), end = str.end();
  do {
    auto escape = find_escape(begin, end);
    out = copy_str<Char>(begin, escape.begin, out);
    begin = escape.end;
    if (!begin) break;
    auto c = static_cast<Char>(escape.cp);
    switch (escape.cp) {
    case '\n':
      *out++ = '\\';
      c = 'n';
      break;
    case '\r':
      *out++ = '\\';
      c = 'r';
      break;
    case '\t':
      *out++ = '\\';
      c = 't';
      break;
    case '"':
      FMT_FALLTHROUGH;
    case '\\':
      *out++ = '\\';
      break;
    default:
      if (is_utf8()) {
        if (escape.cp < 0x100) {
          out = format_to(out, "\\x{:02x}", escape.cp);
          continue;
        }
        if (escape.cp < 0x10000) {
          out = format_to(out, "\\u{:04x}", escape.cp);
          continue;
        }
        if (escape.cp < 0x110000) {
          out = format_to(out, "\\U{:08x}", escape.cp);
          continue;
        }
      }
      for (Char escape_char : basic_string_view<Char>(
               escape.begin, to_unsigned(escape.end - escape.begin))) {
        out = format_to(
            out, "\\x{:02x}",
            static_cast<typename std::make_unsigned<Char>::type>(escape_char));
      }
      continue;
    }
    *out++ = c;
  } while (begin != end);
  *out++ = '"';
  return out;
}

template <typename Char, typename OutputIt, typename T,
          FMT_ENABLE_IF(std::is_convertible<T, std_string_view<char>>::value)>
inline auto write_range_entry(OutputIt out, const T& str) -> OutputIt {
  auto sv = std_string_view<Char>(str);
  return write_range_entry<Char>(out, basic_string_view<Char>(sv));
}

template <typename Char, typename OutputIt, typename Arg,
          FMT_ENABLE_IF(std::is_same<Arg, Char>::value)>
OutputIt write_range_entry(OutputIt out, const Arg v) {
  *out++ = '\'';
  *out++ = v;
  *out++ = '\'';
  return out;
}

template <
    typename Char, typename OutputIt, typename Arg,
    FMT_ENABLE_IF(!is_std_string_like<typename std::decay<Arg>::type>::value &&
                  !std::is_same<Arg, Char>::value)>
OutputIt write_range_entry(OutputIt out, const Arg& v) {
  return write<Char>(out, v);
}

}  // namespace detail

template <typename T> struct is_tuple_like {
  static FMT_CONSTEXPR_DECL const bool value =
      detail::is_tuple_like_<T>::value && !detail::is_range_<T>::value;
};

template <typename TupleT, typename Char>
struct formatter<TupleT, Char, enable_if_t<fmt::is_tuple_like<TupleT>::value>> {
 private:
  // C++11 generic lambda for format().
  template <typename FormatContext> struct format_each {
    template <typename T> void operator()(const T& v) {
      if (i > 0) out = detail::write_delimiter(out);
      out = detail::write_range_entry<Char>(out, v);
      ++i;
    }
    int i;
    typename FormatContext::iterator& out;
  };

 public:
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  template <typename FormatContext = format_context>
  auto format(const TupleT& values, FormatContext& ctx) -> decltype(ctx.out()) {
    auto out = ctx.out();
    *out++ = '(';
    detail::for_each(values, format_each<FormatContext>{0, out});
    *out++ = ')';
    return out;
  }
};

template <typename T, typename Char> struct is_range {
  static FMT_CONSTEXPR_DECL const bool value =
      detail::is_range_<T>::value && !detail::is_std_string_like<T>::value &&
      !detail::is_map<T>::value &&
      !std::is_convertible<T, std::basic_string<Char>>::value &&
      !std::is_constructible<detail::std_string_view<Char>, T>::value;
};

template <typename T, typename Char>
struct formatter<
    T, Char,
    enable_if_t<
        fmt::is_range<T, Char>::value
// Workaround a bug in MSVC 2019 and earlier.
#if !FMT_MSC_VER
        && (is_formattable<detail::value_type<T>, Char>::value ||
            detail::has_fallback_formatter<detail::value_type<T>, Char>::value)
#endif
        >> {
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  template <
      typename FormatContext, typename U,
      FMT_ENABLE_IF(
          std::is_same<U, conditional_t<detail::has_const_begin_end<T>::value,
                                        const T, T>>::value)>
  auto format(U& range, FormatContext& ctx) -> decltype(ctx.out()) {
#ifdef FMT_DEPRECATED_BRACED_RANGES
    Char prefix = '{';
    Char postfix = '}';
#else
    Char prefix = detail::is_set<T>::value ? '{' : '[';
    Char postfix = detail::is_set<T>::value ? '}' : ']';
#endif
    auto out = ctx.out();
    *out++ = prefix;
    int i = 0;
    auto it = std::begin(range);
    auto end = std::end(range);
    for (; it != end; ++it) {
      if (i > 0) out = detail::write_delimiter(out);
      out = detail::write_range_entry<Char>(out, *it);
      ++i;
    }
    *out++ = postfix;
    return out;
  }
};

template <typename T, typename Char>
struct formatter<
    T, Char,
    enable_if_t<
        detail::is_map<T>::value
// Workaround a bug in MSVC 2019 and earlier.
#if !FMT_MSC_VER
        && (is_formattable<detail::value_type<T>, Char>::value ||
            detail::has_fallback_formatter<detail::value_type<T>, Char>::value)
#endif
        >> {
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  template <
      typename FormatContext, typename U,
      FMT_ENABLE_IF(
          std::is_same<U, conditional_t<detail::has_const_begin_end<T>::value,
                                        const T, T>>::value)>
  auto format(U& map, FormatContext& ctx) -> decltype(ctx.out()) {
    auto out = ctx.out();
    *out++ = '{';
    int i = 0;
    for (const auto& item : map) {
      if (i > 0) out = detail::write_delimiter(out);
      out = detail::write_range_entry<Char>(out, item.first);
      *out++ = ':';
      *out++ = ' ';
      out = detail::write_range_entry<Char>(out, item.second);
      ++i;
    }
    *out++ = '}';
    return out;
  }
};

template <typename Char, typename... T> struct tuple_join_view : detail::view {
  const std::tuple<T...>& tuple;
  basic_string_view<Char> sep;

  tuple_join_view(const std::tuple<T...>& t, basic_string_view<Char> s)
      : tuple(t), sep{s} {}
};

template <typename Char, typename... T>
using tuple_arg_join = tuple_join_view<Char, T...>;

// Define FMT_TUPLE_JOIN_SPECIFIERS to enable experimental format specifiers
// support in tuple_join. It is disabled by default because of issues with
// the dynamic width and precision.
#ifndef FMT_TUPLE_JOIN_SPECIFIERS
#  define FMT_TUPLE_JOIN_SPECIFIERS 0
#endif

template <typename Char, typename... T>
struct formatter<tuple_join_view<Char, T...>, Char> {
  template <typename ParseContext>
  FMT_CONSTEXPR auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    return do_parse(ctx, std::integral_constant<size_t, sizeof...(T)>());
  }

  template <typename FormatContext>
  auto format(const tuple_join_view<Char, T...>& value,
              FormatContext& ctx) const -> typename FormatContext::iterator {
    return do_format(value, ctx,
                     std::integral_constant<size_t, sizeof...(T)>());
  }

 private:
  std::tuple<formatter<typename std::decay<T>::type, Char>...> formatters_;

  template <typename ParseContext>
  FMT_CONSTEXPR auto do_parse(ParseContext& ctx,
                              std::integral_constant<size_t, 0>)
      -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  template <typename ParseContext, size_t N>
  FMT_CONSTEXPR auto do_parse(ParseContext& ctx,
                              std::integral_constant<size_t, N>)
      -> decltype(ctx.begin()) {
    auto end = ctx.begin();
#if FMT_TUPLE_JOIN_SPECIFIERS
    end = std::get<sizeof...(T) - N>(formatters_).parse(ctx);
    if (N > 1) {
      auto end1 = do_parse(ctx, std::integral_constant<size_t, N - 1>());
      if (end != end1)
        FMT_THROW(format_error("incompatible format specs for tuple elements"));
    }
#endif
    return end;
  }

  template <typename FormatContext>
  auto do_format(const tuple_join_view<Char, T...>&, FormatContext& ctx,
                 std::integral_constant<size_t, 0>) const ->
      typename FormatContext::iterator {
    return ctx.out();
  }

  template <typename FormatContext, size_t N>
  auto do_format(const tuple_join_view<Char, T...>& value, FormatContext& ctx,
                 std::integral_constant<size_t, N>) const ->
      typename FormatContext::iterator {
    auto out = std::get<sizeof...(T) - N>(formatters_)
                   .format(std::get<sizeof...(T) - N>(value.tuple), ctx);
    if (N > 1) {
      out = std::copy(value.sep.begin(), value.sep.end(), out);
      ctx.advance_to(out);
      return do_format(value, ctx, std::integral_constant<size_t, N - 1>());
    }
    return out;
  }
};

FMT_MODULE_EXPORT_BEGIN

/**
  \rst
  Returns an object that formats `tuple` with elements separated by `sep`.

  **Example**::

    std::tuple<int, char> t = {1, 'a'};
    fmt::print("{}", fmt::join(t, ", "));
    // Output: "1, a"
  \endrst
 */
template <typename... T>
FMT_CONSTEXPR auto join(const std::tuple<T...>& tuple, string_view sep)
    -> tuple_join_view<char, T...> {
  return {tuple, sep};
}

template <typename... T>
FMT_CONSTEXPR auto join(const std::tuple<T...>& tuple,
                        basic_string_view<wchar_t> sep)
    -> tuple_join_view<wchar_t, T...> {
  return {tuple, sep};
}

/**
  \rst
  Returns an object that formats `initializer_list` with elements separated by
  `sep`.

  **Example**::

    fmt::print("{}", fmt::join({1, 2, 3}, ", "));
    // Output: "1, 2, 3"
  \endrst
 */
template <typename T>
auto join(std::initializer_list<T> list, string_view sep)
    -> join_view<const T*, const T*> {
  return join(std::begin(list), std::end(list), sep);
}

FMT_MODULE_EXPORT_END
FMT_END_NAMESPACE

#endif  // FMT_RANGES_H_
