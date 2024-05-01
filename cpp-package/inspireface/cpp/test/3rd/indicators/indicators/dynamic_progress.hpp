
#ifndef INDICATORS_DYNAMIC_PROGRESS
#define INDICATORS_DYNAMIC_PROGRESS

#include <atomic>
#include <functional>
#include <indicators/color.hpp>
#include <indicators/setting.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/cursor_movement.hpp>
#include <indicators/details/stream_helper.hpp>
#include <iostream>
#include <mutex>
#include <vector>

namespace indicators {

template <typename Indicator> class DynamicProgress {
  using Settings = std::tuple<option::HideBarWhenComplete>;

public:
  template <typename... Indicators> explicit DynamicProgress(Indicators &... bars) {
    bars_ = {bars...};
    for (auto &bar : bars_) {
      bar.get().multi_progress_mode_ = true;
      ++total_count_;
      ++incomplete_count_;
    }
  }

  Indicator &operator[](size_t index) {
    print_progress();
    std::lock_guard<std::mutex> lock{mutex_};
    return bars_[index].get();
  }

  size_t push_back(Indicator &bar) {
    std::lock_guard<std::mutex> lock{mutex_};
    bar.multi_progress_mode_ = true;
    bars_.push_back(bar);
    return bars_.size() - 1;
  }

  template <typename T, details::ProgressBarOption id>
  void set_option(details::Setting<T, id> &&setting) {
    static_assert(!std::is_same<T, typename std::decay<decltype(details::get_value<id>(
                                       std::declval<Settings>()))>::type>::value,
                  "Setting has wrong type!");
    std::lock_guard<std::mutex> lock(mutex_);
    get_value<id>() = std::move(setting).value;
  }

  template <typename T, details::ProgressBarOption id>
  void set_option(const details::Setting<T, id> &setting) {
    static_assert(!std::is_same<T, typename std::decay<decltype(details::get_value<id>(
                                       std::declval<Settings>()))>::type>::value,
                  "Setting has wrong type!");
    std::lock_guard<std::mutex> lock(mutex_);
    get_value<id>() = setting.value;
  }

private:
  Settings settings_;
  std::atomic<bool> started_{false};
  std::mutex mutex_;
  std::vector<std::reference_wrapper<Indicator>> bars_;
  std::atomic<size_t> total_count_{0};
  std::atomic<size_t> incomplete_count_{0};

  template <details::ProgressBarOption id>
  auto get_value() -> decltype((details::get_value<id>(std::declval<Settings &>()).value)) {
    return details::get_value<id>(settings_).value;
  }

  template <details::ProgressBarOption id>
  auto get_value() const
      -> decltype((details::get_value<id>(std::declval<const Settings &>()).value)) {
    return details::get_value<id>(settings_).value;
  }

public:
  void print_progress() {
    std::lock_guard<std::mutex> lock{mutex_};
    auto &hide_bar_when_complete = get_value<details::ProgressBarOption::hide_bar_when_complete>();
    if (hide_bar_when_complete) {
      // Hide completed bars
      if (started_) {
        for (size_t i = 0; i < incomplete_count_; ++i) {
          move_up(1);
          erase_line();
          std::cout << std::flush;
        }
      }
      incomplete_count_ = 0;
      for (auto &bar : bars_) {
        if (!bar.get().is_completed()) {
          bar.get().print_progress(true);
          std::cout << "\n";
          ++incomplete_count_;
        }
      }
      if (!started_)
        started_ = true;
    } else {
      // Don't hide any bars
      if (started_)
        move_up(static_cast<int>(total_count_));
      for (auto &bar : bars_) {
        bar.get().print_progress(true);
        std::cout << "\n";
      }
      if (!started_)
        started_ = true;
    }
    total_count_ = bars_.size();
    std::cout << termcolor::reset;
  }
};

} // namespace indicators

#endif