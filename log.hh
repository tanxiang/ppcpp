
#pragma once
#include <iostream>
#include <sstream>

namespace tt {

enum class LogLevel {
    STD_INFO,
    STD_ERROR,
    STD_MSG,

};

class ss {
private:
    std::ostringstream m_ss;
    const LogLevel m_logLevel;

public:
    ss(LogLevel Xi_logLevel)
        : m_logLevel(Xi_logLevel) {};

    ~ss()
    {
        switch (m_logLevel) {
        case LogLevel::STD_ERROR:
            std::cerr << "\033[0;31m" << m_ss.str() << "\033[0m" << std::endl;
            break;
        case LogLevel::STD_INFO:
            std::cout << m_ss.str() << std::endl;
            break;
        case LogLevel::STD_MSG:
            std::cout << "\033[0;32m" << m_ss.str() << "\033[0m" << std::endl;
            break;
        }
    }

    template <typename T>
    ss& operator<<(T const& Xi_val)
    {
        m_ss << Xi_val;
        return *this;
    }
};

class ProgressBar {
public:
    ProgressBar(const char finish = '#', const char unfini = '.')
        : _flags("-\\/"),
          _finish(finish),
          _progress_str(100, unfini),
          _cur_progress(0) {}

    void print_bar(const ushort n) {
        if (_cur_progress != 0 && n <= _cur_progress) {
            std::cerr << "\e[31merror\e[m: n(" << n 
                     << ") should > _cur_progress(" 
                     << _cur_progress << ")" << std::endl; 
            return ;
        }
        for (ushort i = _cur_progress; i < n; i++) {
            _progress_str[i] = _finish;
        }
        _cur_progress = n;
        std::string f, p;
        if (n == 100) {
            f = "\e[1;32mOK\e[m";
            p = "\e[1;32m100%\e[m";
        } else {
            f = _flags[n % 4];
            p = std::to_string(n) + '%';
        }
        std::cout << std::unitbuf
                  << '[' << f << ']'
                  << '[' << _progress_str << ']'
                  << '[' << p << "]" << '\r';
        if (n >= 100) {
            std::cout << std::endl;
        }
    }
private:
    std::string _flags;
    std::string _progress_str;
    ushort _cur_progress;
    char _finish;
};


}

#define ALOG(LOG_LEVEL) tt::ss(tt::LogLevel::STD_##LOG_LEVEL) << __FILE__ << '#' << __LINE__ << " : "
