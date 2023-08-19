
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

}

#define ALOG(LOG_LEVEL) tt::ss(tt::LogLevel::STD_##LOG_LEVEL) << __FILE__ << '#' << __LINE__ << " : "
