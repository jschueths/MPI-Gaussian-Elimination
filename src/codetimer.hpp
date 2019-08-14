#ifndef CODETIMER_HPP
#define CODETIMER_HPP

#include <chrono>

class CodeTimer
{
    public:
        CodeTimer();
        void start();
        void stop();
        std::chrono::duration<double> duration() const;

    private:
        std::chrono::system_clock::time_point m_start;
        std::chrono::system_clock::time_point m_end;
};

#endif // CODETIMER_HPP
