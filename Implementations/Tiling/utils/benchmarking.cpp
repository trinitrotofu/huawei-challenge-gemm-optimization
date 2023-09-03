#include <sys/time.h>

using namespace std;

class Stopwatch {
    private:
        struct timeval start_time, end_time;
        bool running = false;
        double time = 0.f;
    public:
        void reset() {
            time = 0.f;
            running = false;
        }
        void start() {
            if (!running) {
                gettimeofday(&start_time, nullptr);
                running = true;
            }
        }
        void stop() {
            if (running) {
                gettimeofday(&end_time, nullptr);
                time += (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
                running = false;
            }
        }
        double get_time() {
            if (running) {
                gettimeofday(&end_time, nullptr);
                return time + (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
            }
            return time;
        }
};
