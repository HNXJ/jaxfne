// Reads lines of whitespace-separated keys; inserts each line's keys in order into a fresh
// std::unordered_map<size_t, double> with operator[] (as WeightMap::addWeight does) and prints
// the iteration order, one line per input line.
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::unordered_map<size_t, double> m;
        std::istringstream in(line);
        size_t k;
        while (in >> k) m[k] = 1;
        bool first = true;
        for (auto const &e : m) {
            std::cout << (first ? "" : " ") << e.first;
            first = false;
        }
        std::cout << "\n";
    }
    return 0;
}
