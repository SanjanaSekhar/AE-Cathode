#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/tokenize.hpp>
#pragma GCC diagnostic pop

#include <regex>
#include <fstream>
#include <iostream>

using namespace ranges;

int main(){
    std::ifstream in("events_rotate.csv");

    std::string line;
    const auto re = std::regex{"[-+]?[\\d.]+(?:e[-+]?\\d+)?"};
    while(std::getline(in, line)) {
        auto tmp = line | views::tokenize(re) 
                          | views::transform([](const auto &v){ return std::stod(v); })
                          | to<std::vector>();
        auto event = tmp | views::chunk(3) | to<std::vector>();
        std::cout << event.size() << std::endl;
    }
}
