#include <iostream>
#include <fstream>
#include <regex>
#include <cmath>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#include <fastjet/PseudoJet.hh>
#include "fastjet/ClusterSequence.hh"
#include "fastjet/contrib/RecursiveSoftDrop.hh"
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/tokenize.hpp>
#pragma GCC diagnostic pop

#include <cxxopts.hpp>

using namespace fastjet;
using namespace ranges;

PseudoJet Convert(double pt, double eta, double phi) {
    double px = pt*cos(phi);
    double py = pt*sin(phi);
    double pz = pt*sinh(eta);
    double E = pt*cosh(eta);
    return PseudoJet(px, py, pz, E);
}

size_t FindMaxSize(auto &&event, JetDefinition &jet_def) {
    static constexpr double ptmin = 30;
    static size_t max;

    ClusterSequence cs(event, jet_def);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(ptmin));

    for(size_t i = 0; i < 2; ++i) {
        auto constituents = jets[i].constituents();
        if(constituents.size() > max) {
            max = constituents.size();
        }
    }

    return max;
}

void ProcessEvent(auto &&event, JetDefinition &jet_def, std::ostream& out, size_t num) {
    static constexpr double ptmin = 30;
    static size_t min = 100;
    static size_t max = 0;
    static size_t count = 0;
    ClusterSequence cs(event, jet_def);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(ptmin));

    double phi1 = jets[0].phi(), phi2 = jets[1].phi();
    double dphi = std::acos(std::max(std::min((jets[0].px()*jets[1].px()+jets[0].py()*jets[1].py())/(jets[0].pt()*jets[1].pt()),1.0),-1.0));
    std::array<double, 2> shift{-phi1, -phi2};
    if(phi1 > phi2) {
        shift[0] += dphi/2;
        shift[1] -= dphi/2;
    } else {
        shift[0] -= dphi/2;
        shift[1] += dphi/2;
    }

    for(size_t i = 0; i < 2; ++i) {
        auto constituents = jets[i].constituents();
        if(constituents.size() <= 2) count++;
        if(constituents.size() <= min) {
            min = constituents.size();
            std::cout << "New min: " << min << " at " << num << " : " << count << std::endl;
        }
        if(constituents.size() > max) {
            max = constituents.size();
            std::cout << "New max: " << max << std::endl;
        }
        std::sort(constituents.begin(), constituents.end(),
                  [](PseudoJet a, PseudoJet b) { return a.pt() > b.pt(); }); 
        for(auto part : constituents) {
            double px = cos(shift[i])*part.px() - sin(shift[i])*part.py();
            double py = cos(shift[i])*part.py() + sin(shift[i])*part.px();
            part.reset_momentum(px, py, part.pz(), part.E());
            double phi = part.phi();
            out << part.pt() << "," << part.rap() << "," << phi << ",";
        }
    }
}

//----------------------------------------------------------------------
int main(int argc, char** argv){
    cxxopts::Options options("Rotate Events",
                             "Rotate events to ensure that phi=0 is between the two jets");
    options.add_options()
        ("i,input", "Filename to read from", cxxopts::value<std::string>())
        ("o,output", "Output file name", cxxopts::value<std::string>())
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if(!result.count("input")) {
        std::cout << "Program requires the argument --input or -i" << std::endl;
        return 1;
    }
    std::string input = result["input"].as<std::string>();

    if(!result.count("output")) {
        std::cout << "Program requires the argument --output or -o" << std::endl;
        return 1;
    }
    std::string output = result["output"].as<std::string>();

    static constexpr double R = 1.0;
    JetDefinition jet_def(antikt_algorithm, R);

    std::ifstream in(input.c_str());
    std::ofstream out(output.c_str());

    // Parse input
    std::string line;
    const auto re = std::regex{"[-+]?[\\d.]+(?:e[-+]?\\d+)?"};
    size_t num = 0;
    size_t max_size = 0;
    while(std::getline(in, line)) {
        if(num++ % 1000 == 0) {
            std::cout << "Processing event " << num << "\r";
            std::flush(std::cout);
        }
        auto tmp = line | views::tokenize(re) 
                          | views::transform([](const auto &v){ return std::stod(v); })
                          | to<std::vector>();
        auto event = tmp | views::chunk(3) 
                         | views::transform([](const auto &r) { return Convert(r[0], r[1], r[2]); })
                         | to<std::vector>();
        max_size = FindMaxSize(event, jet_def);
    }

    std::cout << "Max size = " << max_size << std::endl; 
}
