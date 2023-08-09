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
#include <range/v3/view/remove_if.hpp>
#include "fastjet/contrib/Nsubjettiness.hh" // In external code, this should be fastjet/contrib/Nsubjettiness.hh
#pragma GCC diagnostic pop

#include <cxxopts.hpp>

using namespace fastjet;
using namespace ranges;
using namespace fastjet::contrib;

PseudoJet Convert(double pt, double eta, double phi) {
    double px = pt*cos(phi);
    double py = pt*sin(phi);
    double pz = pt*sinh(eta);
    double E = pt*cosh(eta);
    return PseudoJet(px, py, pz, E);
}

void ProcessEvent(auto &&event, JetDefinition &jet_def, std::ostream& out) {
    static constexpr double ptmin = 30;
    ClusterSequence cs(event, jet_def);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(ptmin));
    NsubjettinessRatio   nSub21_beta1(2,1, OnePass_WTA_KT_Axes(), UnnormalizedMeasure(1.0));
    double tau21_beta1 = nSub21_beta1(jets[0]);
    double tau21_beta2 = nSub21_beta1(jets[1]);

    out << (jets[0]+jets[1]).m() << "," << jets[0].m() << "," << jets[1].m() << "," << jets[0].delta_R(jets[1]) << "," << tau21_beta1 << "," << tau21_beta2 << "\n";
}

//----------------------------------------------------------------------
int main(int argc, char** argv){
    cxxopts::Options options("Plot Events",
                             "Plot events dijet mass, mj1, mj2, DR12, tau21_1, tau21_2");
    options.add_options()
        ("i,input", "Filename to read from", cxxopts::value<std::string>())
        ("o,output", "Output file name", cxxopts::value<std::string>())
        ("p,ptmin", "Minimum pt allowed for a particle", cxxopts::value<double>()->default_value("1.0"))
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
    double ptmin = result["ptmin"].as<double>();

    static constexpr double R = 1.0;
    JetDefinition jet_def(antikt_algorithm, R);

    std::ifstream in(input.c_str());
    std::ofstream out(output.c_str());

    // Parse input
    std::string line;
    const auto re = std::regex{"[-+]?[\\d.]+(?:e[-+]?\\d+)?"};
    size_t num = 0;
    while(std::getline(in, line)) {
        if(num++ % 1000 == 0) {
            std::cout << "Processing event " << num << "\r";
            std::flush(std::cout);
        }
        auto tmp = line | views::tokenize(re) 
                          | views::transform([](const auto &v){ return std::stod(v); })
                          | to<std::vector>();
        auto event = tmp | views::chunk(3) | views::remove_if([ptmin](const auto &r) { return r[0] < ptmin; })
                         | views::transform([](const auto &r) { return Convert(r[0], r[1], r[2]); })
                         | to<std::vector>();
        ProcessEvent(event, jet_def, out);
    }
}
