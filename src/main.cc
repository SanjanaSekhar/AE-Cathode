#include <fastjet/PseudoJet.hh>
#include <iostream>
#include <fstream>

#include <cmath>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/contrib/RecursiveSoftDrop.hh"
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/range/conversion.hpp>
#pragma GCC diagnostic pop
#include "highfive/H5File.hpp"
#include "Eigen/Dense"

using namespace fastjet;
using namespace ranges;

// forward declaration to make things clearer
void print_raw_prongs(const PseudoJet &jet, std::ostream&);

PseudoJet Convert(double pt, double eta, double phi) {
    double px = pt*cos(phi);
    double py = pt*sin(phi);
    double pz = pt*sinh(eta);
    double E = pt*cosh(eta);
    return PseudoJet(px, py, pz, E);
}

void ProcessEvent(auto &&event, JetDefinition &jet_def, contrib::RecursiveSoftDrop &rsd, std::ostream& out) {
    static constexpr double ptmin = 30;
    ClusterSequence cs(event, jet_def);
    std::vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(ptmin));

    for (unsigned ijet = 0; ijet < 2; ijet++) {
        // Run SoftDrop and examine the output
        PseudoJet rsd_jet = rsd(jets[ijet]);
        print_raw_prongs(rsd_jet, out);
    }
}

//----------------------------------------------------------------------
int main(){

    HighFive::File file("../events_LHCO2020_BlackBox1.h5", HighFive::File::ReadOnly);
    auto dataset = file.getDataSet("df/block0_values");
    auto dims = dataset.getDimensions();
    std::cout << "Shape: (";
    for(const auto &dim : dims)
        std::cout << dim << ", ";
    std::cout << ")" << std::endl;

    static constexpr size_t stride = 10000;
    auto nelements = stride*dims[1];
    std::vector<double> data(nelements);

    static constexpr double R = 1.0;
    static constexpr double z_cut = 0.2;
    double beta  = 0.5;
    int n=-1; // number of layers (-1 <> infinite)
    JetDefinition jet_def(antikt_algorithm, R);
    contrib::RecursiveSoftDrop rsd(beta, z_cut, n, R);
    rsd.set_verbose_structure(true);
    rsd.set_dynamical_R0();

    std::ofstream out("events.csv");

    for(size_t i = 0; i < dims[0]/stride; ++i) {
        std::cout << "Processing chunk " << i+1 << "/" << dims[0]/stride << std::endl;
        size_t start = i*stride;
        dataset.select({start, 0}, {stride, dims[1]}).read(data.data());
        auto particles = data | views::chunk(3) 
            | views::transform([](auto r) { return Convert(r[0], r[1], r[2]); });
        auto events = particles | views::chunk(700);
        for(const auto &event : events) {
            auto vevent = event | to<std::vector>();
            ProcessEvent(vevent, jet_def, rsd, out);
            out << "\n";
        }
    }
}

//----------------------------------------------------------------------
// print all the prongs inside the jet (no clustering info)
void print_raw_prongs(const PseudoJet &jet, std::ostream &out){
    using namespace std;

    vector<PseudoJet> prongs = contrib::recursive_soft_drop_prongs(jet);
    std::sort(prongs.begin(), prongs.end(), [](PseudoJet a, PseudoJet b) { return a.pt() > b.pt(); });
    for (unsigned int iprong=0; iprong<prongs.size(); ++iprong){
        const PseudoJet & prong = prongs[iprong];
        // const contrib::RecursiveSoftDrop::StructureType &structure = prong.structure_of<contrib::RecursiveSoftDrop>();
        out << prong.pt() << "," << prong.rap() << "," << prong.phi() << ",";
    }
}
