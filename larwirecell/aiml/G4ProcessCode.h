/** G4 creation-process name -> integer code, following the CellTree convention
 * (cf. Ningclover larwirecell/aiml/TrackIDPIDMap2h5.cxx).  Unknown processes map
 * to -1.  "Michel" (10001) is not a G4 process: it is a synthetic tag for a decay
 * electron of a muon (pdg == e, process == "Decay", mother pdg == mu).
 *
 * Header-only so that the truth writers in two libraries -- wclsTensorSetLabeler
 * (WireCellAIML, the "process" column of truth_per_track) and
 * wclsTruthInformationAttacher (WireCellLarsoft, the "process" column of
 * truth_pf) -- share ONE table.
 */
#ifndef LARWIRECELL_AIML_G4PROCESSCODE
#define LARWIRECELL_AIML_G4PROCESSCODE

#include <cstdlib>
#include <string>
#include <unordered_map>

namespace wcls {
  namespace truth {
    inline const std::unordered_map<std::string, int>& g4_process_map()
    {
      static const std::unordered_map<std::string, int> m = {
        {"primary", 0},        {"Decay", 1},        {"eIoni", 2},
        {"muIoni", 3},         {"eBrem", 4},        {"compt", 5},
        {"phot", 6},           {"conv", 7},         {"hIoni", 8},
        {"nCapture", 9},       {"muPairProd", 10},  {"CoulombScat", 11},
        {"muBrems", 12},       {"LowEnConversion", 13}, {"annihil", 14},
        {"neutronInelastic", 15}, {"hadElastic", 16},
        {"hBertiniCaptureAtRest", 17}, {"muMinusCaptureAtRest", 18},
        {"protonInelastic", 19}, {"pi+Inelastic", 20}, {"pi-Inelastic", 21},
        {"PhotonInelastic", 22}, {"CHIPSNuclearCaptureAtRest", 23},
        {"Transportation", 24}, {"kaon+Inelastic", 25}, {"kaon-Inelastic", 26},
        {"kaon0LInelastic", 27}, {"ionInelastic", 28}, {"Scintillation", 29},
        {"ionIoni", 30},       {"nKiller", 31},     {"StepLimiter", 32},
        {"dInelastic", 33},    {"Michel", 10001}};
      return m;
    }

    constexpr int kMichelCode = 10001;

    inline bool is_michel(int pdg, const std::string& proc, int mother_pdg)
    {
      return std::abs(pdg) == 11 && proc == "Decay" && std::abs(mother_pdg) == 13;
    }

    inline int g4_process_code(int pdg, const std::string& proc, int mother_pdg)
    {
      if (is_michel(pdg, proc, mother_pdg)) { return kMichelCode; }
      const auto& m = g4_process_map();
      auto it = m.find(proc);
      return it == m.end() ? -1 : it->second;
    }
  }
}

#endif
