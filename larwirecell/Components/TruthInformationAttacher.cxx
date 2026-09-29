#include "TruthInformationAttacher.h"

#include "larwirecell/aiml/G4ProcessCode.h"

#include "WireCellAux/SimpleTensor.h"
#include "WireCellAux/SimpleTensorSet.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/String.h"

#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "canvas/Persistency/Common/FindOneP.h"
#include "canvas/Utilities/InputTag.h"
#include "lardataobj/Simulation/SimEnergyDeposit.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "nusimdata/SimulationBase/MCTruth.h"

#include <cstdlib>
#include <unordered_map>

WIRECELL_FACTORY(wclsTruthInformationAttacher,
                 wcls::TruthInformationAttacher,
                 wcls::IArtEventVisitor,
                 WireCell::ITensorSetFilter,
                 WireCell::IConfigurable)

using namespace wcls;
using namespace WireCell;

const std::vector<std::string>& TruthInformationAttacher::nu_columns()
{
  static const std::vector<std::string> c = {
    "nu_idx", "pdg", "ccnc", "mode", "int_type", "flavor", "E",
    "vtx_x", "vtx_y", "vtx_z", "t", "edep"};
  return c;
}

const std::vector<std::string>& TruthInformationAttacher::pf_columns()
{
  static const std::vector<std::string> c = {
    "nu_row", "trackid", "parent_trackid", "mother_trackid", "pdg", "process",
    "E", "KE",
    "start_x", "start_y", "start_z", "start_t",
    "end_x", "end_y", "end_z", "end_t",
    "start_px", "start_py", "start_pz"};
  return c;
}

TruthInformationAttacher::TruthInformationAttacher()
  : Aux::Logger("TruthInformationAttacher", "larwirecell")
{
}

TruthInformationAttacher::~TruthInformationAttacher() {}

Configuration TruthInformationAttacher::default_configuration() const
{
  Configuration cfg;
  cfg["run_key"] = m_run_key;
  cfg["subrun_key"] = m_sub_key;
  cfg["event_key"] = m_evt_key;
  cfg["truth"] = m_truth;
  cfg["mctruth_label"] = m_mctruth_label;
  cfg["mcparticle_label"] = m_mcparticle_label;
  cfg["deposet_label"] = m_deposet_label;
  cfg["pf_ke_min"] = m_pf_ke_min;
  cfg["nu_datapath"] = m_nu_datapath;
  cfg["pf_datapath"] = m_pf_datapath;
  return cfg;
}

void TruthInformationAttacher::configure(const Configuration& cfg)
{
  m_run_key = get<std::string>(cfg, "run_key", m_run_key);
  m_sub_key = get<std::string>(cfg, "subrun_key", m_sub_key);
  m_evt_key = get<std::string>(cfg, "event_key", m_evt_key);
  m_truth = get<bool>(cfg, "truth", m_truth);
  m_mctruth_label = get<std::string>(cfg, "mctruth_label", m_mctruth_label);
  m_mcparticle_label = get<std::string>(cfg, "mcparticle_label", m_mcparticle_label);
  m_deposet_label = get<std::string>(cfg, "deposet_label", m_deposet_label);
  m_pf_ke_min = get<double>(cfg, "pf_ke_min", m_pf_ke_min);
  m_nu_datapath = get<std::string>(cfg, "nu_datapath", m_nu_datapath);
  m_pf_datapath = get<std::string>(cfg, "pf_datapath", m_pf_datapath);
}

// CellTree's "primary" test was Mother()==0; with the trackid-offset scheme
// the geant "primary" process string is the robust equivalent (as the labeler).
static bool is_primary(const simb::MCParticle& p)
{
  return p.Mother() == 0 || p.Process() == "primary";
}

void TruthInformationAttacher::visit(art::Event& event)
{
  m_run = event.run();
  m_sub = event.subRun();
  m_evt = event.event();
  m_seen = true;
  m_nu.clear();
  m_pf.clear();
  m_n_nu = m_n_pf = 0;
  if (!m_truth) { return; }

  // --- neutrino level: every generator MCTruth with a neutrino, in MCTruth
  // order.  The same fields the labeler writes as nu_* metadata arrays, plus
  // the interaction mode and time.
  art::Handle<std::vector<simb::MCTruth>> mctruths;
  std::unordered_map<int, size_t> nuidx2row; // MCTruth index -> truth_nu row
  if (event.getByLabel(art::InputTag{m_mctruth_label}, mctruths) && mctruths.isValid()) {
    for (size_t i = 0; i < mctruths->size(); ++i) {
      const auto& mct = (*mctruths)[i];
      if (!mct.NeutrinoSet()) { continue; }
      const auto& nu = mct.GetNeutrino();
      const auto& nup = nu.Nu();
      const auto& pos = nup.Position(0);
      const int pdg = nup.PdgCode();
      const int ccnc = nu.CCNC();
      int flavor = 0;                                   // labeler nu_flavor:
      if (ccnc == 1) { flavor = 4; }                    //   "nc"
      else if (std::abs(pdg) == 12) { flavor = 1; }     //   "nue"
      else if (std::abs(pdg) == 14) { flavor = 2; }     //   "numu"
      else if (std::abs(pdg) == 16) { flavor = 3; }     //   "nutau"
      nuidx2row[(int)i] = m_n_nu++;
      for (double v : {(double)i, (double)pdg, (double)ccnc, (double)nu.Mode(),
                       (double)nu.InteractionType(), (double)flavor,
                       nup.Momentum(0).E(), pos.X(), pos.Y(), pos.Z(), pos.T(),
                       0.0 /* edep, below */}) {
        m_nu.push_back(v);
      }
    }
  }
  if (m_n_nu == 0) {
    log->debug("visit run {} sub {} evt {}: no neutrino MCTruth at '{}' -> RSE only",
               m_run, m_sub, m_evt, m_mctruth_label);
    return;
  }

  // --- beam-neutrino origin of each MCParticle (largeant Assns), as the labeler
  art::Handle<std::vector<simb::MCParticle>> mcps;
  if (!(event.getByLabel(art::InputTag{m_mcparticle_label}, mcps) && mcps.isValid())) {
    log->warn("failed to fetch MCParticles with label '{}': truth_pf empty", m_mcparticle_label);
    return;
  }
  const size_t np = mcps->size();
  std::unordered_map<int, size_t> tid2idx;
  tid2idx.reserve(np);
  for (size_t i = 0; i < np; ++i) { tid2idx[(*mcps)[i].TrackId()] = i; }
  std::vector<int> nu_index(np, -1);   // per MCParticle: beam-nu MCTruth index, -1 = not beam
  {
    art::FindOneP<simb::MCTruth> mcp2truth(mcps, event, art::InputTag{m_mcparticle_label});
    if (mcp2truth.isValid()) {
      for (size_t i = 0; i < np; ++i) {
        const auto mct = mcp2truth.at(i);
        if (mct.isNonnull() && mct->Origin() == simb::kBeamNeutrino) { nu_index[i] = (int)mct.key(); }
      }
    }
    else {
      log->warn("no MCParticle<->MCTruth Assns at '{}': truth_pf empty", m_mcparticle_label);
    }
  }

  // --- deposited energy per interaction (the labeler's nu_edep)
  {
    art::Handle<std::vector<sim::SimEnergyDeposit>> seds;
    std::unordered_map<int, double> edep;   // MeV
    if (event.getByLabel(art::InputTag{m_deposet_label}, seds) && seds.isValid()) {
      for (const auto& sed : *seds) {
        const auto it = tid2idx.find(std::abs(sed.TrackID()));
        if (it == tid2idx.end()) { continue; }
        const int nidx = nu_index[it->second];
        if (nidx >= 0) { edep[nidx] += sed.Energy(); }
      }
    }
    else {
      log->warn("failed to fetch SimEnergyDeposits with label '{}': edep = 0", m_deposet_label);
    }
    const size_t nc = nu_columns().size();
    for (const auto& [nidx, row] : nuidx2row) {
      m_nu[row * nc + (nc - 1)] = (edep.count(nidx) ? edep.at(nidx) : 0.0) * 1e-3;   // GeV
    }
  }

  // --- particle level: the labeler's Bee "mc" particle-flow selection
  // (pf_nu_only): beam-nu-derived MCParticles with KE > pf_ke_min, each
  // attached to its nearest KEPT ancestor (0 = directly to the neutrino).
  auto kept = [&](size_t i) {
    const auto& p = (*mcps)[i];
    return nu_index[i] >= 0 && (p.E() - p.Mass()) * 1e3 > m_pf_ke_min;
  };
  for (size_t i = 0; i < np; ++i) {
    if (!kept(i)) { continue; }
    const auto& p = (*mcps)[i];
    int anc = p.Mother();
    while (anc > 0) {
      const auto it = tid2idx.find(anc);
      if (it == tid2idx.end()) { anc = 0; break; }
      if (kept(it->second)) { break; }
      anc = (*mcps)[it->second].Mother();
    }
    const auto itm = tid2idx.find(p.Mother());
    const int mother_pdg = itm == tid2idx.end() ? 0 : (*mcps)[itm->second].PdgCode();
    const auto itr = nuidx2row.find(nu_index[i]);
    const auto& s4 = p.Position(0);
    const auto& e4 = p.EndPosition();
    const auto& sm = p.Momentum(0);
    for (double v : {(double)(itr == nuidx2row.end() ? -1 : (int)itr->second),
                     (double)p.TrackId(), (double)(anc > 0 ? anc : 0), (double)p.Mother(),
                     (double)p.PdgCode(),
                     (double)truth::g4_process_code(p.PdgCode(), p.Process(), mother_pdg),
                     p.E(), p.E() - p.Mass(),
                     s4.X(), s4.Y(), s4.Z(), s4.T(),
                     e4.X(), e4.Y(), e4.Z(), e4.T(),
                     sm.Px(), sm.Py(), sm.Pz()}) {
      m_pf.push_back(v);
    }
    ++m_n_pf;
  }
  size_t nprim = 0;
  for (size_t i = 0; i < np; ++i) { if (nu_index[i] >= 0 && is_primary((*mcps)[i])) { ++nprim; } }
  log->debug("visit run {} sub {} evt {}: {} neutrino(s), {} particle-flow particles "
             "(KE > {} MeV; {} beam-nu primaries)",
             m_run, m_sub, m_evt, m_n_nu, m_n_pf, m_pf_ke_min, nprim);
}

static std::string format_path(const std::string& path, int ident)
{
  if (path.find('%') == std::string::npos) { return path; }
  return String::format(path, ident);
}

static ITensor::pointer make_table(const std::vector<double>& flat, size_t nrows,
                                   const std::vector<std::string>& cols,
                                   const std::string& datatype, const std::string& datapath)
{
  Configuration md;
  md["datapath"] = datapath;
  md["datatype"] = datatype;
  md["units"] = "cm, ns, GeV";
  md["columns"] = Json::arrayValue;
  for (const auto& c : cols) { md["columns"].append(c); }
  // nrows == 0 (a neutrino with no particle above threshold): a [0 x ncols] table.
  return std::make_shared<Aux::SimpleTensor>(std::vector<size_t>{nrows, cols.size()},
                                             nrows ? flat.data() : (const double*)nullptr, md);
}

bool TruthInformationAttacher::operator()(const ITensorSet::pointer& in, ITensorSet::pointer& out)
{
  out = nullptr;
  if (!in) {                    // EOS
    log->debug("EOS at call={}", m_count++);
    return true;
  }
  if (!m_seen) {
    // visit() is driven by wcls from the art event loop; if this component is
    // not an fcl inputer it never fires.  Warn and pass through rather than
    // stamp a wrong RSE (as TensorSetMetadataAttacher).
    log->warn("call={}: visit(art::Event&) never called -- is this component "
              "listed in the fcl 'inputers'?  passing through with no RSE/truth",
              m_count);
    out = in;
    ++m_count;
    return true;
  }

  Configuration md = in->metadata();
  md[m_run_key] = m_run;
  md[m_sub_key] = m_sub;
  md[m_evt_key] = m_evt;

  if (m_n_nu == 0) {
    // RSE only: share the input tensors by pointer (no copy).
    out = std::make_shared<Aux::SimpleTensorSet>(in->ident(), md, in->tensors());
  }
  else {
    auto tens = std::make_shared<ITensor::vector>(*in->tensors());
    const int ident = in->ident();
    tens->push_back(make_table(m_nu, m_n_nu, nu_columns(), "truth_nu", format_path(m_nu_datapath, ident)));
    tens->push_back(make_table(m_pf, m_n_pf, pf_columns(), "truth_pf", format_path(m_pf_datapath, ident)));
    out = std::make_shared<Aux::SimpleTensorSet>(ident, md, tens);
  }
  log->debug("call={} ident={}: stamped rse=({},{},{}), truth: {} nu, {} pf rows",
             m_count, in->ident(), m_run, m_sub, m_evt, m_n_nu, m_n_pf);
  ++m_count;
  return true;
}
