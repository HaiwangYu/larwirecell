#include "TensorSetLabeler.h"

#include "WireCellAux/SimpleTensor.h"
#include "WireCellAux/SimpleTensorSet.h"
#include "WireCellAux/TensorDMcommon.h"
#include "WireCellAux/TensorDMpointtree.h"
#include "WireCellIface/WirePlaneId.h"
#include "WireCellUtil/Exceptions.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Point.h"
#include "WireCellUtil/RayGrid.h"
#include "WireCellUtil/PointTree.h"
#include "WireCellUtil/String.h"
#include "WireCellUtil/Units.h"

#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "canvas/Persistency/Common/FindOneP.h"
#include "canvas/Utilities/InputTag.h"
#include "lardataobj/Simulation/SimEnergyDeposit.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "nusimdata/SimulationBase/MCTruth.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <unordered_map>

WIRECELL_FACTORY(wclsTensorSetLabeler,
                 WireCell::AIML::TensorSetLabeler,
                 wcls::IArtEventVisitor,
                 WireCell::INamed,
                 WireCell::ITensorSetFilter,
                 WireCell::IConfigurable)

using namespace WireCell;
using WireCell::Aux::SimpleTensor;
using WireCell::Aux::SimpleTensorSet;
using namespace WireCell::Aux::TensorDM;
using WireCell::PointCloud::Array;
using WireCell::PointCloud::Dataset;

// Columns of the "truth_per_track" tensor.  Units: LArSoft native
// (positions cm, time ns, momentum/energy GeV).  "process" is the G4
// creation-process code (see g4_process_code below).
static const std::vector<std::string> track_columns = {
  "trackid",  "pdg",      "mother_trackid", "mother_pdg", "status",
  "start_x",  "start_y",  "start_z",        "start_t",    "start_px",
  "start_py", "start_pz", "start_E",        "end_x",      "end_y",
  "end_z",    "end_t",    "end_px",         "end_py",     "end_pz",
  "end_E",    "nu_idx",   "process"};

// G4 creation-process name -> integer code, following the CellTree
// convention (cf. Ningclover larwirecell/aiml/TrackIDPIDMap2h5.cxx).
// Unknown processes map to -1.  "Michel" (10001) is not a G4 process: it
// is a synthetic tag assigned by g4_process_code() below.
static const std::unordered_map<std::string, int> g4_process_map = {
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

// A Michel electron is an e+- created by the decay of a muon: pdg == e,
// process == "Decay", mother pdg == mu.  Such tracks get the synthetic
// "Michel" (10001) code; every other track maps its G4 process string.
static const int kMichelCode = 10001;
static bool is_michel(int pdg, const std::string& proc, int mother_pdg)
{
  return std::abs(pdg) == 11 && proc == "Decay" && std::abs(mother_pdg) == 13;
}
static int g4_process_code(int pdg, const std::string& proc, int mother_pdg)
{
  if (is_michel(pdg, proc, mother_pdg)) { return kMichelCode; }
  auto it = g4_process_map.find(proc);
  return it == g4_process_map.end() ? -1 : it->second;
}

// Human-readable particle name for the Bee "mc" tree text.
static std::string pdg_name(int pdg)
{
  switch (pdg) {
    case 13: return "mu-";     case -13: return "mu+";
    case 11: return "e-";      case -11: return "e+";
    case 22: return "gamma";
    case 2212: return "proton"; case -2212: return "antiproton";
    case 2112: return "neutron";
    case 211: return "pi+";    case -211: return "pi-";   case 111: return "pi0";
    case 321: return "K+";     case -321: return "K-";
    case 130: return "K0L";    case 310: return "K0S";    case 311: return "K0";
    case 12: case -12: return "nue";
    case 14: case -14: return "numu";
    case 3122: return "lambda";
    case 1000010020: return "deuteron";
    case 1000010030: return "triton";
    case 1000020040: return "alpha";
    default: {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "pdg %d", pdg);
      return buf;
    }
  }
}

AIML::TensorSetLabeler::TensorSetLabeler()
  : Aux::Logger("TensorSetLabeler", "aiml")
  , m_drift_speed(1.563 * units::mm / units::us)
  , m_time_offset(-205 * units::us)
  , m_tick(0.5 * units::us)
  , m_DL(4.0 * units::cm * units::cm / units::s)   // wcsimsp_sbnd.fcl DL
  , m_DT(8.8 * units::cm * units::cm / units::s)   // wcsimsp_sbnd.fcl DT
  , m_sp_smear_time(1.0 / (2 * 3.141592653589793 * 0.10 * units::megahertz))
  , m_pf_ke_min(10 * units::MeV)
    // = 1.59 us: Gaus_wide sigma = 0.10 MHz (sbnd sp-filters.jsonnet),
    // sigma_t = 1/(2*pi*f) per dunereco docs/smear-dnn-campaign.md.
    // sp_smear_wire defaults (pitch units) = 1/(2*sqrt(pi)*k) with
    // k = 1.05 (Wire_ind) / 3.60 (Wire_col) from the same jsonnet.
{}

AIML::TensorSetLabeler::~TensorSetLabeler() {}

Configuration AIML::TensorSetLabeler::default_configuration() const
{
  Configuration cfg;
  cfg["inpath"] = m_inpath;
  cfg["grouping"] = m_grouping;
  cfg["truth_datapath"] = m_truth_datapath;
  cfg["deposet_label"] = m_deposet_label;
  cfg["mctruth_label"] = m_mctruth_label;
  cfg["mcparticle_label"] = m_mcparticle_label;
  // MUST match the BlobSampler configuration that made the "3d" PCs.
  cfg["drift_speed"] = m_drift_speed;
  cfg["time_offset"] = m_time_offset;
  // Residual shift ADDED to depo times before drift conversion (see header).
  cfg["depo_time_offset"] = m_depo_time_offset;
  cfg["tick"] = m_tick;
  // Acceptance slop (in wires / ticks) around the blob bounds, standing in
  // for the diffusion extents BlobDepoFill integrates (point-like depos).
  cfg["wire_slop"] = m_wire_slop;
  cfg["tick_slop"] = m_tick_slop;
  // readout length in ticks; used to clip the truth_depo_sce Bee display
  // to the window blobs can exist in.
  cfg["nticks"] = m_nticks;
  // Depo diffusion (see header): drift diffusion DL/DT + SP filter smearing
  // (time sigma + per-plane-type wire sigma in pitch units); acceptance is
  // widened by nsigma * the quadrature sum.
  cfg["DL"] = m_DL;
  cfg["DT"] = m_DT;
  cfg["sp_smear_time"] = m_sp_smear_time;
  cfg["sp_smear_wire_ind"] = m_sp_smear_wire_ind;
  cfg["sp_smear_wire_col"] = m_sp_smear_wire_col;
  cfg["nsigma"] = m_nsigma;
  // truth_depo_sce Bee set: Gaussian samples per depo diffusion ball.
  cfg["n_sample_truth_depo_sce"] = m_nsample_depo;
  // ISCEField with the TrueFwd (true->reco) displacement map; empty = off.
  cfg["sce_field"] = "";
  cfg["sce_correction"] = m_sce_correction;
  // KE cut for the Bee "mc" particle-flow tree.
  cfg["pf_ke_min"] = m_pf_ke_min;
  // Optional IFiducial: keep a particle in the "mc" tree only if its start
  // or end point is inside this volume; empty = no FV cut.
  cfg["pf_fiducial"] = "";
  // "mc" tree: keep only particles derived from a beam neutrino (drops all
  // cosmics, including FV-crossing multi-GeV muons).
  cfg["pf_nu_only"] = m_pf_nu_only;
  // truth_per_track: keep only particles descending from the generator
  // (neutrino) MCTruth (no cosmic-muon truth).
  cfg["truth_tracks_nu_only"] = m_truth_tracks_nu_only;
  // Bee "trackid merging": display Michel electrons under their mother muon
  // cluster_id (Bee only; the blob scalar PC keeps the true trackid).
  cfg["bee_michel_merge"] = m_bee_michel_merge;
  cfg["anodes"] = Json::arrayValue;
  cfg["bee_sink"] = "";
  cfg["bee_detector"] = m_bee_detector;
  cfg["bee_algorithm"] = m_bee_algorithm;
  cfg["initial_index"] = m_bee_index;
  return cfg;
}

void AIML::TensorSetLabeler::configure(const Configuration& cfg)
{
  m_inpath = get(cfg, "inpath", m_inpath);
  m_grouping = get(cfg, "grouping", m_grouping);
  m_truth_datapath = get(cfg, "truth_datapath", m_truth_datapath);
  m_deposet_label = get(cfg, "deposet_label", m_deposet_label);
  m_mctruth_label = get(cfg, "mctruth_label", m_mctruth_label);
  m_mcparticle_label = get(cfg, "mcparticle_label", m_mcparticle_label);
  m_drift_speed = get(cfg, "drift_speed", m_drift_speed);
  m_time_offset = get(cfg, "time_offset", m_time_offset);
  m_depo_time_offset = get(cfg, "depo_time_offset", m_depo_time_offset);
  m_tick = get(cfg, "tick", m_tick);
  m_wire_slop = get(cfg, "wire_slop", m_wire_slop);
  m_tick_slop = get(cfg, "tick_slop", m_tick_slop);
  m_nticks = get(cfg, "nticks", m_nticks);
  m_DL = get(cfg, "DL", m_DL);
  m_DT = get(cfg, "DT", m_DT);
  m_sp_smear_time = get(cfg, "sp_smear_time", m_sp_smear_time);
  m_sp_smear_wire_ind = get(cfg, "sp_smear_wire_ind", m_sp_smear_wire_ind);
  m_sp_smear_wire_col = get(cfg, "sp_smear_wire_col", m_sp_smear_wire_col);
  m_nsigma = get(cfg, "nsigma", m_nsigma);
  m_nsample_depo = get(cfg, "n_sample_truth_depo_sce", m_nsample_depo);
  m_sce_correction = get(cfg, "sce_correction", m_sce_correction);
  m_truth_tracks_nu_only = get(cfg, "truth_tracks_nu_only", m_truth_tracks_nu_only);
  m_pf_nu_only = get(cfg, "pf_nu_only", m_pf_nu_only);
  m_pf_ke_min = get(cfg, "pf_ke_min", m_pf_ke_min);
  m_bee_michel_merge = get(cfg, "bee_michel_merge", m_bee_michel_merge);

  m_anodes.clear();
  m_faces.clear();
  for (const auto& janode : cfg["anodes"]) {
    auto anode = Factory::find_tn<IAnodePlane>(janode.asString());
    m_anodes.push_back(anode);
    for (const auto& face : anode->faces()) {
      FaceCtx fc;
      fc.face = face;
      // same origin as BlobSampler::plane_x(2)
      fc.xw = face->planes()[2]->wires().front()->center().x();
      fc.dirx = face->dirx();
      const auto sens = face->sensitive();
      const auto& bb = sens.bounds();
      fc.xmin = std::min(bb.first.x(), bb.second.x());
      fc.xmax = std::max(bb.first.x(), bb.second.x());
      for (int ip = 0; ip < 3; ++ip) {
        fc.pitch[ip] = face->raygrid().pitch_mags()[2 + ip];
      }
      m_faces[{anode->ident(), face->which()}] = fc;
    }
  }
  if (m_faces.empty()) {
    THROW(ValueError() << errmsg{"wclsTensorSetLabeler requires a non-empty 'anodes' list"});
  }

  const std::string fid_tn = get<std::string>(cfg, "pf_fiducial", "");
  m_pf_fiducial = fid_tn.empty() ? nullptr : Factory::find_tn<IFiducial>(fid_tn);

  const std::string sce_tn = get<std::string>(cfg, "sce_field", "");
  m_sce = nullptr;
  if (!sce_tn.empty()) {
    m_sce = Factory::find_tn<ISCEField>(sce_tn);
    // Sanity probes: the TrueFwd map should be roughly OPPOSITE to the
    // TrueBkwd (reco->true) displacements the SCECorrection probes log.
    const std::vector<Point> probes = {
      {-10 * units::cm, 100 * units::cm, 250 * units::cm},
      {-190 * units::cm, 100 * units::cm, 250 * units::cm},
      {10 * units::cm, 100 * units::cm, 250 * units::cm},
      {190 * units::cm, 100 * units::cm, 250 * units::cm},
    };
    for (const auto& pp : probes) {
      const int apa = pp.x() < 0 ? 0 : 1;
      const double dx = m_sce->displacement_x(apa, pp.x(), pp.y(), pp.z());
      const double dy = m_sce->displacement_y(apa, pp.x(), pp.y(), pp.z());
      const double dz = m_sce->displacement_z(apa, pp.x(), pp.y(), pp.z());
      log->debug("SCE TrueFwd probe apa{} ({:.0f},{:.0f},{:.0f})cm -> "
                 "d=({:.3f},{:.3f},{:.3f})cm |d|={:.3f}cm (true->reco, {})",
                 apa, pp.x() / units::cm, pp.y() / units::cm, pp.z() / units::cm,
                 dx / units::cm, dy / units::cm, dz / units::cm,
                 std::sqrt(dx * dx + dy * dy + dz * dz) / units::cm,
                 m_sce_correction ? "APPLIED to depos" : "sce_correction=false, NOT applied");
    }
  }
  else {
    log->debug("no sce_field configured: depos used at true (priorSCE) positions");
  }

  const std::string bee_tn = get<std::string>(cfg, "bee_sink", "");
  if (!bee_tn.empty()) {
    m_bee_sink = Factory::find_tn<Clus::IBeeSink>(bee_tn);
    m_bee_sink->acquire();
    m_bee_index = get(cfg, "initial_index", m_bee_index);
    m_bee_detector = get(cfg, "bee_detector", m_bee_detector);
    m_bee_algorithm = get(cfg, "bee_algorithm", m_bee_algorithm);
    log->debug("using shared Bee sink {} for '{}' dump", bee_tn, m_bee_algorithm);
  }

  log->debug("depo diffusion: DL {} DT {} cm2/s, SP smear time {} us, "
             "wire ind/col {}/{} pitch, nsigma {}, depo-ball samples {}",
             m_DL / (units::cm * units::cm / units::s),
             m_DT / (units::cm * units::cm / units::s),
             m_sp_smear_time / units::us,
             m_sp_smear_wire_ind, m_sp_smear_wire_col,
             m_nsigma, m_nsample_depo);
  log->debug("labeling '{}' at '{}': depos '{}', mctruth '{}', mcparticles '{}', "
             "drift_speed {} mm/us, time_offset {} us, depo_time_offset {} us, tick {} us",
             m_grouping,
             m_inpath,
             m_deposet_label,
             m_mctruth_label,
             m_mcparticle_label,
             m_drift_speed / (units::mm / units::us),
             m_time_offset / units::us,
             m_depo_time_offset / units::us,
             m_tick / units::us);
}

void AIML::TensorSetLabeler::finalize()
{
  if (m_bee_sink) {
    m_bee_sink->release();
    m_bee_sink = nullptr;
  }
}

// CellTree's "primary" test was Mother()==0; with the trackid-offset scheme
// the geant "primary" process string is the robust equivalent.
static bool is_primary(const simb::MCParticle& p)
{
  return p.Mother() == 0 || p.Process() == "primary";
}

void AIML::TensorSetLabeler::visit(art::Event& event)
{
  m_run = event.run();
  m_sub = event.subRun();
  m_evt = event.event();
  m_evtmd = Json::objectValue;
  m_tracks.clear();
  m_depos.clear();
  m_michel_mother.clear();
  m_nu_edep.clear();

  // --- neutrino truth (cf. Truth2h5) ---
  // An event can carry SEVERAL beam-neutrino interactions (rockbox: the
  // in-detector interaction plus dirt/rock ones), so every nu_* field is an
  // ARRAY with one entry per interaction; "n_nu" is the count and "nu_idx"
  // gives each entry's generator-MCTruth index (matches the truth_per_track
  // "nu_idx" column, the mc-tree node id offset 9000000+nu_idx, and the
  // m_nu_edep keys).  Entry 0 is the "main" interaction.  nu_edep is filled
  // in the Edep pass below (deposits are read after the MCParticles).
  for (const char* f : {"nu_idx", "nu_pdg", "nu_ccnc", "nu_int_type",
                        "nu_energy", "nu_vtx_x", "nu_vtx_y", "nu_vtx_z",
                        "nu_flavor"}) {
    m_evtmd[f] = Json::arrayValue;
  }
  m_evtmd["n_nu"] = 0;
  art::Handle<std::vector<simb::MCTruth>> mctruth_handle;
  if (event.getByLabel(art::InputTag{m_mctruth_label}, mctruth_handle) &&
      mctruth_handle.isValid()) {
    for (size_t i = 0; i < mctruth_handle->size(); ++i) {
      const auto& mct = (*mctruth_handle)[i];
      if (!mct.NeutrinoSet()) { continue; }
      const auto& nu = mct.GetNeutrino();
      const auto& nu_particle = nu.Nu();
      const auto& position = nu_particle.Position(0);
      const auto& momentum = nu_particle.Momentum(0);
      const int pdg = nu_particle.PdgCode();
      const int ccnc = nu.CCNC(); // 0 = CC, 1 = NC
      std::string flavor = "none";
      if (ccnc == 1) { flavor = "nc"; }
      else if (std::abs(pdg) == 12) { flavor = "nue"; }
      else if (std::abs(pdg) == 14) { flavor = "numu"; }
      else if (std::abs(pdg) == 16) { flavor = "nutau"; }
      m_evtmd["nu_idx"].append((int)i);
      m_evtmd["nu_pdg"].append(pdg);
      m_evtmd["nu_ccnc"].append(ccnc);
      m_evtmd["nu_int_type"].append(nu.InteractionType());
      m_evtmd["nu_energy"].append(momentum.E()); // GeV
      m_evtmd["nu_vtx_x"].append(position.X());   // cm
      m_evtmd["nu_vtx_y"].append(position.Y());
      m_evtmd["nu_vtx_z"].append(position.Z());
      m_evtmd["nu_flavor"].append(flavor);
    }
    m_evtmd["n_nu"] = (int)m_evtmd["nu_idx"].size();
  }
  if (m_evtmd["n_nu"].asInt() == 0) {
    log->debug("no neutrino MCTruth at '{}' for run {} sub {} evt {}",
               m_mctruth_label, m_run, m_sub, m_evt);
  }

  // --- per-track truth table (cf. TrackIDPIDMap2h5) ---
  art::Handle<std::vector<simb::MCParticle>> mcps;
  std::unordered_map<int, int> tid2pdg;
  std::unordered_map<int, size_t> tid2idx;
  std::vector<char> nu_origin; // per mcps index: origin is a beam-nu MCTruth
  std::vector<int> nu_index;   // per mcps index: beam-nu MCTruth key (0 = main), -1 = not beam
  if (event.getByLabel(art::InputTag{m_mcparticle_label}, mcps) && mcps.isValid()) {
    tid2pdg.reserve(mcps->size());
    tid2idx.reserve(mcps->size());
    for (size_t i = 0; i < mcps->size(); ++i) {
      const auto& p = (*mcps)[i];
      tid2pdg[p.TrackId()] = p.PdgCode();
      tid2idx[p.TrackId()] = i;
    }
    // Michel electron -> mother muon trackid, for the optional Bee
    // "trackid merging" (display Michel charge under its parent muon).
    for (size_t i = 0; i < mcps->size(); ++i) {
      const auto& p = (*mcps)[i];
      const auto itm = tid2pdg.find(p.Mother());
      const int mother_pdg = (itm == tid2pdg.end() ? 0 : itm->second);
      if (is_michel(p.PdgCode(), p.Process(), mother_pdg)) {
        m_michel_mother[p.TrackId()] = p.Mother();
      }
    }
    // Tag particles derived from a BEAM neutrino: the largeant Assns maps
    // every stored MCParticle to its origin MCTruth; require
    // Origin() == simb::kBeamNeutrino (cf. larreco CellTree_module.cc
    // processMC "nuOnly": mctruth->Origin()==1 && particle->Mother()==0 --
    // the Mother()==0 primary requirement is applied where used below).
    nu_origin.assign(mcps->size(), 0);
    nu_index.assign(mcps->size(), -1);
    {
      art::FindOneP<simb::MCTruth> mcp2truth(mcps, event, art::InputTag{m_mcparticle_label});
      if (mcp2truth.isValid()) {
        size_t nnu = 0, nprim = 0;
        for (size_t i = 0; i < mcps->size(); ++i) {
          const auto mct = mcp2truth.at(i);
          if (mct.isNonnull() && mct->Origin() == simb::kBeamNeutrino) {
            nu_origin[i] = 1;
            nu_index[i] = (int)mct.key(); // index within the generator MCTruth vector
            ++nnu;
            if (is_primary((*mcps)[i])) { ++nprim; }
          }
        }
        log->debug("{} of {} MCParticles derive from a beam neutrino ({} primaries)",
                   nnu, mcps->size(), nprim);
      }
      else {
        log->warn("no MCParticle<->MCTruth Assns at '{}': no beam-nu tagging",
                  m_mcparticle_label);
      }
    }
    m_tracks.reserve(m_truth_tracks_nu_only ? 64 : mcps->size());
    for (size_t i = 0; i < mcps->size(); ++i) {
      const auto& p = (*mcps)[i];
      // default: save only the beam-neutrino primaries, the larreco CellTree
      // "nuOnly" cut (Origin()==kBeamNeutrino && primary).  CellTree tested
      // Mother()==0; with the modern trackid-offset scheme (GENIE primaries
      // carry mother = the 1e7 offset root) Process()=="primary" is the
      // equivalent robust test.  No cosmic-muon (or secondary) truth.
      if (m_truth_tracks_nu_only &&
          !(i < nu_origin.size() && nu_origin[i] && is_primary(p))) {
        continue;
      }
      const auto& s4 = p.Position(0);
      const auto& sm = p.Momentum(0);
      const auto& e4 = p.EndPosition();
      const auto& em = p.EndMomentum();
      const auto itmom = tid2pdg.find(p.Mother());
      const int mother_pdg = (itmom == tid2pdg.end() ? 0 : itmom->second);
      m_tracks.push_back({(double)p.TrackId(),
                          (double)p.PdgCode(),
                          (double)p.Mother(),
                          (double)mother_pdg,
                          (double)p.StatusCode(),
                          s4.X(), s4.Y(), s4.Z(), s4.T(),
                          sm.Px(), sm.Py(), sm.Pz(), sm.E(),
                          e4.X(), e4.Y(), e4.Z(), e4.T(),
                          em.Px(), em.Py(), em.Pz(), em.E(),
                          (double)(i < nu_index.size() ? nu_index[i] : -1),
                          (double)g4_process_code(p.PdgCode(), p.Process(), mother_pdg)});
    }
  }
  else {
    log->warn("failed to fetch MCParticles with label '{}'", m_mcparticle_label);
  }

  // --- neutrino deposited energy (Edep): sum sim::SimEnergyDeposit::Energy()
  // over deposits whose (abs) trackid belongs to a beam-neutrino
  // interaction, accumulated per interaction (nu_idx).  This is the visible
  // (reconstructable) energy -- used for the event metadata and, in the
  // Bee "mc" tree, as the mother-neutrino node energy (not the nu total E).
  {
    art::Handle<std::vector<sim::SimEnergyDeposit>> edep_seds;
    if (event.getByLabel(art::InputTag{m_deposet_label}, edep_seds) && edep_seds.isValid()) {
      for (const auto& sed : *edep_seds) {
        const auto iit = tid2idx.find(std::abs(sed.TrackID()));
        if (iit == tid2idx.end() || iit->second >= nu_index.size()) { continue; }
        const int nidx = nu_index[iit->second];
        if (nidx >= 0) { m_nu_edep[nidx] += sed.Energy(); } // MeV
      }
    }
    // Per-interaction Edep into the event metadata, GeV to parallel
    // nu_energy, aligned entry-by-entry with the nu_idx array.
    m_evtmd["nu_edep"] = Json::arrayValue;
    for (const auto& jidx : m_evtmd["nu_idx"]) {
      const int k = jidx.asInt();
      m_evtmd["nu_edep"].append((m_nu_edep.count(k) ? m_nu_edep.at(k) : 0.0) * 1e-3);
    }
  }

  // --- Bee "mc" particle-flow tree: MCParticles with KE > pf_ke_min,
  // nested under the nearest KEPT ancestor by Mother() tracing.  Particles
  // are GROUPED under a per-interaction "initial mother neutrino" node
  // built from the generator MCTruth (rockbox events carry several beam-nu
  // interactions per event), keyed by the assns MCTruth index (nu_index).
  struct NuNode {
    bool valid{false};
    int pdg{0};
    double vx{0}, vy{0}, vz{0};
  };
  std::vector<NuNode> nutruths;
  if (mctruth_handle.isValid()) {
    for (const auto& mct : *mctruth_handle) {
      NuNode nn;
      if (mct.NeutrinoSet()) {
        const auto& nu_p = mct.GetNeutrino().Nu();
        nn.valid = true;
        nn.pdg = nu_p.PdgCode();
        const auto& pos = nu_p.Position(0);
        nn.vx = pos.X();
        nn.vy = pos.Y();
        nn.vz = pos.Z();
      }
      nutruths.push_back(nn);
    }
  }
  m_pf_particles = Json::arrayValue;
  if (mcps.isValid()) {
    std::unordered_map<int, const simb::MCParticle*> by_tid;
    by_tid.reserve(mcps->size());
    for (const auto& p : *mcps) {
      by_tid[p.TrackId()] = &p;
    }
    // FV cut: keep if start or end is inside the FV, or if the start-end
    // line section crosses it (sampled every 5 cm, capped for km-scale
    // segments -- FV path lengths below the step are negligible).
    auto fv_ok = [&](const simb::MCParticle& p) {
      if (!m_pf_fiducial) { return true; }
      const auto& s4 = p.Position(0);
      const auto& e4 = p.EndPosition();
      const Point a(s4.X() * units::cm, s4.Y() * units::cm, s4.Z() * units::cm);
      const Point b(e4.X() * units::cm, e4.Y() * units::cm, e4.Z() * units::cm);
      if (m_pf_fiducial->contained(a) || m_pf_fiducial->contained(b)) { return true; }
      const auto d = b - a;
      const double step = 5 * units::cm;
      const int n = (int)std::min(d.magnitude() / step, 20000.0);
      for (int i = 1; i < n; ++i) {
        if (m_pf_fiducial->contained(a + d * (i / (double)n))) { return true; }
      }
      return false;
    };
    auto kept = [&](const simb::MCParticle& p) {
      if ((p.E() - p.Mass()) * units::GeV <= m_pf_ke_min) { return false; }
      auto it = tid2idx.find(p.TrackId());
      const bool is_nu = it != tid2idx.end() && it->second < nu_origin.size() &&
                         nu_origin[it->second];
      // beam-nu-derived particles skip the FV cut
      if (is_nu) { return true; }
      // pf_nu_only (default true): nothing but beam-nu-derived particles --
      // in particular no FV-crossing cosmic muons.
      if (m_pf_nu_only) { return false; }
      return fv_ok(p);
    };
    std::unordered_map<int, std::vector<int>> pf_children; // kept parent -> kept kids
    std::map<int, std::vector<int>> pf_nu_roots; // nu_index -> interaction-level tids
    std::vector<int> pf_roots;                   // non-beam (cosmic) roots
    for (const auto& p : *mcps) {
      if (!kept(p)) { continue; }
      int anc = p.Mother();
      while (anc > 0) {
        auto it = by_tid.find(anc);
        if (it == by_tid.end()) { anc = 0; break; }
        if (kept(*it->second)) { break; }
        anc = it->second->Mother();
      }
      if (anc > 0 && by_tid.count(anc)) {
        pf_children[anc].push_back(p.TrackId());
        continue;
      }
      // interaction-level particle: group under its mother neutrino node
      const auto iit = tid2idx.find(p.TrackId());
      const int nidx = (iit != tid2idx.end() && iit->second < nu_index.size())
                         ? nu_index[iit->second] : -1;
      if (nidx >= 0) { pf_nu_roots[nidx].push_back(p.TrackId()); }
      else { pf_roots.push_back(p.TrackId()); }
    }
    std::function<Configuration(int)> make_pf_node = [&](int tid) -> Configuration {
      const auto& p = *by_tid.at(tid);
      Configuration node;
      node["id"] = tid;
      char text[64];
      std::snprintf(text, sizeof(text), "%s  %.1f MeV",
                    pdg_name(p.PdgCode()).c_str(), (p.E() - p.Mass()) * 1e3);
      node["text"] = text;
      Configuration dj;
      const auto& s4 = p.Position(0);
      const auto& e4 = p.EndPosition();
      dj["start"][0] = s4.X(); // already cm, as Bee wants
      dj["start"][1] = s4.Y();
      dj["start"][2] = s4.Z();
      dj["end"][0] = e4.X();
      dj["end"][1] = e4.Y();
      dj["end"][2] = e4.Z();
      node["data"] = dj;
      node["children"] = Json::arrayValue;
      auto cit = pf_children.find(tid);
      if (cit != pf_children.end()) {
        for (int ctid : cit->second) {
          node["children"].append(make_pf_node(ctid));
        }
      }
      if (node["children"].empty()) { node["icon"] = "jstree-file"; }
      return node;
    };
    // One root node per beam-nu interaction that has kept particles: the
    // initial mother neutrino, with start = end = the interaction vertex.
    for (const auto& [nidx, tids] : pf_nu_roots) {
      Configuration nu_node;
      // synthetic id: below the 1e7 GENIE trackid-offset range, unique per
      // interaction, no collision with G4 trackids.
      nu_node["id"] = 9000000 + nidx;
      char text[64];
      if (nidx < (int)nutruths.size() && nutruths[nidx].valid) {
        const auto& nn = nutruths[nidx];
        // node energy = the interaction's DEPOSITED energy (MeV), not the
        // neutrino total energy.
        const double edep_mev = m_nu_edep.count(nidx) ? m_nu_edep.at(nidx) : 0.0;
        std::snprintf(text, sizeof(text), "%s  Edep %.1f MeV",
                      pdg_name(nn.pdg).c_str(), edep_mev);
        Configuration dj;
        dj["start"][0] = nn.vx; // cm, as Bee wants
        dj["start"][1] = nn.vy;
        dj["start"][2] = nn.vz;
        dj["end"] = dj["start"];
        nu_node["data"] = dj;
      }
      else {
        std::snprintf(text, sizeof(text), "nu interaction %d", nidx);
      }
      nu_node["text"] = text;
      nu_node["children"] = Json::arrayValue;
      for (int tid : tids) {
        nu_node["children"].append(make_pf_node(tid));
      }
      m_pf_particles.append(nu_node);
    }
    for (int tid : pf_roots) {
      m_pf_particles.append(make_pf_node(tid));
    }
  }

  // --- energy deposits ---
  art::Handle<std::vector<sim::SimEnergyDeposit>> seds;
  if (event.getByLabel(art::InputTag{m_deposet_label}, seds) && seds.isValid()) {
    m_depos.reserve(seds->size());
    const bool do_sce = m_sce && m_sce_correction;
    size_t nsce = 0;
    for (const auto& sed : *seds) {
      Depo d;
      d.x = sed.MidPointX() * units::cm;
      d.y = sed.MidPointY() * units::cm;
      d.z = sed.MidPointZ() * units::cm;
      d.t = sed.Time() * units::ns;
      // larg4 marks deposits from DROPPED (unsaved) descendants -- e.g.
      // delta rays along a muon -- with the NEGATIVE of the saved ancestor
      // trackid.  Fold them into the ancestor, else the blob sections where
      // delta-ray charge dominates come out "unlabeled" (tid<0).
      d.trackid = std::abs(sed.TrackID());
      d.weight = sed.NumElectrons() > 0 ? (double)sed.NumElectrons() : sed.Energy();
      if (do_sce) {
        // "postSCE" on the fly: shift the true position by the TrueFwd
        // (true->reco) displacement of the depo's TPC.
        for (const auto& [af, fc] : m_faces) {
          if (d.x < fc.xmin || d.x > fc.xmax) { continue; }
          const double x0 = d.x, y0 = d.y, z0 = d.z;
          d.x += m_sce->displacement_x(af.first, x0, y0, z0);
          d.y += m_sce->displacement_y(af.first, x0, y0, z0);
          d.z += m_sce->displacement_z(af.first, x0, y0, z0);
          ++nsce;
          break;
        }
      }
      m_depos.push_back(d);
    }
    if (do_sce) {
      log->debug("SCE-shifted {}/{} depos to reco positions", nsce, m_depos.size());
    }
  }
  else {
    log->warn("failed to fetch SimEnergyDeposits with label '{}'", m_deposet_label);
  }

  const std::string flav0 =
    m_evtmd["nu_flavor"].size() ? m_evtmd["nu_flavor"][0u].asString() : "none";
  log->debug("visit run {} sub {} evt {}: n_nu {}, main flavor {}, "
             "main nu_edep {:.1f} MeV, {} tracks, {} depos, "
             "{} Michel e- (Bee merge to mother muon: {})",
             m_run, m_sub, m_evt,
             m_evtmd["n_nu"].asInt(),
             flav0,
             m_nu_edep.count(0) ? m_nu_edep.at(0) : 0.0,
             m_tracks.size(),
             m_depos.size(),
             m_michel_mother.size(),
             m_bee_michel_merge ? "on" : "off");
}

// format a "path/%d" style path (cf. MABC format_path, no subpath map)
static std::string format_path(std::string path, int ident)
{
  if (path.find("%") == std::string::npos) { return path; }
  return String::format(path, ident);
}

bool AIML::TensorSetLabeler::operator()(const input_pointer& in, output_pointer& out)
{
  out = nullptr;
  if (!in) {
    log->debug("EOS at call={}", m_count++);
    return true;
  }
  const int ident = in->ident();
  const std::string livepath = format_path(m_inpath, ident) + "/" + m_grouping;

  // Deserialize the grouping point tree.
  const auto& intens = *in->tensors();
  std::unique_ptr<PointCloud::Tree::Points::node_t> root;
  try {
    root = as_pctree(intens, livepath);
  }
  catch (WireCell::KeyError& err) {
    log->warn("call={} no pc-tree at datapath '{}', passing through unlabeled", m_count, livepath);
    out = in;
    ++m_count;
    return true;
  }

  // Project each depo onto its anode face: (apa,face) -> tick -> depos with
  // per-plane wire-in-plane indices.  This mirrors BlobDepoFill's
  // slice-and-dice but point-like (SimEnergyDeposits carry no diffusion
  // extents; depo step size << blob size).
  struct PDepo {
    int wip[3];
    int trackid;
    double weight;
    double stick;    // diffusion+SP sigma along drift, in ticks
    double swire[3]; // diffusion+SP sigma along pitch, in wires, per plane
  };
  std::map<std::pair<int, int>, std::unordered_map<int, std::vector<PDepo>>> proj;
  size_t nproj = 0;
  // SCE-shifted SimEnergyDeposit cloud, drawn at the DRIFTED (apparent)
  // position that directly fills the blobs: x_app (drift + t_dep shift) at
  // the post-SCE y,z -- overlays the raw-coordinate blob points exactly.
  // Bee cluster_id "trackid merging": display a Michel electron's charge
  // under its mother muon so decay electrons don't fragment the muon
  // cluster.  Affects the Bee display only -- the blob scalar PC keeps the
  // true (Michel) trackid.
  auto bee_cid = [&](int t) -> int {
    if (m_bee_michel_merge && t >= 0) {
      auto it = m_michel_mother.find(t);
      if (it != m_michel_mother.end()) { return it->second; }
    }
    return t;
  };
  Bee::Points bpts_depo(m_bee_detector, m_bee_depo_algorithm);
  bpts_depo.rse(m_run, m_sub, m_evt);
  const bool dump_depo = m_bee_sink && m_sce && m_sce_correction;
  const int nsample = std::max(1, m_nsample_depo);
  std::normal_distribution<double> gaus(0.0, 1.0);
  double max_stick = 0; // widest depo time-sigma, sets the blob tick window
  for (const auto& d : m_depos) {
    const double tdep = d.t + m_depo_time_offset;
    for (const auto& [af, fc] : m_faces) {
      // apparent (drifted) x: later deposit -> deeper into the volume
      const double x_app = d.x + fc.dirx * m_drift_speed * tdep;
      if (d.x < fc.xmin || d.x > fc.xmax) { continue; } // wrong TPC
      const double t_sig = (x_app - fc.xw) * fc.dirx / m_drift_speed - m_time_offset;
      const int itick = (int)std::floor(t_sig / m_tick);
      PDepo pd;
      pd.trackid = d.trackid;
      pd.weight = d.weight;
      // Diffusion ball: drift diffusion sigma = sqrt(2*D*t_drift) from the
      // ACTUAL drift distance, plus the SP filter smearing, in quadrature.
      {
        const double t_drift = std::max(0.0, (d.x - fc.xw) * fc.dirx / m_drift_speed);
        const double sigL = std::sqrt(2 * m_DL * t_drift); // longitudinal (drift/time)
        const double sigT = std::sqrt(2 * m_DT * t_drift); // transverse (pitch)
        const double sig_time = std::sqrt(sigL / m_drift_speed * (sigL / m_drift_speed) +
                                          m_sp_smear_time * m_sp_smear_time);
        pd.stick = sig_time / m_tick;
        for (int ip = 0; ip < 3; ++ip) {
          const double spw = (ip < 2 ? m_sp_smear_wire_ind : m_sp_smear_wire_col) * fc.pitch[ip];
          pd.swire[ip] = std::sqrt(sigT * sigT + spw * spw) / fc.pitch[ip];
        }
        max_stick = std::max(max_stick, pd.stick);
      }
      // Draw only depos whose apparent position lands inside the readout
      // window (blobs only exist for ticks [0, nticks)); far-out-of-time
      // depos (e.g. radiologicals) would fly off to |x|~km otherwise.
      // Sample nsample points from the diffusion ball (q split evenly);
      // the transverse display sigma uses the U-plane (largest) value.
      if (dump_depo && itick >= -m_tick_slop && itick < m_nticks + m_tick_slop) {
        const double sig_x = pd.stick * m_tick * m_drift_speed;
        const double sig_yz = pd.swire[0] * fc.pitch[0];
        const int cid = bee_cid(d.trackid);
        for (int k = 0; k < nsample; ++k) {
          bpts_depo.append(Point(x_app + gaus(m_rng) * sig_x,
                                 d.y + gaus(m_rng) * sig_yz,
                                 d.z + gaus(m_rng) * sig_yz),
                           d.weight / nsample, cid, cid);
        }
      }
      const Point pos(d.x, d.y, d.z);
      // Wire-in-plane indices from the face RayGrid -- the same coordinates
      // the tiling used to define the blob strip bounds (layers 2,3,4 =
      // u,v,w).  Out-of-plane depos simply fail every blob's range test.
      const auto& coords = fc.face->raygrid();
      for (int ip = 0; ip < 3; ++ip) {
        const int layer = 2 + ip;
        const auto& pdir = coords.pitch_dirs()[layer];
        const auto& cen = coords.centers()[layer];
        const double pitch = (pos - cen).dot(pdir);
        pd.wip[ip] = coords.pitch_index(pitch, layer);
      }
      proj[af][itick].push_back(pd);
      ++nproj;
      break; // a depo lives in exactly one TPC
    }
  }

  // Walk grouping -> clusters -> blobs, assign dominant trackid.
  Bee::Points bpts(m_bee_detector, m_bee_algorithm);
  bpts.rse(m_run, m_sub, m_evt);
  Bee::Points bpts_unlab(m_bee_detector, m_bee_unlabeled_algorithm);
  bpts_unlab.rse(m_run, m_sub, m_evt);

  size_t nblobs = 0, nlabeled = 0;
  for (auto* cnode : root->children()) {
    // reco cluster ident, used as the cluster_id of the unlabeled dump
    int reco_clid = -1;
    {
      auto cit = cnode->value.local_pcs().find("cluster_scalar");
      if (cit != cnode->value.local_pcs().end()) {
        auto arr = cit->second.get("ident");
        if (arr) { reco_clid = arr->elements<int>()[0]; }
      }
    }
    for (auto* bnode : cnode->children()) {
      auto& lpcs = bnode->value.local_pcs();
      auto sit = lpcs.find("scalar");
      if (sit == lpcs.end()) { continue; }
      Dataset& scalar = sit->second;
      ++nblobs;

      const WirePlaneId wpid(scalar.get("wpid")->elements<int>()[0]);
      const int smin = scalar.get("slice_index_min")->elements<int>()[0];
      const int smax = scalar.get("slice_index_max")->elements<int>()[0];
      int wmin[3], wmax[3];
      const char* pnames[3] = {"u", "v", "w"};
      for (int ip = 0; ip < 3; ++ip) {
        wmin[ip] = scalar.get(std::string(pnames[ip]) + "_wire_index_min")->elements<int>()[0];
        wmax[ip] = scalar.get(std::string(pnames[ip]) + "_wire_index_max")->elements<int>()[0];
      }

      int tid = -1;
      auto pit = proj.find({wpid.apa(), wpid.face()});
      if (pit != proj.end()) {
        std::unordered_map<int, double> acc;
        // Gaussian acceptance: a depo fills the blob when its center lies
        // within (slop + nsigma*sigma) of the blob bounds, per dimension.
        const int twin = m_tick_slop + (int)std::ceil(m_nsigma * max_stick);
        for (int itick = smin - twin; itick < smax + twin; ++itick) {
          auto bit = pit->second.find(itick);
          if (bit == pit->second.end()) { continue; }
          for (const auto& pd : bit->second) {
            const double tout = itick < smin  ? (double)(smin - itick) :
                                itick >= smax ? (double)(itick - smax + 1) : 0.0;
            if (tout > m_tick_slop + m_nsigma * pd.stick) { continue; }
            bool inside = true;
            for (int ip = 0; ip < 3; ++ip) {
              const double wout = pd.wip[ip] < wmin[ip]  ? (double)(wmin[ip] - pd.wip[ip]) :
                                  pd.wip[ip] >= wmax[ip] ? (double)(pd.wip[ip] - wmax[ip] + 1) : 0.0;
              if (wout > m_wire_slop + m_nsigma * pd.swire[ip]) {
                inside = false;
                break;
              }
            }
            if (inside) { acc[pd.trackid] += pd.weight; }
          }
        }
        double best = 0;
        for (const auto& [t, w] : acc) {
          if (w > best) {
            best = w;
            tid = t;
          }
        }
      }

      // write back into the scalar PC
      if (scalar.has("trackid")) {
        const int v = tid;
        scalar.get("trackid")->assign(&v, PointCloud::Array::shape_t{1}, false);
      }
      else {
        scalar.add("trackid", Array({(int)tid}));
      }
      if (tid >= 0) { ++nlabeled; }

      // debug Bee dump: every 3d point in raw coords, cluster_id = trackid
      if (m_bee_sink) {
        auto dit = lpcs.find("3d");
        if (dit != lpcs.end() && dit->second.size_major() > 0) {
          const auto x = dit->second.get("x")->elements<double>();
          const auto y = dit->second.get("y")->elements<double>();
          const auto z = dit->second.get("z")->elements<double>();
          const double q = scalar.get("charge")->elements<double>()[0];
          const double qpp = x.size() ? std::max(q / x.size(), 1.0) : 1.0;
          const int cid = bee_cid(tid);
          for (size_t i = 0; i < x.size(); ++i) {
            if (tid >= 0) {
              bpts.append(Point(x[i], y[i], z[i]), qpp, cid, cid);
            }
            else {
              bpts_unlab.append(Point(x[i], y[i], z[i]), qpp, reco_clid, reco_clid);
            }
          }
        }
      }
    }
  }

  // Re-serialize the labeled grouping; pass through all other tensors.
  ITensor::vector outtens = as_tensors(*root, livepath);
  const std::string liveprefix = livepath + "/";
  for (const auto& ten : intens) {
    const auto dp = ten->metadata()["datapath"].asString();
    if (dp == livepath || dp.rfind(liveprefix, 0) == 0) { continue; }
    outtens.push_back(ten);
  }

  // truth_per_track tensor: [ntracks x ncols] doubles.
  {
    const size_t nrows = m_tracks.size();
    const size_t ncols = track_columns.size();
    std::vector<double> flat(nrows * ncols, 0.0);
    for (size_t r = 0; r < nrows; ++r) {
      for (size_t c = 0; c < ncols && c < m_tracks[r].size(); ++c) {
        flat[r * ncols + c] = m_tracks[r][c];
      }
    }
    Json::Value md = Json::objectValue;
    md["datapath"] = format_path(m_truth_datapath, ident);
    md["datatype"] = "truth_per_track";
    md["units"] = "cm, ns, GeV";
    for (const auto& col : track_columns) {
      md["columns"].append(col);
    }
    outtens.push_back(
      std::make_shared<SimpleTensor>(std::vector<size_t>{nrows, ncols}, flat.data(), md));
  }

  // Event-level metadata.
  Configuration set_md = in->metadata();
  set_md["runNo"] = m_run;
  set_md["subRunNo"] = m_sub;
  set_md["eventNo"] = m_evt;
  for (const auto& key : m_evtmd.getMemberNames()) {
    set_md[key] = m_evtmd[key];
  }

  out = std::make_shared<SimpleTensorSet>(ident, set_md,
                                          std::make_shared<ITensor::vector>(outtens));

  if (m_bee_sink) {
    m_bee_sink->write(bpts, m_bee_index, m_run, m_sub, m_evt);
    m_bee_sink->write(bpts_unlab, m_bee_index, m_run, m_sub, m_evt);
    if (!bpts_depo.empty()) {
      m_bee_sink->write(bpts_depo, m_bee_index, m_run, m_sub, m_evt);
    }
    if (!m_pf_particles.empty()) {
      Bee::ParticleTree pf(m_bee_pf_name);
      pf.set_particles(m_pf_particles);
      m_bee_sink->write(pf, m_bee_index, m_run, m_sub, m_evt);
    }
    ++m_bee_index;
  }

  log->debug("call={} ident={} rse=({},{},{}): projected {}/{} depos, "
             "labeled {}/{} blobs, {} truth tracks",
             m_count, ident, m_run, m_sub, m_evt,
             nproj, m_depos.size(), nlabeled, nblobs, m_tracks.size());
  ++m_count;
  return true;
}
