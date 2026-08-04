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
#include "WireCellClus/Facade.h"
#include "WireCellClus/Graphs.h"

#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "canvas/Persistency/Common/FindOneP.h"
#include "canvas/Utilities/InputTag.h"
#include "lardataobj/Simulation/SimEnergyDeposit.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "nusimdata/SimulationBase/MCTruth.h"

#include <hdf5.h>
#include <boost/graph/adjacency_list.hpp>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Human-readable name for simb::MCNeutrino::InteractionType() (the "mc" tree
// text).  Handles both the base int_type_ enum (0..13) and the Nuance-offset
// codes (1000+) GENIE writes -- families collapsed to a short mode name.
// (nusimdata SimulationBase/MCNeutrino.h.)
static std::string int_type_name(int t)
{
  switch (t) {
    case -1: return "unknown";
    case 0:  return "QE";     case 1:  return "RES";   case 2:  return "DIS";
    case 3:  return "COH";    case 4:  return "CohElastic";
    case 5:  return "NuEEL";  case 6:  return "IMDAnn"; case 7:  return "IBD";
    case 8:  return "Glashow";case 9:  return "AMNuGamma";
    case 10: return "MEC";    case 11: return "Diffractive";
    case 12: return "EM";     case 13: return "WeakMix";
  }
  // Nuance-offset codes: InteractionType = kNuanceOffset(1000) + NuanceReactionCode
  // (GENIE2ART.cxx).  MEC/2p2h has no NUANCE code -> reaction code 0 -> exactly 1000.
  if (t == 1000 || t == 1010)              return "MEC";
  if (t == 1001 || t == 1002 || t == 1095) return "QE";   // (N)CQE, CCQEHyperon
  if (t >= 1003 && t <= 1090)              return "RES";
  if (t == 1091 || t == 1092)              return "DIS";
  if (t == 1096 || t == 1097)              return "COH";
  if (t == 1098)                           return "NuEEL";
  if (t == 1099)                           return "IMD";
  char buf[24];
  std::snprintf(buf, sizeof(buf), "int %d", t);
  return buf;
}

AIML::TensorSetLabeler::TensorSetLabeler()
  : Aux::Logger("TensorSetLabeler", "aiml")
  , m_drift_speed(1.563 * units::mm / units::us)
  , m_time_offset(-205 * units::us)
  , m_tick(0.5 * units::us)
  // readout window = [tick0_time, tick0_time + nticks*tick], SBND nticks=3427
  , m_readout_tmin(-205 * units::us)
  , m_readout_tmax(1508.5 * units::us)
  , m_DL(4.0 * units::cm * units::cm / units::s)   // wcsimsp_sbnd.fcl DL
  , m_DT(8.8 * units::cm * units::cm / units::s)   // wcsimsp_sbnd.fcl DT
  , m_sp_smear_time(1.0 / (2 * 3.141592653589793 * 0.10 * units::megahertz))
  , m_pf_ke_min(10 * units::MeV)
  , m_ctpc_x_tol(5.0 * units::mm)     // pywcml config x_tolerance
  , m_ctpc_pitch_gap(6.0 * units::mm) // pywcml config pitch_gap_tolerance
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
  cfg["reality"] = m_reality;
  // MUST match the BlobSampler configuration that made the "3d" PCs.
  cfg["drift_speed"] = m_drift_speed;
  cfg["time_offset"] = m_time_offset;
  // Residual shift ADDED to depo times before drift conversion (see header).
  cfg["depo_time_offset"] = m_depo_time_offset;
  cfg["tick"] = m_tick;
  // "Readout" pseudo-sim cut: keep a depo only if its pseudo-sim time
  // (x_app - x_W)*dirx/drift_speed (= t_sig + time_offset, the trigger-frame
  // slice time; tick 0 at time_offset) is within this window.  SBND default
  // [-205us, 1508.5us] = [tick0_time, tick0_time + nticks*tick] (nticks=3427).
  cfg["readout_time_min"] = m_readout_tmin;
  cfg["readout_time_max"] = m_readout_tmax;
  // Acceptance slop (in wires / ticks) around the blob bounds, standing in
  // for the diffusion extents BlobDepoFill integrates (point-like depos).
  cfg["wire_slop"] = m_wire_slop;
  cfg["tick_slop"] = m_tick_slop;
  // legacy (superseded by readout_time_min/max); kept for config back-compat.
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
  // SED pseudo-sim Bee sets: Gaussian samples per depo diffusion ball.
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
  // nugraph HDF5 output (see HDF5 OUTPUT in the header).
  cfg["hdf5_output"] = m_hdf5_output;
  cfg["hdf5_filename"] = m_hdf5_filename;
  cfg["plane_knn"] = m_plane_knn;
  // IDetectorVolumes + IPCTransformSet for the "ctpc" blob-blob graph flavor.
  cfg["detector_volumes"] = "";
  cfg["pc_transforms"] = "";
  cfg["anodes"] = Json::arrayValue;
  cfg["bee_sink"] = "";
  cfg["bee_detector"] = m_bee_detector;
  cfg["bee_algorithm"] = m_bee_algorithm;
  cfg["tagger_coords"] = Json::arrayValue;
  cfg["beam_window"][0] = m_beam_window_low;
  cfg["beam_window"][1] = m_beam_window_high;
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
  m_reality = get(cfg, "reality", m_reality);
  if (cfg["tagger_coords"].isArray() && cfg["tagger_coords"].size() == 3) {
    m_tagger_coords.clear();
    for (const auto& c : cfg["tagger_coords"]) m_tagger_coords.push_back(c.asString());
  }
  if (cfg["beam_window"].isArray() && cfg["beam_window"].size() == 2) {
    m_beam_window_low  = cfg["beam_window"][0].asDouble();
    m_beam_window_high = cfg["beam_window"][1].asDouble();
  }
  m_drift_speed = get(cfg, "drift_speed", m_drift_speed);
  m_time_offset = get(cfg, "time_offset", m_time_offset);
  m_depo_time_offset = get(cfg, "depo_time_offset", m_depo_time_offset);
  m_tick = get(cfg, "tick", m_tick);
  m_readout_tmin = get(cfg, "readout_time_min", m_readout_tmin);
  m_readout_tmax = get(cfg, "readout_time_max", m_readout_tmax);
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
  m_hdf5_output = get(cfg, "hdf5_output", m_hdf5_output);
  m_hdf5_filename = get(cfg, "hdf5_filename", m_hdf5_filename);
  m_plane_knn = get(cfg, "plane_knn", m_plane_knn);
  {
    const std::string dv_tn = get<std::string>(cfg, "detector_volumes", "");
    const std::string pcts_tn = get<std::string>(cfg, "pc_transforms", "");
    m_dv = dv_tn.empty() ? nullptr : Factory::find_tn<IDetectorVolumes>(dv_tn);
    m_pcts = pcts_tn.empty() ? nullptr : Factory::find_tn<Clus::IPCTransformSet>(pcts_tn);
  }

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
  log->debug("readout pseudo-sim-t window [{}, {}] us (tick 0 = time_offset {} us)",
             m_readout_tmin / units::us, m_readout_tmax / units::us,
             m_time_offset / units::us);
}

void AIML::TensorSetLabeler::finalize()
{
  if (m_hdf5_output) {
    write_hdf5();
  }
  if (m_bee_sink) {
    m_bee_sink->release();
    m_bee_sink = nullptr;
  }
}

// Write the accumulated per-event graphs to a pynuml H5DataModule container
// (raw HDF5 C API, cf. Truth2h5.cxx).  Each event is one SCALAR compound
// record at /dataset/<sample_name>; every node/edge store is one compound
// member whose name embeds a '/', which HDF5 allows for member names.
void AIML::TensorSetLabeler::write_hdf5()
{
  hid_t file = H5Fcreate(m_hdf5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) {
    log->error("nugraph: cannot create hdf5 file {}", m_hdf5_filename);
    return;
  }
  hid_t vstr = H5Tcopy(H5T_C_S1);
  H5Tset_size(vstr, H5T_VARIABLE);

  auto write_strvec = [&](const char* name, const std::vector<std::string>& v) {
    hsize_t dim = v.size();
    hid_t sp = H5Screate_simple(1, &dim, NULL);
    std::vector<const char*> p;
    for (auto& s : v) { p.push_back(s.c_str()); }
    hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
    H5Pset_create_intermediate_group(lcpl, 1);
    hid_t d = H5Dcreate2(file, name, vstr, sp, lcpl, H5P_DEFAULT, H5P_DEFAULT);
    if (dim) { H5Dwrite(d, vstr, H5S_ALL, H5S_ALL, H5P_DEFAULT, p.data()); }
    H5Dclose(d); H5Pclose(lcpl); H5Sclose(sp);
  };
  auto write_i64vec = [&](const char* name, const std::vector<long long>& v) {
    hsize_t dim = v.size();
    hid_t sp = H5Screate_simple(1, &dim, NULL);
    hid_t d = H5Dcreate2(file, name, H5T_NATIVE_LLONG, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dim) { H5Dwrite(d, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()); }
    H5Dclose(d); H5Sclose(sp);
  };

  // Top-level H5DataModule datasets.  All events go to the "train" split by
  // default (datasize = [ntrain, nval, ntest]).
  write_strvec("/planes", {"u", "v", "y"});
  write_strvec("/semantic_classes", {"nu", "cosmic"});
  write_i64vec("/gen", {3});
  const long long N = (long long)m_events.size();
  write_i64vec("/datasize", {N, 0, 0});
  std::vector<std::string> names;
  for (auto& ev : m_events) { names.push_back(ev.sample_name); }
  write_strvec("/samples/train", names);
  write_strvec("/samples/val", {});
  write_strvec("/samples/test", {});

  // One scalar compound record per event.
  for (auto& ev : m_events) {
    std::vector<size_t> offs;
    size_t total = 0;
    for (auto& m : ev.members) {
      size_t n = 1;
      for (auto dd : m.dims) { n *= (size_t)dd; }
      offs.push_back(total);
      total += n * (m.is_float ? sizeof(float) : sizeof(long long));
    }
    hid_t ctype = H5Tcreate(H5T_COMPOUND, total);
    for (size_t k = 0; k < ev.members.size(); ++k) {
      auto& m = ev.members[k];
      hid_t base = m.is_float ? H5T_NATIVE_FLOAT : H5T_NATIVE_LLONG;
      hid_t mt;
      if (m.dims.empty()) {
        mt = H5Tcopy(base);
      }
      else {
        std::vector<hsize_t> ad(m.dims.begin(), m.dims.end());
        mt = H5Tarray_create2(base, (unsigned)ad.size(), ad.data());
      }
      H5Tinsert(ctype, m.name.c_str(), offs[k], mt);
      H5Tclose(mt);
    }
    std::vector<char> buf(total);
    for (size_t k = 0; k < ev.members.size(); ++k) {
      auto& m = ev.members[k];
      if (m.is_float) { std::memcpy(buf.data() + offs[k], m.f.data(), m.f.size() * sizeof(float)); }
      else { std::memcpy(buf.data() + offs[k], m.i.data(), m.i.size() * sizeof(long long)); }
    }
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
    H5Pset_create_intermediate_group(lcpl, 1);
    const std::string path = "/dataset/" + ev.sample_name;
    hid_t d = H5Dcreate2(file, path.c_str(), ctype, sp, lcpl, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(d, ctype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(d); H5Pclose(lcpl); H5Sclose(sp); H5Tclose(ctype);
  }

  H5Tclose(vstr);
  H5Fclose(file);
  log->debug("nugraph: wrote {} with {} event record(s)", m_hdf5_filename, m_events.size());
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
  m_nu_trackids.clear();

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

  // DATA mode: no MC truth exists.  Keep only run/subrun/event (captured
  // above) and the empty nu_* metadata; skip every MCTruth/MCParticle/
  // SimEnergyDeposit read.  operator() then builds an input-only HDF5 graph
  // (reco nodes/edges, truth fields = sentinels) and no Bee/truth_per_track.
  if (m_reality != "sim") {
    log->debug("visit run {} sub {} evt {}: DATA mode (RSE only, no truth)",
               m_run, m_sub, m_evt);
    return;
  }

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
            m_nu_trackids.insert((*mcps)[i].TrackId()); // for node semantic labels
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
    int int_type{-1};   // simb::MCNeutrino::InteractionType()
    double vx{0}, vy{0}, vz{0};
    double etot{0};     // incoming neutrino total energy [MeV]
    double time{0};     // interaction time [us] (nu vertex 4-position T)
  };
  std::vector<NuNode> nutruths;
  if (mctruth_handle.isValid()) {
    for (const auto& mct : *mctruth_handle) {
      NuNode nn;
      if (mct.NeutrinoSet()) {
        const auto& nu_p = mct.GetNeutrino().Nu();
        nn.valid = true;
        nn.pdg = nu_p.PdgCode();
        nn.int_type = mct.GetNeutrino().InteractionType();
        const auto& pos = nu_p.Position(0);
        nn.vx = pos.X();
        nn.vy = pos.Y();
        nn.vz = pos.Z();
        nn.etot = nu_p.Momentum(0).E() * 1e3; // GeV -> MeV (original total nu energy)
        nn.time = pos.T() * 1e-3;             // ns -> us (interaction time)
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
      char text[160];
      if (nidx < (int)nutruths.size() && nutruths[nidx].valid) {
        const auto& nn = nutruths[nidx];
        // text = "<nu_idx> <nu type> <interaction type> Etot <total nu E> MeV
        //         Edep <deposited> MeV T <interaction time> us".
        // nu_idx is the 1-based Bee convention (matches the sed point sets:
        // 1,2,... per beam-nu interaction); Etot is the incoming neutrino total
        // energy, Edep the interaction's deposited (visible) energy, T the
        // interaction time (nu vertex 4-position T, us).
        const double edep_mev = m_nu_edep.count(nidx) ? m_nu_edep.at(nidx) : 0.0;
        std::snprintf(text, sizeof(text),
                      "%d %s %s Etot %.1f MeV Edep %.1f MeV T %.3f us",
                      nidx + 1, pdg_name(nn.pdg).c_str(),
                      int_type_name(nn.int_type).c_str(), nn.etot, edep_mev, nn.time);
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
      // per-point charge (q) and energy (e) carried onto the sed Bee sets.
      d.q = (double)sed.NumElectrons();
      d.e = sed.Energy(); // MeV
      // neutrino-interaction index for the sed points: |trackid| -> mcps index
      // -> beam-nu MCTruth key (0-based, -1 = not beam), remapped to the Bee
      // convention 0 = non-neutrino activity, 1,2,... = beam-nu interactions.
      {
        const auto iit = tid2idx.find(d.trackid);
        const int nidx = (iit != tid2idx.end() && iit->second < nu_index.size())
                           ? nu_index[iit->second] : -1;
        d.nu_idx = (nidx >= 0) ? nidx + 1 : 0;
      }
      // keep the TRUE (pre-SCE) position for the smear-only pseudo-sim set
      d.x0 = d.x;
      d.y0 = d.y;
      d.z0 = d.z;
      if (do_sce) {
        // "postSCE" on the fly: shift the true position by the TrueFwd
        // (true->reco) displacement of the depo's TPC.
        for (const auto& [af, fc] : m_faces) {
          if (d.x0 < fc.xmin || d.x0 > fc.xmax) { continue; }
          d.x += m_sce->displacement_x(af.first, d.x0, d.y0, d.z0);
          d.y += m_sce->displacement_y(af.first, d.x0, d.y0, d.z0);
          d.z += m_sce->displacement_z(af.first, d.x0, d.y0, d.z0);
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
  // DATA mode: no truth.  m_depos/m_tracks/m_nu_* are empty (visit returned
  // early), so blob trackid comes out -1, the HDF5 truth fields become
  // sentinels, and the Bee truth sets + truth_per_track tensor are skipped.
  const bool is_sim = (m_reality == "sim");

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
  // Three SED pseudo-sim clouds (see PSEUDO-SIM in the header):
  //   sed-sce_drift_smear_readout: all 4 effects (overlays the blobs),
  //   sed-smear_readout: smear + readout only, at the TRUE position,
  //   sed-sce_smear_readout: SCE + smear + readout (readout cut on the full
  //     SCE+drift time), drawn at the SCE-only position (no drift-x shift).
  Bee::Points bpts_depo(m_bee_detector, m_bee_depo_algorithm);
  bpts_depo.rse(m_run, m_sub, m_evt);
  Bee::Points bpts_sr(m_bee_detector, m_bee_sr_algorithm);
  bpts_sr.rse(m_run, m_sub, m_evt);
  Bee::Points bpts_ssr(m_bee_detector, m_bee_ssr_algorithm);
  bpts_ssr.rse(m_run, m_sub, m_evt);
  const bool dump_sdsr = is_sim && m_bee_sink && m_sce && m_sce_correction; // needs SCE chain
  const bool dump_sr = is_sim && (bool)m_bee_sink;                          // sim only
  const bool dump_ssr = is_sim && m_bee_sink && m_sce && m_sce_correction;  // needs SCE chain
  const int nsample = std::max(1, m_nsample_depo);
  std::normal_distribution<double> gaus(0.0, 1.0);
  double max_stick = 0; // widest depo time-sigma, sets the blob tick window
  // Ball sampler: apply the "readout" cut on the pseudo-sim time
  //   pseudo_t = (x_app - x_W)*dirx/drift_speed  (= t_sig + time_offset)
  // then Gaussian-sample the diffusion+SP ball (sigma from x_drift's drift
  // distance) into the given Bee set.  q split evenly across samples.
  auto dump_ball = [&](Bee::Points& bp, const FaceCtx& fc, double x_cut,
                       double x_point, double yv, double zv, double x_drift,
                       int trackid, double q, double e, int nu_idx) {
    // Readout cut on the FULL pseudo-sim (SCE+drift) apparent time (x_cut) --
    // even for the smear-only set, whose points then sit at the TRUE x_point.
    const double pseudo_t = (x_cut - fc.xw) * fc.dirx / m_drift_speed;
    if (pseudo_t < m_readout_tmin || pseudo_t > m_readout_tmax) { return; } // readout cut
    const double t_drift = std::max(0.0, (x_drift - fc.xw) * fc.dirx / m_drift_speed);
    const double sigL = std::sqrt(2 * m_DL * t_drift);
    const double sigT = std::sqrt(2 * m_DT * t_drift);
    const double sig_time = std::sqrt(sigL / m_drift_speed * (sigL / m_drift_speed) +
                                      m_sp_smear_time * m_sp_smear_time);
    const double sig_x = sig_time * m_drift_speed;
    const double spw = m_sp_smear_wire_ind * fc.pitch[0]; // U-plane transverse
    const double sig_yz = std::sqrt(sigT * sigT + spw * spw);
    const int cid = bee_cid(trackid);
    for (int k = 0; k < nsample; ++k) {
      bp.append(Point(x_point + gaus(m_rng) * sig_x,
                      yv + gaus(m_rng) * sig_yz,
                      zv + gaus(m_rng) * sig_yz),
                q / nsample, cid, cid, e / nsample, nu_idx);
    }
  };
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
      // "readout" pseudo-sim: keep only depos whose pseudo-sim time is in the
      // window (far-out-of-time depos otherwise fly off to |x|~km).
      //   set 1 (SCE+drift+smear+readout): apparent x_app at the post-SCE y,z,
      //   set 2 (smear+readout): the TRUE position, no SCE, no drift shift.
      if (dump_sdsr) {
        // all 4 effects: readout cut AND point both at the SCE+drift apparent x.
        dump_ball(bpts_depo, fc, x_app, x_app, d.y, d.z, d.x,
                  d.trackid, d.q, d.e, d.nu_idx);
      }
      if (dump_sr) {
        // smear+readout only: readout cut on the FULL pseudo-sim time (x_app),
        // but the point stays at the TRUE (pre-SCE, pre-drift) position (x0).
        dump_ball(bpts_sr, fc, x_app, d.x0, d.y0, d.z0, d.x0,
                  d.trackid, d.q, d.e, d.nu_idx);
      }
      if (dump_ssr) {
        // SCE+smear+readout: readout cut on the FULL SCE+drift time (x_app),
        // but the point sits at the SCE reco position (d.x, no drift-x shift);
        // diffusion sigma from the SCE reco drift distance (d.x).  Isolates the
        // SCE displacement from the drift-x apparent shift of sed-sce_drift_....
        dump_ball(bpts_ssr, fc, x_app, d.x, d.y, d.z, d.x,
                  d.trackid, d.q, d.e, d.nu_idx);
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
  // Cosmic/containment tagger verdicts (per-cluster flags flag_STM/flag_TGM/
  // flag_FC, set upstream by TaggerCheckSTM/TGM/FC).  One Bee set each, same
  // 3d points as clustering_global, cluster_id = 0 (untagged) / 1 (tagged),
  // real_cluster_id left 0 so Bee colors by the tag.  Dumped in sim AND data.
  Bee::Points bpts_stm(m_bee_detector, "tagger_stm"); bpts_stm.rse(m_run, m_sub, m_evt);
  Bee::Points bpts_tgm(m_bee_detector, "tagger_tgm"); bpts_tgm.rse(m_run, m_sub, m_evt);
  Bee::Points bpts_fc (m_bee_detector, "tagger_fc");  bpts_fc.rse(m_run, m_sub, m_evt);
  // LM (light-mismatch) verdict from QLMatching's lm_tagger: the cluster scalar
  // "lm_flag" (0 pass / 1 low-energy / 2 light-mismatch).  Binary encoding to
  // match fc/stm/tgm: cluster_id = 1 iff lm_flag==2 (the nusel LM label), else 0.
  Bee::Points bpts_lm (m_bee_detector, "tagger_lm");  bpts_lm.rse(m_run, m_sub, m_evt);

  size_t nblobs = 0, nlabeled = 0;
  for (auto* cnode : root->children()) {
    // reco cluster ident, used as the cluster_id of the unlabeled dump
    int reco_clid = -1;
    // Tagger Bee sets contain ONLY the clusters the taggers actually evaluate --
    // the "beam-window candidates": Flags::main_cluster clusters (QLMatching flags
    // the main of every matched flash bundle) whose cluster_t0 is in the beam gate
    // [low, high).  Out-of-window mains and non-mains (associated / unmatched) are
    // omitted entirely, so the display is just the candidates, colored by verdict:
    //   cluster_id = 0 (not tagged) / 1 (tagged, flag_STM/TGM/FC set).
    bool tagger_candidate = false;
    int tag_stm = 0, tag_tgm = 0, tag_fc = 0, tag_lm = 0;
    {
      auto cit = cnode->value.local_pcs().find("cluster_scalar");
      if (cit != cnode->value.local_pcs().end()) {
        auto& cs = cit->second;
        auto arr = cs.get("ident");
        if (arr) { reco_clid = arr->elements<int>()[0]; }
        // per-cluster int flags (0 if the flag/PC is absent)
        auto rf = [&](const char* k) -> int {
          auto a = cs.get(k); return (a && a->size_major() > 0 && a->elements<int>()[0] != 0) ? 1 : 0; };
        const int is_main = rf("flag_main_cluster");
        // cluster_t0 (double; unmatched clusters carry -1e12 -> out of window)
        double t0 = 0;
        { auto a = cs.get("cluster_t0"); if (a && a->size_major() > 0) t0 = a->elements<double>()[0]; }
        const bool in_window = (t0 >= m_beam_window_low && t0 < m_beam_window_high);
        tagger_candidate = (is_main != 0) && in_window;
        tag_stm = rf("flag_STM");
        tag_tgm = rf("flag_TGM");
        tag_fc  = rf("flag_FC");
        // lm_flag is a 3-state int (0/1/2), not a boolean flag: tag iff ==2.
        { auto a = cs.get("lm_flag");
          tag_lm = (a && a->size_major() > 0 && a->elements<int>()[0] == 2) ? 1 : 0; }
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

      // debug Bee dump (sim only -- truth-derived): 3d points in raw coords,
      // cluster_id = trackid.
      if (m_bee_sink) {
        auto dit = lpcs.find("3d");
        if (dit != lpcs.end() && dit->second.size_major() > 0) {
          auto& d3 = dit->second;
          const auto x = d3.get("x")->elements<double>();
          const auto y = d3.get("y")->elements<double>();
          const auto z = d3.get("z")->elements<double>();
          // Tagger sets overlay clustering_global, so they use the SAME corrected
          // coords (m_tagger_coords: data x_t0cor/y_cor/z_cor, sim x_sce/y_sce/
          // z_sce).  Fall back to the raw x,y,z when unset or absent.  Copied into
          // owned vectors (elements<>() may return a temporary).
          std::vector<double> txv, tyv, tzv;
          if (m_tagger_coords.size() == 3) {
            auto ax = d3.get(m_tagger_coords[0]);
            auto ay = d3.get(m_tagger_coords[1]);
            auto az = d3.get(m_tagger_coords[2]);
            if (ax && ay && az &&
                ax->size_major() == x.size() && ay->size_major() == x.size() &&
                az->size_major() == x.size()) {
              auto sx = ax->elements<double>(); txv.assign(sx.begin(), sx.end());
              auto sy = ay->elements<double>(); tyv.assign(sy.begin(), sy.end());
              auto sz = az->elements<double>(); tzv.assign(sz.begin(), sz.end());
            }
          }
          const bool have_tc = !txv.empty();
          const double q = scalar.get("charge")->elements<double>()[0];
          const double qpp = x.size() ? std::max(q / x.size(), 1.0) : 1.0;
          const int cid = bee_cid(tid);
          for (size_t i = 0; i < x.size(); ++i) {
            const Point p(x[i], y[i], z[i]);
            // truth-derived sets: sim only (true depo positions -> raw coords)
            if (is_sim) {
              if (tid >= 0) { bpts.append(p, qpp, cid, cid); }
              else { bpts_unlab.append(p, qpp, reco_clid, reco_clid); }
            }
            // tagger verdict sets: sim AND data; only beam-window main candidates
            // are dumped, cluster_id = 0 (not tagged) / 1 (tagged), real_cluster_id
            // 0 (Bee colors by cluster_id).  Coords match clustering_global.
            if (tagger_candidate) {
              const Point pt = have_tc ? Point(txv[i], tyv[i], tzv[i]) : p;
              bpts_stm.append(pt, qpp, tag_stm, 0);
              bpts_tgm.append(pt, qpp, tag_tgm, 0);
              bpts_fc.append(pt, qpp, tag_fc, 0);
              bpts_lm.append(pt, qpp, tag_lm, 0);
            }
          }
        }
      }
    }
  }

  // ===== nugraph heterogeneous-graph record for this event =====
  // Built from the labeled grouping (trackid already in each blob scalar).
  // Truth is EXACT (SED->blob), unlike the reference point-distance matcher.
  if (m_hdf5_output) {
    namespace Fac = WireCell::Clus::Facade;
    auto* grouping = root->value.facade<Fac::Grouping>();
    auto sc_i = [](Dataset& s, const char* k) -> long long {
      auto a = s.get(k); return a ? (long long)a->elements<int>()[0] : 0; };
    auto sc_d = [](Dataset& s, const char* k) -> double {
      auto a = s.get(k); return a ? a->elements<double>()[0] : 0.0; };

    // main (nu_idx 0) vertex in mm, for the vtx features
    const bool has_nu = m_evtmd["n_nu"].asInt() > 0;
    double vtx[3] = {0, 0, 0};
    if (has_nu) {
      vtx[0] = m_evtmd["nu_vtx_x"][0u].asDouble() * 10.0; // cm -> mm
      vtx[1] = m_evtmd["nu_vtx_y"][0u].asDouble() * 10.0;
      vtx[2] = m_evtmd["nu_vtx_z"][0u].asDouble() * 10.0;
    }

    // ---- sp (3D = blob) nodes ----
    struct BInfo {
      int apa, face;
      int wmin[3], wmax[3], smin, smax;
      long long tid;
      int sem;
      double vd, vdx, vdy, vdz;
    };
    std::vector<float> sp_pos, sp_feat, sp_rawvtx;
    std::vector<long long> sp_sem, sp_inst;
    std::vector<BInfo> binfo;
    std::unordered_map<const Fac::Blob*, int> blob2idx;
    std::vector<Fac::Cluster*> clusters;
    int gidx = 0;
    for (auto* cluster : grouping->children()) {
      clusters.push_back(cluster);
      long long reco_clid = -1;
      {
        auto& cpc = cluster->value().local_pcs();
        auto it = cpc.find("cluster_scalar");
        if (it != cpc.end()) { auto a = it->second.get("ident"); if (a) reco_clid = a->elements<int>()[0]; }
      }
      for (auto* blob : cluster->children()) {
        auto& lpcs = blob->value().local_pcs();
        auto sit = lpcs.find("scalar");
        if (sit == lpcs.end()) { continue; }
        Dataset& s = sit->second;
        const long long tid = sc_i(s, "trackid");
        const WirePlaneId wpid((int)sc_i(s, "wpid"));
        const double cx = sc_d(s, "center_x"), cy = sc_d(s, "center_y"), cz = sc_d(s, "center_z");
        const double q = sc_d(s, "charge");
        int sem = tid < 0 ? -1 : (m_nu_trackids.count((int)tid) ? 0 : 1);
        double vd = -1, vdx = 0, vdy = 0, vdz = 0;
        if (has_nu && sem >= 0) {
          vdx = cx / units::mm - vtx[0];
          vdy = cy / units::mm - vtx[1];
          vdz = cz / units::mm - vtx[2];
          vd = std::sqrt(vdx * vdx + vdy * vdy + vdz * vdz);
        }
        sp_pos.push_back((float)(cx / units::mm));
        sp_pos.push_back((float)(cy / units::mm));
        sp_pos.push_back((float)(cz / units::mm));
        sp_feat.push_back((float)q);
        sp_feat.push_back((float)reco_clid);
        sp_feat.push_back((float)vd);
        sp_feat.push_back((float)vdx);
        sp_feat.push_back((float)vdy);
        sp_feat.push_back((float)vdz);
        sp_sem.push_back(sem);
        sp_inst.push_back(tid >= 0 ? tid : -1);
        sp_rawvtx.push_back((float)vd);
        BInfo bi;
        bi.apa = wpid.apa(); bi.face = wpid.face();
        const char* pn[3] = {"u", "v", "w"};
        for (int ip = 0; ip < 3; ++ip) {
          bi.wmin[ip] = (int)sc_i(s, (std::string(pn[ip]) + "_wire_index_min").c_str());
          bi.wmax[ip] = (int)sc_i(s, (std::string(pn[ip]) + "_wire_index_max").c_str());
        }
        bi.smin = (int)sc_i(s, "slice_index_min");
        bi.smax = (int)sc_i(s, "slice_index_max");
        bi.tid = tid; bi.sem = sem; bi.vd = vd; bi.vdx = vdx; bi.vdy = vdy; bi.vdz = vdz;
        binfo.push_back(bi);
        blob2idx[blob] = gidx++;
      }
    }
    const int Nsp = gidx;

    // ---- sp<->sp (blob-blob) edges ----
    // Preferred: the WCT "ctpc" graph flavor (Facade find_graph with detector
    // volumes + PC transforms).  BUT this needs the clustering-time internal
    // maps (map_mcell_*, graph cache) which as_pctree() does NOT reconstruct,
    // so on the labeler's restored tree it throws (map::at); "basic" likewise
    // yields no edges.  We therefore fall back to an intra-cluster blob-center
    // kNN message-passing graph.
    std::set<std::pair<int, int>> bbset;
    bool ctpc_ok = m_dv && m_pcts;
    auto knn_cluster = [&](Fac::Cluster* cluster) {
      std::vector<int> gi;
      for (auto* blob : cluster->children()) {
        auto it = blob2idx.find(blob);
        if (it != blob2idx.end()) { gi.push_back(it->second); }
      }
      const int M = (int)gi.size();
      for (int a = 0; a < M; ++a) {
        const int ia = gi[a];
        std::vector<std::pair<double, int>> dd;
        dd.reserve(M);
        for (int b = 0; b < M; ++b) {
          if (b == a) { continue; }
          const int ib = gi[b];
          const double dx = sp_pos[3 * ia] - sp_pos[3 * ib];
          const double dy = sp_pos[3 * ia + 1] - sp_pos[3 * ib + 1];
          const double dz = sp_pos[3 * ia + 2] - sp_pos[3 * ib + 2];
          dd.push_back({dx * dx + dy * dy + dz * dz, ib});
        }
        const int kk = std::min(m_plane_knn, (int)dd.size());
        std::partial_sort(dd.begin(), dd.begin() + kk, dd.end());
        for (int k = 0; k < kk; ++k) {
          int u = ia, v = dd[k].second;
          if (u > v) { std::swap(u, v); }
          bbset.insert({u, v});
        }
      }
    };
    const bool tried_ctpc = ctpc_ok;
    if (ctpc_ok) {
      try {
        for (auto* cluster : clusters) {
          if (cluster->nchildren() < 2 || cluster->npoints() < 2) { continue; }
          const auto& g = cluster->find_graph("ctpc", m_dv, m_pcts);
          auto ep = boost::edges(g);
          for (auto it = ep.first; it != ep.second; ++it) {
            const size_t p1 = boost::source(*it, g), p2 = boost::target(*it, g);
            auto* b1 = cluster->blob_with_point(p1);
            auto* b2 = cluster->blob_with_point(p2);
            if (!b1 || !b2 || b1 == b2) { continue; }
            auto i1 = blob2idx.find(b1), i2 = blob2idx.find(b2);
            if (i1 == blob2idx.end() || i2 == blob2idx.end()) { continue; }
            int a = i1->second, b = i2->second;
            if (a > b) { std::swap(a, b); }
            bbset.insert({a, b});
          }
        }
      }
      catch (const std::exception& e) {
        log->warn("nugraph: 'ctpc' graph unavailable on the deserialized tree "
                  "({}); using blob-center kNN for sp-sp edges", e.what());
        ctpc_ok = false;
        bbset.clear();
      }
    }
    if (!ctpc_ok) { // dv/pcts not configured, or ctpc failed above
      for (auto* cluster : clusters) {
        if (cluster->nchildren() < 2) { continue; }
        knn_cluster(cluster);
      }
    }
    log->debug("nugraph: {} sp-sp edges ({})", bbset.size(),
               (tried_ctpc && ctpc_ok) ? "ctpc flavor" : "knn fallback");

    // ---- 2D (u/v/y) nodes from the grouping ctpc_a*f*p{U,V,W} PCs ----
    struct Node2D {
      double x, pitch;                 // pos (WCT length)
      double tot_charge, mean_cerr;
      int nhits, apa, face, plane;     // plane 0/1/2 = u/v/y
      int wmin, wmax, smin, smax;
      double pmin, pmax;               // pitch extent
    };
    std::vector<Node2D> nodes2d[3];    // per plane letter u,v,y
    // plane letter -> nugraph plane index: U->0(u), V->1(v), W->2(y)
    auto plane_of = [](char c) -> int { return c == 'U' ? 0 : c == 'V' ? 1 : c == 'W' ? 2 : -1; };
    for (auto& kv : root->value.local_pcs()) {
      const std::string& nm = kv.first;
      if (nm.rfind("ctpc_", 0) != 0) { continue; }
      const int pl = plane_of(nm.back());
      if (pl < 0) { continue; }
      // parse a{A}f{F}
      int apa = 0, face = 0;
      { auto pa = nm.find('a'); auto pf = nm.find('f');
        if (pa != std::string::npos) { apa = std::atoi(nm.c_str() + pa + 1); }
        if (pf != std::string::npos) { face = std::atoi(nm.c_str() + pf + 1); } }
      Dataset& d = kv.second;
      auto ax = d.get("x"); auto ay = d.get("y");
      auto aq = d.get("charge"); auto ae = d.get("charge_err");
      auto aw = d.get("wind"); auto as = d.get("slice_index");
      if (!ax || !ay || !aq || !aw || !as) { continue; }
      const auto vx = ax->elements<double>();
      const auto vy = ay->elements<double>();
      const auto vq = aq->elements<double>();
      std::vector<double> ve;
      if (ae) { auto es = ae->elements<double>(); ve.assign(es.begin(), es.end()); }
      else { ve.assign(vx.size(), 0.0); }
      const auto vw = aw->elements<int>();
      const auto vs = as->elements<int>();
      const size_t nh = vx.size();
      if (!nh) { continue; }
      // sort hit indices by x
      std::vector<size_t> ord(nh);
      for (size_t i = 0; i < nh; ++i) { ord[i] = i; }
      std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return vx[a] < vx[b]; });
      // greedy x-groups (running-mean reference, tol m_ctpc_x_tol), then split
      // each group on pitch gaps > m_ctpc_pitch_gap into contiguous runs.
      size_t i0 = 0;
      while (i0 < nh) {
        double xsum = 0; size_t i1 = i0;
        while (i1 < nh) {
          const double xi = vx[ord[i1]];
          if (i1 > i0 && std::abs(xi - xsum / (i1 - i0)) > m_ctpc_x_tol) { break; }
          xsum += xi; ++i1;
        }
        // hits [i0,i1) form an x-group; sort by pitch
        std::vector<size_t> grp(ord.begin() + i0, ord.begin() + i1);
        std::sort(grp.begin(), grp.end(), [&](size_t a, size_t b) { return vy[a] < vy[b]; });
        size_t j0 = 0;
        while (j0 < grp.size()) {
          size_t j1 = j0 + 1;
          while (j1 < grp.size() && (vy[grp[j1]] - vy[grp[j1 - 1]]) <= m_ctpc_pitch_gap) { ++j1; }
          // run [j0,j1) -> one 2D node
          Node2D n;
          n.apa = apa; n.face = face; n.plane = pl;
          n.nhits = (int)(j1 - j0);
          double qsum = 0, xw = 0, pw = 0, esum = 0;
          n.wmin = INT_MAX; n.wmax = INT_MIN; n.smin = INT_MAX; n.smax = INT_MIN;
          n.pmin = 1e30; n.pmax = -1e30;
          for (size_t j = j0; j < j1; ++j) {
            const size_t h = grp[j];
            const double qq = std::max(vq[h], 0.0);
            qsum += vq[h]; esum += ve[h];
            xw += vx[h] * (qq + 1e-9); pw += vy[h] * (qq + 1e-9);
            n.wmin = std::min(n.wmin, vw[h]); n.wmax = std::max(n.wmax, vw[h]);
            n.smin = std::min(n.smin, vs[h]); n.smax = std::max(n.smax, vs[h]);
            n.pmin = std::min(n.pmin, vy[h]); n.pmax = std::max(n.pmax, vy[h]);
          }
          double wsum = 0;
          for (size_t j = j0; j < j1; ++j) { wsum += std::max(vq[grp[j]], 0.0) + 1e-9; }
          n.x = xw / wsum; n.pitch = pw / wsum;
          n.tot_charge = qsum; n.mean_cerr = n.nhits ? esum / n.nhits : 0.0;
          nodes2d[pl].push_back(n);
          j0 = j1;
        }
        i0 = i1;
      }
    }

    // ---- {p}_nexus_sp edges (2D hit -> blob) from the TRUE wire/slice box
    // overlap; {p}/y_semantic,y_instance from linked blobs; vtx features. ----
    const char* plane_names[3] = {"u", "v", "y"};
    std::vector<int> nx_src[3], nx_dst[3];        // per plane: hit idx, sp idx
    std::vector<long long> n2_sem[3], n2_inst[3];
    std::vector<float> n2_vtx[3];                 // 4 vtx feats per node, flat
    for (int pl = 0; pl < 3; ++pl) {
      const int wp = pl; // plane letter index into blob wmin/wmax (u,v,w)
      for (size_t ni = 0; ni < nodes2d[pl].size(); ++ni) {
        const Node2D& n = nodes2d[pl][ni];
        std::unordered_map<long long, int> inst_votes;
        int any_nu = 0, any_cos = 0;
        double svd = 0, svx = 0, svy = 0, svz = 0; int nvtx = 0;
        for (int bi = 0; bi < Nsp; ++bi) {
          const BInfo& b = binfo[bi];
          if (b.apa != n.apa || b.face != n.face) { continue; }
          const bool wover = b.wmin[wp] < n.wmax + 1 && n.wmin < b.wmax[wp];
          const bool sover = b.smin < n.smax + 1 && n.smin < b.smax;
          if (!(wover && sover)) { continue; }
          nx_src[pl].push_back((int)ni);
          nx_dst[pl].push_back(bi);
          if (b.sem == 0) { ++any_nu; } else if (b.sem == 1) { ++any_cos; }
          if (b.tid >= 0) { inst_votes[b.tid]++; }
          svd += b.vd; svx += b.vdx; svy += b.vdy; svz += b.vdz; ++nvtx;
        }
        n2_sem[pl].push_back(any_nu ? 0 : (any_cos ? 1 : -1));
        long long best = -1; int bestv = 0;
        for (auto& kv : inst_votes) { if (kv.second > bestv) { bestv = kv.second; best = kv.first; } }
        n2_inst[pl].push_back(best);
        if (nvtx) { n2_vtx[pl].push_back((float)(svd / nvtx)); n2_vtx[pl].push_back((float)(svx / nvtx));
                    n2_vtx[pl].push_back((float)(svy / nvtx)); n2_vtx[pl].push_back((float)(svz / nvtx)); }
        else { n2_vtx[pl].push_back(-1); n2_vtx[pl].push_back(0); n2_vtx[pl].push_back(0); n2_vtx[pl].push_back(0); }
      }
    }

    // NOTE: intra-plane {p}_plane_{p} edges are intentionally NOT produced
    // here -- they are added downstream in post-processing (e.g. Delaunay).

    // ---- assemble the compound record members ----
    // Skip empty events (HDF5 array members need every dim >= 1).
    if (Nsp > 0) {
      EventGraph ev;
      char sn[128];
      std::snprintf(sn, sizeof(sn), "%d_%d_%d__rec-lab-apa0-1", m_run, m_sub, m_evt);
      ev.sample_name = sn;
      auto addF = [&](const std::string& name, std::vector<float>&& v,
                      std::vector<unsigned long long> dims) {
        H5Member m; m.name = name; m.is_float = true; m.dims = dims; m.f = std::move(v);
        ev.members.push_back(std::move(m)); };
      auto addI = [&](const std::string& name, std::vector<long long>&& v,
                      std::vector<unsigned long long> dims) {
        H5Member m; m.name = name; m.is_float = false; m.dims = dims; m.i = std::move(v);
        ev.members.push_back(std::move(m)); };
      // dummy [[0],[0]] when an edge set is empty (avoids zero-dim members).
      auto add_edges = [&](const std::string& name, std::vector<int>& s, std::vector<int>& d) {
        std::vector<long long> flat;
        if (s.empty()) { flat = {0, 0}; addI(name, std::move(flat), {2, 1}); return; }
        flat.reserve(2 * s.size());
        for (int v : s) { flat.push_back(v); }
        for (int v : d) { flat.push_back(v); }
        addI(name, std::move(flat), {2, (unsigned long long)s.size()});
      };

      // metadata / evt
      addI("metadata/run", {m_run}, {});
      addI("metadata/subrun", {m_sub}, {});
      addI("metadata/event", {m_evt}, {});
      addI("evt/num_nodes", {1}, {});
      addI("evt/y", {has_nu ? 1LL : 0LL}, {1});

      // sp nodes
      addF("sp/pos", std::move(sp_pos), {(unsigned long long)Nsp, 3});
      addF("sp/features", std::move(sp_feat), {(unsigned long long)Nsp, 6});
      addF("sp/raw_vtx_dist", std::move(sp_rawvtx), {(unsigned long long)Nsp});
      addI("sp/y_semantic", std::move(sp_sem), {(unsigned long long)Nsp});
      addI("sp/y_instance", std::move(sp_inst), {(unsigned long long)Nsp});

      // sp supervision edges from the blob-blob graph, labeled by trackid.
      {
        std::vector<int> ss, sd; std::vector<long long> ey, elab;
        for (auto& e : bbset) {
          ss.push_back(e.first); sd.push_back(e.second);
          const long long t1 = binfo[e.first].tid, t2 = binfo[e.second].tid;
          const bool labelable = t1 >= 0 && t2 >= 0;
          elab.push_back(labelable ? 1 : 0);
          ey.push_back(labelable && t1 == t2 ? 1 : 0);
        }
        if (ss.empty()) { ss = {0}; sd = {0}; ey = {0}; elab = {0}; }
        std::vector<int> se = ss; // for add_edges signature
        add_edges("sp/edge_label_index", se, sd);
        addI("sp/edge_y", std::move(ey), {(unsigned long long)ey.size()});
        addI("sp/edge_labelable", std::move(elab), {(unsigned long long)elab.size()});
      }
      // sp<->sp message-passing edges (same blob-blob graph, undirected-unique)
      {
        std::vector<int> ms, md;
        for (auto& e : bbset) { ms.push_back(e.first); md.push_back(e.second); }
        add_edges("sp_nexus_sp/edge_index", ms, md);
      }

      // 2D nodes + edges per plane
      for (int pl = 0; pl < 3; ++pl) {
        const auto& nn = nodes2d[pl];
        const int M = (int)nn.size();
        std::vector<float> pos, x15; std::vector<long long> id;
        for (int i = 0; i < M; ++i) {
          pos.push_back((float)(nn[i].x / units::mm));
          pos.push_back((float)(nn[i].pitch / units::mm));
          x15.push_back((float)nn[i].tot_charge);
          x15.push_back((float)nn[i].mean_cerr);
          x15.push_back((float)nn[i].nhits);
          x15.push_back((float)(nn[i].pmin / units::mm));
          x15.push_back((float)(nn[i].pmax / units::mm));
          x15.push_back(n2_vtx[pl][4 * i + 0]);
          x15.push_back(n2_vtx[pl][4 * i + 1]);
          x15.push_back(n2_vtx[pl][4 * i + 2]);
          x15.push_back(n2_vtx[pl][4 * i + 3]);
          for (int z = 0; z < 6; ++z) { x15.push_back(0.f); } // sidecar zeros
          id.push_back(i);
        }
        const std::string p = plane_names[pl];
        if (M > 0) {
          addF(p + "/pos", std::move(pos), {(unsigned long long)M, 2});
          addF(p + "/x", std::move(x15), {(unsigned long long)M, 15});
          addI(p + "/id", std::move(id), {(unsigned long long)M});
          addI(p + "/y_semantic", std::move(n2_sem[pl]), {(unsigned long long)M});
          addI(p + "/y_instance", std::move(n2_inst[pl]), {(unsigned long long)M});
        }
        else { // rare empty plane: one dummy node so members stay non-empty
          addF(p + "/pos", {0, 0}, {1, 2});
          addF(p + "/x", std::vector<float>(15, 0.f), {1, 15});
          addI(p + "/id", {0}, {1});
          addI(p + "/y_semantic", {-1}, {1});
          addI(p + "/y_instance", {-1}, {1});
        }
        add_edges(p + "_nexus_sp/edge_index", nx_src[pl], nx_dst[pl]);
      }

      m_events.push_back(std::move(ev));
      log->debug("nugraph: {} sp, {}/{}/{} 2D (u/v/y) nodes, {} sp-sp edges",
                 Nsp, (int)nodes2d[0].size(), (int)nodes2d[1].size(),
                 (int)nodes2d[2].size(), (int)bbset.size());
    }
    else {
      log->warn("nugraph: 0 blobs for rse=({},{},{}), skipping HDF5 record",
                m_run, m_sub, m_evt);
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

  // truth_per_track tensor: [ntracks x ncols] doubles (sim only).
  if (is_sim) {
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
    // truth-derived sets: sim only
    if (is_sim) {
      m_bee_sink->write(bpts, m_bee_index, m_run, m_sub, m_evt);
      m_bee_sink->write(bpts_unlab, m_bee_index, m_run, m_sub, m_evt);
      if (!bpts_depo.empty()) {
        m_bee_sink->write(bpts_depo, m_bee_index, m_run, m_sub, m_evt);
      }
      if (!bpts_sr.empty()) {
        m_bee_sink->write(bpts_sr, m_bee_index, m_run, m_sub, m_evt);
      }
      if (!bpts_ssr.empty()) {
        m_bee_sink->write(bpts_ssr, m_bee_index, m_run, m_sub, m_evt);
      }
      if (!m_pf_particles.empty()) {
        Bee::ParticleTree pf(m_bee_pf_name);
        pf.set_particles(m_pf_particles);
        m_bee_sink->write(pf, m_bee_index, m_run, m_sub, m_evt);
      }
    }
    // tagger verdict sets: sim AND data
    m_bee_sink->write(bpts_stm, m_bee_index, m_run, m_sub, m_evt);
    m_bee_sink->write(bpts_tgm, m_bee_index, m_run, m_sub, m_evt);
    m_bee_sink->write(bpts_fc, m_bee_index, m_run, m_sub, m_evt);
    m_bee_sink->write(bpts_lm, m_bee_index, m_run, m_sub, m_evt);
    ++m_bee_index;
  }

  log->debug("call={} ident={} rse=({},{},{}): projected {}/{} depos, "
             "labeled {}/{} blobs, {} truth tracks",
             m_count, ident, m_run, m_sub, m_evt,
             nproj, m_depos.size(), nlabeled, nblobs, m_tracks.size());
  ++m_count;
  return true;
}
