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
// (positions cm, time ns, momentum/energy GeV).
static const std::vector<std::string> track_columns = {
  "trackid",  "pdg",      "mother_trackid", "mother_pdg", "status",
  "start_x",  "start_y",  "start_z",        "start_t",    "start_px",
  "start_py", "start_pz", "start_E",        "end_x",      "end_y",
  "end_z",    "end_t",    "end_px",         "end_py",     "end_pz",
  "end_E",    "nu_idx"};

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
  , m_pf_ke_min(10 * units::MeV)
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
  m_sce_correction = get(cfg, "sce_correction", m_sce_correction);
  m_truth_tracks_nu_only = get(cfg, "truth_tracks_nu_only", m_truth_tracks_nu_only);
  m_pf_nu_only = get(cfg, "pf_nu_only", m_pf_nu_only);
  m_pf_ke_min = get(cfg, "pf_ke_min", m_pf_ke_min);

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

  // --- neutrino truth (cf. Truth2h5) ---
  m_evtmd["nu_flavor"] = "none";
  art::Handle<std::vector<simb::MCTruth>> mctruth_handle;
  if (event.getByLabel(art::InputTag{m_mctruth_label}, mctruth_handle) &&
      mctruth_handle.isValid() && !mctruth_handle->empty() &&
      mctruth_handle->front().NeutrinoSet()) {
    const auto& nu = mctruth_handle->front().GetNeutrino();
    const auto& nu_particle = nu.Nu();
    const auto& position = nu_particle.Position(0);
    const auto& momentum = nu_particle.Momentum(0);
    const int pdg = nu_particle.PdgCode();
    const int ccnc = nu.CCNC(); // 0 = CC, 1 = NC
    m_evtmd["nu_pdg"] = pdg;
    m_evtmd["nu_ccnc"] = ccnc;
    m_evtmd["nu_int_type"] = nu.InteractionType();
    m_evtmd["nu_energy"] = momentum.E();   // GeV
    m_evtmd["nu_vtx_x"] = position.X();    // cm
    m_evtmd["nu_vtx_y"] = position.Y();
    m_evtmd["nu_vtx_z"] = position.Z();
    if (ccnc == 1) { m_evtmd["nu_flavor"] = "nc"; }
    else if (std::abs(pdg) == 12) { m_evtmd["nu_flavor"] = "nue"; }
    else if (std::abs(pdg) == 14) { m_evtmd["nu_flavor"] = "numu"; }
    else if (std::abs(pdg) == 16) { m_evtmd["nu_flavor"] = "nutau"; }
  }
  else {
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
      m_tracks.push_back({(double)p.TrackId(),
                          (double)p.PdgCode(),
                          (double)p.Mother(),
                          (double)(itmom == tid2pdg.end() ? 0 : itmom->second),
                          (double)p.StatusCode(),
                          s4.X(), s4.Y(), s4.Z(), s4.T(),
                          sm.Px(), sm.Py(), sm.Pz(), sm.E(),
                          e4.X(), e4.Y(), e4.Z(), e4.T(),
                          em.Px(), em.Py(), em.Pz(), em.E(),
                          (double)(i < nu_index.size() ? nu_index[i] : -1)});
    }
  }
  else {
    log->warn("failed to fetch MCParticles with label '{}'", m_mcparticle_label);
  }

  // --- Bee "mc" particle-flow tree: MCParticles with KE > pf_ke_min,
  // nested under the nearest KEPT ancestor by Mother() tracing ---
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
    std::vector<int> pf_roots;
    for (const auto& p : *mcps) {
      if (!kept(p)) { continue; }
      int anc = p.Mother();
      while (anc > 0) {
        auto it = by_tid.find(anc);
        if (it == by_tid.end()) { anc = 0; break; }
        if (kept(*it->second)) { break; }
        anc = it->second->Mother();
      }
      if (anc > 0 && by_tid.count(anc)) { pf_children[anc].push_back(p.TrackId()); }
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
      d.trackid = sed.TrackID();
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

  log->debug("visit run {} sub {} evt {}: nu_flavor {}, {} tracks, {} depos",
             m_run, m_sub, m_evt,
             m_evtmd["nu_flavor"].asString(),
             m_tracks.size(),
             m_depos.size());
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
  };
  std::map<std::pair<int, int>, std::unordered_map<int, std::vector<PDepo>>> proj;
  size_t nproj = 0;
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
        for (int itick = smin - m_tick_slop; itick < smax + m_tick_slop; ++itick) {
          auto bit = pit->second.find(itick);
          if (bit == pit->second.end()) { continue; }
          for (const auto& pd : bit->second) {
            bool inside = true;
            for (int ip = 0; ip < 3; ++ip) {
              if (pd.wip[ip] < wmin[ip] - m_wire_slop || pd.wip[ip] >= wmax[ip] + m_wire_slop) {
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
          for (size_t i = 0; i < x.size(); ++i) {
            bpts.append(Point(x[i], y[i], z[i]), qpp, tid, tid);
            if (tid < 0) {
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
