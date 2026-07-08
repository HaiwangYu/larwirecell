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
#include "canvas/Utilities/InputTag.h"
#include "lardataobj/Simulation/SimEnergyDeposit.h"
#include "nusimdata/SimulationBase/MCParticle.h"
#include "nusimdata/SimulationBase/MCTruth.h"

#include <algorithm>
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
  "end_E"};

AIML::TensorSetLabeler::TensorSetLabeler()
  : Aux::Logger("TensorSetLabeler", "aiml")
  , m_drift_speed(1.563 * units::mm / units::us)
  , m_time_offset(-205 * units::us)
  , m_tick(0.5 * units::us)
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
  if (event.getByLabel(art::InputTag{m_mcparticle_label}, mcps) && mcps.isValid()) {
    tid2pdg.reserve(mcps->size());
    for (const auto& p : *mcps) {
      tid2pdg[p.TrackId()] = p.PdgCode();
    }
    m_tracks.reserve(mcps->size());
    for (const auto& p : *mcps) {
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
                          em.Px(), em.Py(), em.Pz(), em.E()});
    }
  }
  else {
    log->warn("failed to fetch MCParticles with label '{}'", m_mcparticle_label);
  }

  // --- energy deposits ---
  art::Handle<std::vector<sim::SimEnergyDeposit>> seds;
  if (event.getByLabel(art::InputTag{m_deposet_label}, seds) && seds.isValid()) {
    m_depos.reserve(seds->size());
    for (const auto& sed : *seds) {
      Depo d;
      d.x = sed.MidPointX() * units::cm;
      d.y = sed.MidPointY() * units::cm;
      d.z = sed.MidPointZ() * units::cm;
      d.t = sed.Time() * units::ns;
      d.trackid = sed.TrackID();
      d.weight = sed.NumElectrons() > 0 ? (double)sed.NumElectrons() : sed.Energy();
      m_depos.push_back(d);
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
  size_t nblobs = 0, nlabeled = 0;
  for (auto* cnode : root->children()) {
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
    ++m_bee_index;
  }

  log->debug("call={} ident={} rse=({},{},{}): projected {}/{} depos, "
             "labeled {}/{} blobs, {} truth tracks",
             m_count, ident, m_run, m_sub, m_evt,
             nproj, m_depos.size(), nlabeled, nblobs, m_tracks.size());
  ++m_count;
  return true;
}
