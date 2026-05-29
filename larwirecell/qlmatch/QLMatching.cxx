#include "QLMatching.h"
#include "util.h"

#include "Opflash.h"
#include "TimingTPCBundle.h"
#include "WireCellAux/TensorDMcommon.h"
#include "WireCellAux/TensorDMdataset.h"
#include "WireCellAux/TensorDMpointtree.h"
#include "WireCellClus/Facade.h"
// Flag names mirrored from WireCellClus/ClusteringFuncs.h ("main_cluster",
// "beam_flash").  Inlined here rather than included to avoid pulling
// Bee.h → miniz.h, which the larsoft mrb include path can't resolve.
#include "WireCellUtil/Exceptions.h"
#include "WireCellUtil/ExecMon.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Persist.h"
#include "WireCellUtil/Ress.h"
#include "WireCellUtil/String.h"
#include "WireCellUtil/Units.h"

#include <cmath>
#include <limits>

#include "art/Utilities/make_tool.h"
#include "cetlib/filepath_maker.h"
#include "fhiclcpp/ParameterSet.h"
#include "larcoreobj/SimpleTypesAndConstants/geo_vectors.h"
#include "larsim/PhotonPropagation/OpticalPathTools/OpticalPath.h"
#include "larsim/PhotonPropagation/SemiAnalyticalModel.h"

// Factory name `wclsQLMatching` disambiguates the larwirecell-side QLMatching
// from the wire-cell-toolkit-side `Match::QLMatching` (libWireCellMatch.so),
// which both classes used to register under the same `QLMatching` string.
// SBND wcls fcl/jsonnet should reference `wclsQLMatching` for this plugin.
WIRECELL_FACTORY(wclsQLMatching,
                 WireCell::QLMatch::QLMatching,
                 WireCell::INamed,
                 WireCell::ITensorSetFanin,
                 WireCell::IConfigurable)

// Forward declaration of the wire-cell-toolkit helper that pads every cluster
// in a grouping with the union of flag_* keys.  Linked from libWireCellClus.so
// (defined in clus/src/ClusteringFuncs.cxx).  Declared here rather than
// `#include "WireCellClus/ClusteringFuncs.h"` to avoid pulling Bee.h/miniz.h
// through the larsoft mrb include path.
namespace WireCell::Clus::Facade {
    class Grouping;
    void normalize_cluster_flags(Grouping& grouping,
                                 WireCell::Log::logptr_t log,
                                 const std::string& grouping_name,
                                 int ident);
}

using namespace WireCell;
using namespace WireCell::Clus::Facade;

WireCell::QLMatch::QLMatching::QLMatching() : Aux::Logger("QLMatching", "matching") {}

WireCell::QLMatch::QLMatching::~QLMatching() {}

std::vector<std::string> WireCell::QLMatch::QLMatching::input_types()
{
  const std::string tname = std::string(typeid(input_type).name());
  std::vector<std::string> ret(m_multiplicity, tname);
  return ret;
}

void WireCell::QLMatch::QLMatching::configure(const WireCell::Configuration& cfg)
{
  m_anode = Factory::find_tn<IAnodePlane>(cfg["anode"].asString());
  m_dv = Factory::find_tn<IDetectorVolumes>(cfg["detector_volumes"].asString());

  m_inpath = get(cfg, "inpath", m_inpath);
  m_outpath = get(cfg, "outpath", m_outpath);
  m_bee_dir = get(cfg, "bee_dir", m_bee_dir);
  m_cluster_t0 = get(cfg, "cluster_t0", m_cluster_t0);

  m_pmts = get(cfg, "pmts", m_pmts);
  m_data = get(cfg, "data", m_data);
  m_beamonly = get(cfg, "beamonly", m_beamonly);

  if (cfg.isMember("ch_mask") && cfg["ch_mask"].isArray()) {
    m_ch_mask.clear();
    for (const auto& jch : cfg["ch_mask"]) {
      m_ch_mask.push_back(jch.asInt());
    }
  }

  m_flash_minPE = get(cfg, "flash_minPE", m_flash_minPE);

  m_max_beam_flash_time = get(cfg, "max_beam_flash_time", m_max_beam_flash_time);

  if (m_beamonly) {
    m_flash_mintime = m_beam_mintime;
    m_flash_maxtime = m_beam_maxtime;
  }

  m_QtoL = get(cfg, "QtoL", m_QtoL);
  m_strength_cutoff = get(cfg, "strength_cutoff", m_strength_cutoff);

  if (cfg["VUVEfficiency"].isArray()) {
    m_VUVEfficiency.clear();
    for (auto vuv_eff : cfg["VUVEfficiency"])
      m_VUVEfficiency.push_back(vuv_eff.asDouble());
  }

  if (cfg["VISEfficiency"].isArray()) {
    m_VISEfficiency.clear();
    for (auto vis_eff : cfg["VISEfficiency"])
      m_VISEfficiency.push_back(vis_eff.asDouble());
  }
}

WireCell::Configuration WireCell::QLMatch::QLMatching::default_configuration() const
{
  Configuration cfg;

  cfg["inpath"] = m_inpath;
  cfg["outpath"] = m_outpath;
  cfg["bee_dir"] = m_bee_dir;

  cfg["pmts"] = m_pmts;
  cfg["data"] = m_data;
  cfg["beamonly"] = m_beamonly;
  cfg["ch_mask"] = Json::arrayValue;

  cfg["flash_minPE"] = m_flash_minPE;
  cfg["flash_mintime"] = m_flash_mintime;
  cfg["flash_maxtime"] = m_flash_maxtime;
  cfg["beam_mintime"] = m_beam_mintime;
  cfg["beam_maxtime"] = m_beam_maxtime;
  cfg["max_beam_flash_time"] = m_max_beam_flash_time;
  cfg["QtoL"] = m_QtoL;
  cfg["strength_cutoff"] = m_strength_cutoff;

  return cfg;
}

bool WireCell::QLMatch::QLMatching::operator()(const input_vector& invec, output_pointer& out)
{
  out = nullptr;
  using WireCell::Clus::Facade::float_t;
  // check input size
  if (invec.size() != m_multiplicity) {
    raise<ValueError>("unexpected multiplicity got %d want %d", invec.size(), m_multiplicity);
    return true;
  }

  // boilerplate for EOS handling
  size_t neos = 0;
  for (const auto& in : invec) {
    if (!in) { ++neos; }
  }
  if (neos == invec.size()) {
    // all inputs are EOS, good.
    log->debug("EOS at call {}", m_count++);
    return true;
  }
  if (neos) {
    log->debug("port0 {} port1 {}", (invec[0] ? "valid" : "EOS"), (invec[1] ? "valid" : "EOS"));
    raise<ValueError>("missing %d input tensors ", neos);
  }

  ExecMon em("starting QLMatching");

  // ! start things to move to config block

  std::vector<uint> opdet_mask;
  if (m_pmts)
    opdet_mask = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};

  for (size_t idet = 0; idet < m_ch_mask.size(); idet++) {
    opdet_mask[m_ch_mask[idet]] = 0;
  }

  // ! end things to move to config block
  std::string filename = "semimodel_sbnd.fcl";
  std::string pathvar("FHICL_FILE_PATH");
  const std::string vuv_key = "VUVHits";
  const std::string vis_key = "VISHits";
  cet::filepath_lookup maker(pathvar);
  fhicl::ParameterSet pset = fhicl::ParameterSet::make(filename, maker);
  auto const vuv_pset = pset.get<fhicl::ParameterSet>(vuv_key);
  auto const vis_pset = pset.get<fhicl::ParameterSet>(vis_key);
  auto opticalPath = std::shared_ptr<phot::OpticalPath>(
    art::make_tool<phot::OpticalPath>(pset.get<fhicl::ParameterSet>("OpticalPathTool")));
  auto semi_model = std::make_unique<phot::SemiAnalyticalModel>(
    vuv_pset, vis_pset, opticalPath, true, false, false);

  const auto& charge_ts = invec[0];
  const int charge_ident = charge_ts->ident();
  std::string inpath = m_inpath;
  if (inpath.find("%") != std::string::npos) { inpath = String::format(inpath, charge_ident); }

  const auto& charge_tens = *charge_ts->tensors();
  log->debug("charge_tens.size {}", charge_tens.size());
  auto root_live = Aux::TensorDM::as_pctree(charge_tens, inpath + "/live");
  if (!root_live) {
    log->error("Failed to get point cloud tree from \"{}\"", inpath);
    return false;
  }
  log->debug("Got live pctree with {} children", root_live->nchildren());
  log->debug(em("got live pctree"));

  // check Opflahs object
  std::vector<Opflash::pointer> flashes;
  log->debug("checking Opflahs object");
  const auto& tens = invec[1]->tensors();
  if (tens->size() != 1) { raise<ValueError>("Expected 1 tensor, got %d", tens->size()); }
  const auto& ten = tens->at(0);
  if (ten->shape().size() != 2) {
    raise<ValueError>("input tensor dim %d != 2", ten->shape().size());
  }
  const int nrow = ten->shape()[0];
  const int ncol = ten->shape()[1];
  log->debug("nrow {} ncol {}", nrow, ncol);
  const int nchan = ncol - 1;
  // if (nrow < 1) { raise<ValueError>("input tensor nrow %d < 1", nrow); }
  for (int iflash = 0; iflash < nrow; ++iflash) {
    // Opflash flash(ten, iflash, 0.0, nchan);
    Opflash::pointer flash = std::make_shared<Opflash>(ten, iflash, 0.0, nchan);
    if (flash->get_time() < m_flash_mintime || flash->get_time() > m_flash_maxtime) { continue; }
    if (flash->get_total_PE() < m_flash_minPE) { continue; }
    flashes.push_back(flash);
  }

  // check TimingTPCBundle object
  auto grouping = root_live->value.facade<Grouping>();
  grouping->set_anodes({m_anode});
  grouping->set_detector_volumes(m_dv);
  std::vector<Cluster*> clusters = grouping->children();
  std::sort(clusters.begin(), clusters.end(), [](const Cluster* cluster1, const Cluster* cluster2) {
    return cluster1->get_length() > cluster2->get_length();
  });

  double total_charge_blob = 0.0;
  double total_charge_point = 0.0;
  double total_charge_blob_all = 0.0;
  for (auto cluster : clusters) {
    std::vector<Blob*> blobs = cluster->children();
    for (auto blob : blobs) {
        total_charge_blob_all += blob->charge();
    }
  }

  // add default cluster_t0 to all clusters
  std::for_each(
    clusters.begin(), clusters.end(), [this](Cluster* cluster) { cluster->set_cluster_t0(-1e12); });

  // create global maps
  std::map<Opflash*, int> global_flash_idx_map;
  std::map<Cluster*, int> global_cluster_idx_map;

  for (size_t i = 0; i < flashes.size(); ++i) {
    global_flash_idx_map[flashes[i].get()] = i;
  }

  for (size_t i = 0; i < clusters.size(); ++i) {
    global_cluster_idx_map[clusters[i]] = i;
  }

  std::vector<TimingTPCBundle::pointer> all_bundles;
  TimingTPCBundleSet pre_bundles;

  uint tpc = m_anode->ident();
  int sign_offset = (tpc == 0) ? -1 : 1;
  double lo_x_bound = (tpc == 0) ? -2000 : 0;
  double hi_x_bound = (tpc == 0) ? 0 : 2000;

  for (size_t idet = 0; idet < opdet_mask.size(); idet++) {
    if ((tpc == 0) && (idet % 2 == 1)) opdet_mask[idet] = 0;
    if ((tpc == 1) && (idet % 2 == 0)) opdet_mask[idet] = 0;
  }

  for (auto flash : flashes) {
    auto flash_time = flash->get_time();
    auto flash_x_offset =
      sign_offset * flash_time * 1.563e-3; // 1.563e-3 is SBND drift velocity in mm/ns

    // per flash mask
    std::vector<uint> flash_opdet_mask = opdet_mask;
    // ! warning: this is a temporary fix to identify simulated saturated PMTs
    for (size_t idet = 0; idet < size_t(flash->get_num_channels()); idet++) {
      auto pe_det = flash->get_PE(idet);
      if ((flash->get_total_PE() > 5000) & (pe_det == 0) & (m_data == false))
        flash_opdet_mask[idet] = 0;
    }

    log->debug("flash time {} flash PE {} flash_x_offset {}",
               int(flash_time) / 100.,
               int(flash->get_total_PE() * 100) / 100.,
               int(flash_x_offset * 100) / 100.);

    for (size_t icluster = 0; icluster < clusters.size(); ++icluster) {
      Cluster* cluster = clusters[icluster];
      // TimingTPCBundle bundle(flash.get(), cluster, flash->get_flash_id(), icluster);
      TimingTPCBundle::pointer bundle =
        std::make_shared<TimingTPCBundle>(flash.get(), cluster, flash->get_flash_id(), icluster);
      all_bundles.push_back(bundle);

      bundle->set_opdet_mask(flash_opdet_mask);

      size_t nopdets = flash->get_num_channels();
      std::vector<double> pred_flash(nopdets, 0.0);

      size_t npt = cluster->npoints();
      int npt_outside_drift = 0;
      int npt_outside_bounds = 0;

      bool drifted_outside = false;

      // log->debug("flash {} and cluster {} with {} children", flash->get_flash_id(), icluster, cluster->nchildren());
      std::vector<Blob*> blobs = cluster->children();
      for (auto blob : blobs) {
        total_charge_blob += blob->charge();
        auto q = blob->charge() / blob->npoints();
        std::vector<geo_point_t> points = blob->points("3d", {"x", "y", "z"});

        for (int i = 0; i != blob->npoints(); i++) {
          total_charge_point += q;
          auto x = points.at(i).x() + flash_x_offset;
          auto y = points.at(i).y();
          auto z = points.at(i).z();

          if (x < lo_x_bound || x > hi_x_bound) {
            npt_outside_drift++;
            continue;
          }
          if (abs(y) > 2000 || z < 0 || z > 5000) {
            npt_outside_bounds++;
            continue;
          }

          if (abs(x) && bundle->get_flag_at_x_boundary() == false)
            bundle->set_flag_close_to_PMT(true);

          if (abs(x) > 1950 && bundle->get_flag_close_to_PMT() == false)
            bundle->set_flag_close_to_PMT(true);

          if (npt_outside_drift > 0.25 * npt) {
            drifted_outside = true;
            break;
          }

          geo::Point_t xyz_cm = {x / 10, y / 10, z / 10};
          std::vector<double> direct_visibilities;
          semi_model->detectedDirectVisibilities(direct_visibilities, xyz_cm);
          std::vector<double> reflected_visibilities;
          semi_model->detectedReflectedVisibilities(reflected_visibilities, xyz_cm);

          // TODO: add opdet channel mask configurable
          for (size_t idet = 0; idet < nopdets; idet++) {
            if (flash_opdet_mask.at(idet) == 0) continue;
            auto dir_vis = direct_visibilities.at(idet);
            auto ref_vis = reflected_visibilities.at(idet);
            auto dir_eff = m_VUVEfficiency.at(idet);
            auto ref_eff = m_VISEfficiency.at(idet);
            pred_flash.at(idet) += q * m_QtoL * dir_vis * dir_eff + q * m_QtoL * ref_vis * ref_eff;
          }
        }
        if (drifted_outside) break;
      }

      // pre-selection stage
      // * if over 20% of the points ar outside the drift volume, skip
      if (drifted_outside) {
        bundle->set_potential_bad_match_flag(true);
        continue;
      }

      bundle->set_pred_flash(pred_flash);
      if (bundle->get_total_pred_light() < 10) continue;
      bundle->examine_bundle();

      if (bundle->get_ks_dis() == 1) {
        bundle->set_potential_bad_match_flag(true);
        continue;
      }
      if (bundle->get_chi2() / bundle->get_ndf() > 1e4) {
        bundle->set_potential_bad_match_flag(true);
        continue;
      }

      log->debug("initial eval: flash {} and cluster {}, meas PE {}, pred PE {}, npts {}, ks_dis "
                 "{}, chi2/ndf {}, ndf {}",
                 flash->get_flash_id(),
                 global_cluster_idx_map[cluster],
                 int(flash->get_total_PE() * 100) / 100.,
                 int(bundle->get_total_pred_light() * 100) / 100.,
                 npt,
                 int(bundle->get_ks_dis() * 1000) / 1000.,
                 int(bundle->get_chi2() / bundle->get_ndf() * 100) / 100.,
                 bundle->get_ndf());

      pre_bundles.insert(bundle);

    } // end first cluster loop

  } // end flash loop
  log->debug("n preselected bundles: {}", pre_bundles.size());

  // * construct maps
  FlashBundlesMap flash_bundles_map;
  ClusterBundlesMap cluster_bundles_map;
  std::map<std::pair<Opflash*, Cluster*>, TimingTPCBundle::pointer> flash_cluster_bundles_map;

  std::vector<TimingTPCBundle::pointer> consistent_bundles;

  for (auto it = pre_bundles.begin(); it != pre_bundles.end(); ++it) {
    auto bundle = *it;
    auto flash = bundle->get_flash();
    auto cluster = bundle->get_main_cluster();

    if (bundle->get_consistent_flag()) consistent_bundles.push_back(bundle);

    if (flash_bundles_map.find(flash) == flash_bundles_map.end()) {
      std::vector<TimingTPCBundle::pointer> bundle_v;
      bundle_v.push_back(bundle);
      flash_bundles_map[flash] = bundle_v;
    }
    else
      flash_bundles_map[flash].push_back(bundle);
    if (cluster_bundles_map.find(cluster) == cluster_bundles_map.end()) {
      std::vector<TimingTPCBundle::pointer> bundle_v;
      bundle_v.push_back(bundle);
      cluster_bundles_map[cluster] = bundle_v;
    }
    else
      cluster_bundles_map[cluster].push_back(bundle);

    flash_cluster_bundles_map[std::make_pair(flash, cluster)] = bundle;
  } // end pre-selected bundle loop

  // Deterministic iteration order over flashes/clusters/bundles. Without
  // these, the LASSO matrix column / row order depends on heap allocator
  // ordering of Opflash* / Cluster* / shared_ptr addresses --- two runs
  // with identical inputs then permute matrix columns and produce slightly
  // different solution() vectors, enough to flip bundles across the
  // m_strength_cutoff threshold.
  //
  // Outer order: flash_id (stable, from the input tensor row index).
  // Cluster order: global index from the length-sorted 'clusters' vector.
  // Inner per-flash bundles: cluster_index_id (same global index).
  auto sort_inner_by_cluster_idx = [](FlashBundlesMap& m) {
    for (auto& kv : m) {
      std::sort(kv.second.begin(), kv.second.end(),
                [](const TimingTPCBundle::pointer& a,
                   const TimingTPCBundle::pointer& b) {
                  return a->get_cluster_index_id() < b->get_cluster_index_id();
                });
    }
  };
  auto flash_iter_order = [](const FlashBundlesMap& m) {
    std::vector<Opflash*> v;
    v.reserve(m.size());
    for (auto& kv : m) v.push_back(kv.first);
    std::sort(v.begin(), v.end(),
              [](Opflash* a, Opflash* b) { return a->get_flash_id() < b->get_flash_id(); });
    return v;
  };
  auto cluster_iter_order = [&global_cluster_idx_map](const ClusterBundlesMap& m) {
    std::vector<Cluster*> v;
    v.reserve(m.size());
    for (auto& kv : m) v.push_back(kv.first);
    std::sort(v.begin(), v.end(), [&](Cluster* a, Cluster* b) {
      return global_cluster_idx_map.at(a) < global_cluster_idx_map.at(b);
    });
    return v;
  };
  sort_inner_by_cluster_idx(flash_bundles_map);

  TimingTPCBundleSelection to_be_removed;
  for (auto good_bundle : consistent_bundles) {
    auto flash = good_bundle->get_flash();
    auto cluster = good_bundle->get_main_cluster();
    auto flash_bundles = flash_bundles_map[flash];
    auto cluster_bundles = cluster_bundles_map[cluster];

    for (auto bundle : cluster_bundles) {
      if (bundle == good_bundle) continue;
      if (bundle->get_consistent_flag()) continue;
      to_be_removed.push_back(bundle);
    }
  }
  remove_bundle_selection(
    to_be_removed, flash_bundles_map, cluster_bundles_map, flash_cluster_bundles_map);
  remove_bundle_selection(to_be_removed, pre_bundles);

  to_be_removed.clear();

  // set parameters (used for both matching rounds)
  double lambda = 0.1;
  double delta_charge = 0.01;
  double delta_light = 0.025;
  double delta_shape = 0.01; // used for 2nd matching round only

  // set "fudge factors" for the total error
  // double factor_pe = 1.0;
  // double factor_pe_err = 1.0;

  uint nopdet = 0;
  std::vector<int> opdet_idx_v;
  for (size_t idet = 0; idet < opdet_mask.size(); idet++) {
    if (opdet_mask.at(idet) == 1) {
      opdet_idx_v.push_back(int(idet));
      nopdet++;
    }
  }
  log->debug("nopdet {}", nopdet);
  log->debug("opdet_idx_v size {}", opdet_idx_v.size());

  // * first matching round
  {
    uint nbundle = pre_bundles.size();
    uint nflash = flash_bundles_map.size();
    uint ncluster = cluster_bundles_map.size();

    auto flashes_ordered  = flash_iter_order(flash_bundles_map);
    auto clusters_ordered = cluster_iter_order(cluster_bundles_map);

    // create map between flash object and flash vector/matrix index
    std::map<Opflash*, int> flash_idx_map;
    // create map between cluster object and cluster vector/matrix index
    std::map<Cluster*, int> cluster_idx_map;

    int cluster_idx = 0;
    int flash_idx = 0;
    for (auto* cluster : clusters_ordered) {
      cluster_idx_map[cluster] = cluster_idx;
      cluster_idx++;
    }
    for (auto* flash : flashes_ordered) {
      flash_idx_map[flash] = flash_idx;
      flash_idx++;
    }

    for (auto* flash : flashes_ordered) {
      auto& bundles = flash_bundles_map[flash];
      for (size_t i = 0; i < bundles.size(); i++) {
        auto bundle = bundles.at(i);
        if (bundle->get_consistent_flag()) {
          log->debug("flash {}, cluster {}, consistent bundle: with ks_dis {}, chi2/ndf {}, ndf "
                     "{}, pred light {}, meas light {}",
                     flash->get_flash_id(),
                     global_cluster_idx_map[bundle->get_main_cluster()],
                     int(bundle->get_ks_dis() * 1000) / 1000.,
                     int(bundle->get_chi2() / bundle->get_ndf() * 100) / 100.,
                     bundle->get_ndf(),
                     int(bundle->get_total_pred_light() * 100) / 100.,
                     int(flash->get_total_PE() * 100) / 100.);
        }
      }
    }

    Ress::vector_t M = Ress::vector_t::Zero(nopdet * nflash);                   // measurement
    Ress::matrix_t P = Ress::matrix_t::Zero(nopdet * nflash, nbundle + nflash); // prediction
    Ress::vector_t MF = Ress::vector_t::Zero(ncluster + nflash);                // measurement flag
    Ress::matrix_t PF =
      Ress::matrix_t::Zero(ncluster + nflash, nbundle + nflash); // prediction flag
    Ress::vector_t weights = Ress::vector_t::Zero(nbundle + nflash);

    log->debug("M dim {}", M.rows());
    log->debug("P dim {} {}", P.rows(), P.cols());
    log->debug("MF dim {} {}", MF.rows(), MF.cols());
    log->debug("PF dim {} {}", PF.rows(), PF.cols());
    log->debug("weights dim {}", weights.rows());

    std::vector<std::pair<Opflash*, Cluster*>> pairs;

    size_t i = 0;  // flash index counter
    size_t ik = 0; // weights index counter
    for (auto* flash : flashes_ordered) {
      auto& bundles = flash_bundles_map[flash];

      for (uint j = 0; j < nopdet; j++) {
        auto opdet_idx = opdet_idx_v.at(j);
        auto pe = flash->get_PE(opdet_idx);
        auto pe_err = sqrt(flash->get_PE(opdet_idx) + pow(flash->get_PE_err(opdet_idx), 2));

        M(i * nopdet + j) = pe / pe_err;              // measurement term
        P(i * nopdet + j, nbundle + i) = pe / pe_err; // measurement alone term
      }

      for (size_t k = 0; k < bundles.size(); k++) {
        auto bundle = bundles.at(k);
        auto pred_flash = bundle->get_pred_flash();

        for (uint j = 0; j < nopdet; j++) {
          auto opdet_idx = opdet_idx_v.at(j);
          auto pred_pe = pred_flash.at(opdet_idx);
          auto pe_err = sqrt(flash->get_PE(opdet_idx) + pow(flash->get_PE_err(opdet_idx), 2));
          P(i * nopdet + j, pairs.size()) = pred_pe / pe_err;
        }

        pairs.push_back(std::make_pair(flash, bundle->get_main_cluster()));

        auto meas_pe_tot = flash->get_total_PE();
        auto pred_pe_tot = bundle->get_total_pred_light();
        if (abs(pred_pe_tot - meas_pe_tot) > 0.3 * meas_pe_tot) {
          weights(ik) = abs(pred_pe_tot - meas_pe_tot) / meas_pe_tot;
          ik++;
        }
        else {
          weights(ik) = 0.3;
          ik++;
        }
      } // loop over bundles in flash
      // MF(ncluster+i) = 0;
      PF(ncluster + i, nbundle + i) = 1. / delta_light;

      flash_idx_map[flash] = nbundle + i;
      i++;
    } // end loop over flashes

    for (uint i = 0; i < nflash; i++) {
      weights(nbundle + i) = 0.5;
    }

    for (uint k = 0; k < ncluster; k++) {
      MF(k) = 1. / delta_charge;
    }

    for (size_t n = 0; n < pairs.size(); n++) {
      auto cluster = pairs.at(n).second;
      PF(cluster_idx_map[cluster], n) = 1. / delta_charge;
    }

    Ress::matrix_t PT = P.transpose();
    Ress::matrix_t PFT = PF.transpose();

    Ress::vector_t y = PT * M + PFT * MF; // predicted x measured + p1/p2 x measured (bi x Mij)
    Ress::matrix_t X = PT * P + PFT * PF; // predicted^2 + p1/p2 x predicted

    Ress::vector_t initial = Ress::vector_t::Zero(nbundle + nflash);
    for (size_t n = 0; n < pairs.size(); n++) {
      initial(n) = 1.0;
    }
    log->debug("initial dim {}", initial.rows());

    Ress::Params params;
    params.model = Ress::lasso;
    params.lambda = lambda;
    // params.tolerance = 1e-2;

    log->debug("solving");
    Ress::vector_t solution = Ress::solve(X, y, params, initial, weights);

    int n = 0;
    for (auto* flash : flashes_ordered) {
      auto& bundles = flash_bundles_map[flash];
      for (size_t k = 0; k < bundles.size(); k++) {
        auto bundle = bundles.at(k);

        if (solution(n) > m_strength_cutoff || m_beamonly)
          log->debug("first match: flash {} and cluster {}, solution={}",
                     flash->get_flash_id(),
                     global_cluster_idx_map[bundle->get_main_cluster()],
                     solution(n));
        else {
          to_be_removed.push_back(bundle);
        }
        n++;
      }
    }
    int m = 0;
    for (auto* flash : flashes_ordered) {
      if (solution(nbundle + m) != 0)
        log->debug(
          "flash-only: flash {}, solution={}", flash->get_flash_id(), solution(nbundle + m));
      m++;
    }
    remove_bundle_selection(
      to_be_removed, flash_bundles_map, cluster_bundles_map, flash_cluster_bundles_map);
    remove_bundle_selection(to_be_removed, pre_bundles);
    to_be_removed.clear();
  } // end matching round
  // second matching round
  {
    uint nbundle = pre_bundles.size();
    uint nflash = flash_bundles_map.size();
    uint ncluster = cluster_bundles_map.size();

    // Rebuild ordered iteration (round 1 may have removed bundles/flashes/clusters).
    auto flashes_ordered  = flash_iter_order(flash_bundles_map);
    auto clusters_ordered = cluster_iter_order(cluster_bundles_map);

    // create map between cluster object and cluster vector/matrix index
    // create map between flash object and flash vector/matrix index
    std::map<Cluster*, int> cluster_idx_map;
    std::map<Opflash*, int> flash_idx_map;

    int cluster_idx = 0;
    int flash_idx = 0;
    for (auto* cluster : clusters_ordered) {
      cluster_idx_map[cluster] = cluster_idx;
      cluster_idx++;
    }
    for (auto* flash : flashes_ordered) {
      flash_idx_map[flash] = flash_idx;
      flash_idx++;
    }

    Ress::vector_t M = Ress::vector_t::Zero(nopdet * nflash);          // measurement
    Ress::matrix_t P = Ress::matrix_t::Zero(nopdet * nflash, nbundle); // prediction
    Ress::vector_t MF = Ress::vector_t::Zero(ncluster);                // measurement flag
    Ress::matrix_t PF = Ress::matrix_t::Zero(ncluster, nbundle);       // prediction flag
    Ress::vector_t weights = Ress::vector_t::Zero(nbundle);
    std::vector<std::pair<Opflash*, Cluster*>> pairs;

    log->debug("M dim {}", M.rows());
    log->debug("P dim {} {}", P.rows(), P.cols());
    log->debug("MF dim {} {}", MF.rows(), MF.cols());
    log->debug("PF dim {} {}", PF.rows(), PF.cols());
    log->debug("weights dim {}", weights.rows());

    size_t i = 0;  // flash index counter
    size_t ik = 0; // weights index counter
    log->debug("flash_bundles_map size {}", flash_bundles_map.size());
    for (auto* flash : flashes_ordered) {
      auto& bundles = flash_bundles_map[flash];

      for (uint j = 0; j < nopdet; j++) {
        auto opdet_idx = opdet_idx_v.at(j);
        auto pe = flash->get_PE(opdet_idx);
        auto pe_err = sqrt(flash->get_PE(opdet_idx) + pow(flash->get_PE_err(opdet_idx), 2));

        M(i * nopdet + j) = pe / pe_err; // measurement term
      }

      for (size_t k = 0; k < bundles.size(); k++) {
        auto bundle = bundles.at(k);
        auto ks_dis = bundle->get_ks_dis();
        auto pred_flash = bundle->get_pred_flash();

        for (uint j = 0; j < nopdet; j++) {
          auto opdet_idx = opdet_idx_v.at(j);
          auto pred_pe = pred_flash.at(opdet_idx);
          auto pe_err = sqrt(flash->get_PE(opdet_idx) + pow(flash->get_PE_err(opdet_idx), 2));
          P(i * nopdet + j, pairs.size()) = pred_pe / pe_err;
        }

        pairs.push_back(std::make_pair(flash, bundle->get_main_cluster()));

        auto meas_pe_tot = flash->get_total_PE();
        auto pred_pe_tot = bundle->get_total_pred_light();

        if (abs(pred_pe_tot - meas_pe_tot) > 0.3 * meas_pe_tot) {
          weights(ik) =
            abs(pred_pe_tot - meas_pe_tot) / meas_pe_tot + delta_shape * nopdet * ks_dis / lambda;
          ik++;
        }
        else {
          weights(ik) = 0.3 + delta_shape * nopdet * ks_dis / lambda;
          ik++;
        }
      } // loop over bundles in flash
      i++;
    } // end loop over flashes

    for (uint k = 0; k < ncluster; k++) {
      MF(k) = 1. / delta_charge;
    }

    for (size_t n = 0; n < pairs.size(); n++) {
      auto cluster = pairs.at(n).second;
      PF(cluster_idx_map[cluster], n) = 1. / delta_charge;
    }

    Ress::matrix_t PT = P.transpose();
    Ress::matrix_t PFT = PF.transpose();

    Ress::vector_t y = PT * M + PFT * MF;
    Ress::matrix_t X = PT * P + PFT * PF;

    Ress::vector_t initial = Ress::vector_t::Zero(nbundle);
    for (size_t n = 0; n < pairs.size(); n++) {
      initial(n) = 1.0;
    }
    Ress::Params params;
    params.model = Ress::lasso;
    params.lambda = lambda;
    // params.tolerance = 1e-2;

    log->debug("solving");
    Ress::vector_t solution = Ress::solve(X, y, params, initial, weights);
    log->debug("solution size {}", solution.size());
    int n = 0;
    for (auto* flash : flashes_ordered) {
      auto& bundles = flash_bundles_map[flash];
      for (size_t k = 0; k < bundles.size(); k++) {
        auto bundle = bundles.at(k);
        bundle->set_strength(solution(n));

        if (solution(n) > m_strength_cutoff || m_beamonly) {
          log->debug("second match: flash {} and cluster {}, time {}, meas PE {}, pred PE {}, "
                     "solution {}, ks_dis {}, chi2/ndf {}, consistent {}",
                     flash->get_flash_id(),
                     global_cluster_idx_map[bundle->get_main_cluster()],
                     int(flash->get_time()) / 1e3,
                     int(flash->get_total_PE() * 100) / 100.,
                     int(bundle->get_total_pred_light() * 100) / 100.,
                     int(solution(n) * 1e4) / 10000.,
                     int(bundle->get_ks_dis() * 1000) / 1000.,
                     int(bundle->get_chi2() / bundle->get_ndf() * 100) / 100.,
                     bundle->get_consistent_flag());
        }
        else {
          to_be_removed.push_back(bundle);
        }
        n++;
      }
    }
    remove_bundle_selection(
      to_be_removed, flash_bundles_map, cluster_bundles_map, flash_cluster_bundles_map);
    remove_bundle_selection(to_be_removed, pre_bundles);
    to_be_removed.clear();

    // save only the best match for each cluster
    // at this point, one cluster can be matched to at most one flash
    // but one flash can be matched to multiple clusters
    std::map<int, std::pair<Opflash*, double>> matched_pairs;
    for (size_t i = 0; i != pairs.size(); i++) {
      if (solution(i) > m_strength_cutoff) {
        int cluster_idx = cluster_idx_map[pairs.at(i).second];
        auto flash = pairs.at(i).first;
        if (matched_pairs.find(cluster_idx) == matched_pairs.end()) {
          matched_pairs[cluster_idx] = std::make_pair(flash, solution(i));
        }
        else {
          if (solution(i) > matched_pairs[cluster_idx].second) {
            matched_pairs[cluster_idx] = std::make_pair(flash, solution(i));
          }
        }
      }
    }
    // * good matches should have ks_dis < 0.2 and chi2/ndf < 20
    TimingTPCBundleSelection results_bundles;

    for (auto it = clusters.begin(); it != clusters.end(); ++it) {
      auto cluster = *it;
      if (cluster_idx_map.find(cluster) != cluster_idx_map.end()) {
        auto cluster_idx = cluster_idx_map[cluster];
        if (matched_pairs.find(cluster_idx) != matched_pairs.end()) {
          auto flash = matched_pairs[cluster_idx].first;
          // auto strength = matched_pairs[cluster_idx].second;
          auto bundle = flash_cluster_bundles_map[std::make_pair(flash, cluster)];
          results_bundles.push_back(bundle);
        }
        else {
          // if cluster is not present in the matched pairs
          // create a bundle with no flash
          Opflash* flash = nullptr;
          auto bundle = std::make_shared<TimingTPCBundle>(flash, cluster, 0, cluster_idx);
          bundle->set_strength(0);
          results_bundles.push_back(bundle);
        }
      }
    }
    organize_bundles(results_bundles, flash_cluster_bundles_map);

    // * given results_bundles, create new FlashBundlesMap
    FlashBundlesMap results_flash_bundles_map;
    for (auto it = results_bundles.begin(); it != results_bundles.end(); ++it) {
      auto bundle = *it;
      auto flash = bundle->get_flash();
      if (results_flash_bundles_map.find(flash) == results_flash_bundles_map.end()) {
        std::vector<TimingTPCBundle::pointer> bundles;
        bundles.push_back(bundle);
        results_flash_bundles_map[flash] = bundles;
      }
      else {
        results_flash_bundles_map[flash].push_back(bundle);
      }
    }

    // debug block
    {
      for (auto [flash, bundles] : results_flash_bundles_map) {
        for (const auto& bundle : bundles) {
          log->debug("results_flash_bundles_map: flash id {} time {} and cluster gidx {}, "
                     "total_pred_light {}, strength {}, ks_dis {}, chi2/ndf {}",
                     flash->get_flash_id(),
                     flash->get_time(),
                     global_cluster_idx_map[bundle->get_main_cluster()],
                     bundle->get_total_pred_light(),
                     int(bundle->get_strength() * 1e4) / 10000.,
                     int(bundle->get_ks_dis() * 1000) / 1000.,
                     int(bundle->get_chi2() / bundle->get_ndf() * 100) / 100.);
        }
      }
    }

    // BEE debug direct imaging output and dead blobs
    log->debug("done with matching");
    if (!m_bee_dir.empty()) {
      std::string sub_dir = String::format("%s/%d", m_bee_dir, m_bee_index);
      Persist::assuredir(sub_dir);
      QLMatch::dump_bee_3d(
        *root_live.get(),
        String::format("%s/%d-img-apa%d.json", sub_dir, m_bee_index, m_anode->ident()));
      // QLMatch::dump_bee_bundle(
      //   results_flash_bundles_map, global_cluster_idx_map, String::format("%s/%d-op-apa%d.json", sub_dir, m_bee_index, m_anode->ident()));
      QLMatch::dump_light(
        flashes,
        flash_bundles_map,
        global_cluster_idx_map,
        String::format("%s/%d-op-apa%d.json", sub_dir, m_bee_index, m_anode->ident()));
      m_bee_index++;
    }
    log->debug(em("dump bee"));

  } // end second matching round

  // Apply matched t0s and tag beam-window clusters with
  // Clus::Facade::"beam_flash" / main_cluster so the downstream
  // MABC pipeline (ClusteringTaggerFlagTransfer, ClusteringRecoveringBundle,
  // TaggerCheckNeutrino, ...) can pick the in-beam main cluster.
  //
  // Selection rule (mirrors WCP for uboone):
  //   - Any matched cluster whose flash satisfies |flash_time| <
  //     m_max_beam_flash_time gets the beam_flash flag.
  //   - Among those, the cluster whose flash has the smallest |time|
  //     gets the main_cluster flag.
  Cluster* main_cluster_candidate = nullptr;
  double main_cluster_abs_time = std::numeric_limits<double>::infinity();
  for (auto* flash : flash_iter_order(flash_bundles_map)) {
    auto& bundles = flash_bundles_map[flash];
    const double abs_flash_time = std::abs(flash->get_time());
    const bool in_beam = abs_flash_time < m_max_beam_flash_time;
    for (auto bundle : bundles) {
      auto* cluster = bundle->get_main_cluster();
      cluster->set_cluster_t0(flash->get_time() * units::ns);
      if (in_beam) {
        cluster->set_flag("beam_flash");
        if (abs_flash_time < main_cluster_abs_time) {
          main_cluster_abs_time = abs_flash_time;
          main_cluster_candidate = cluster;
        }
      }
      log->debug(
        "flash_bundles_map: flash id {} time {} ns, cluster gidx {} total_pred_light {} t0 {} in_beam {}",
        flash->get_flash_id(),
        flash->get_time(),
        global_cluster_idx_map[cluster],
        bundle->get_total_pred_light(),
        cluster->get_cluster_t0(),
        in_beam);
    }
  }
  if (main_cluster_candidate) {
    main_cluster_candidate->set_flag("main_cluster");
    log->debug("QLMatching: main_cluster tagged on cluster gidx {} "
               "(|flash_time|={} ns, threshold={} ns)",
               global_cluster_idx_map[main_cluster_candidate],
               main_cluster_abs_time, m_max_beam_flash_time);
  } else {
    log->debug("QLMatching: no matched cluster within |flash_time| < {} ns; "
               "main_cluster flag not set", m_max_beam_flash_time);
  }

  {
    // Pad cluster_scalar so every live cluster carries the same flag_* keys.
    // Without this, Aux::TensorDM::as_tensors silently drops flag values set
    // on only a subset of clusters (Dataset::append uses the first cluster's
    // schema), so the main_cluster / beam_flash flags set above would be lost
    // at the QLMatching -> PointTreeMerging -> MABC tensor IO boundary, and
    // TaggerCheckNeutrino would see no main_cluster.
    Clus::Facade::normalize_cluster_flags(*grouping, log, "live", charge_ident);

    ITensor::vector outtens;

    auto tens_live = Aux::TensorDM::as_tensors(*root_live, inpath + "/live");
    log->debug("Output {} tensors for live", tens_live.size());
    outtens.insert(outtens.end(), tens_live.begin(), tens_live.end());

    auto root_dead = Aux::TensorDM::as_pctree(charge_tens, inpath + "/dead");
    auto tens_dead = Aux::TensorDM::as_tensors(*root_dead, inpath + "/dead");
    log->debug("Output {} tensors for dead", tens_dead.size());
    outtens.insert(outtens.end(), tens_dead.begin(), tens_dead.end());

    out = Aux::TensorDM::as_tensorset(outtens, charge_ident);
  }

  if (flashes.size() > 0) {
    log->debug("total_charge_blob {} total_charge_point {} total_charge_blob_all {}",
               total_charge_blob / flashes.size(),
               total_charge_point / flashes.size(),
               total_charge_blob_all);
  }
  else {
    log->debug("total_charge_blob {} total_charge_point {} total_charge_blob_all {}",
               0, 0, total_charge_blob_all);
  }

  // dumy output
  // out = invec[0];
  m_count++;
  return true;
}

void WireCell::QLMatch::QLMatching::remove_bundle_selection(TimingTPCBundleSelection to_be_removed,
                                                            TimingTPCBundleSet& bundle_set)
{
  for (auto it = to_be_removed.begin(); it != to_be_removed.end(); ++it) {
    auto rm_bundle = *it;
    bundle_set.erase(rm_bundle);
  }
}

void WireCell::QLMatch::QLMatching::remove_bundle_selection(
  TimingTPCBundleSelection to_be_removed,
  FlashBundlesMap& flash_bundles_map,
  ClusterBundlesMap& cluster_bundles_map,
  std::map<std::pair<Opflash*, Cluster*>, TimingTPCBundle::pointer>& flash_cluster_bundles_map)
{
  for (auto it = to_be_removed.begin(); it != to_be_removed.end(); ++it) {
    auto rm_bundle = *it;
    auto rm_flash = rm_bundle->get_flash();
    auto rm_cluster = rm_bundle->get_main_cluster();

    flash_cluster_bundles_map.erase(std::make_pair(rm_flash, rm_cluster));
    {
      auto flash_it = flash_bundles_map.find(rm_flash);
      if (flash_it != flash_bundles_map.end()) {
        auto& temp_bundles = flash_it->second;
        auto vec_it = find(temp_bundles.begin(), temp_bundles.end(), rm_bundle);
        if (vec_it != temp_bundles.end()) { temp_bundles.erase(vec_it); }
        if (temp_bundles.empty()) { flash_bundles_map.erase(flash_it); }
      }
    }
    {
      auto cluster_it = cluster_bundles_map.find(rm_cluster);
      if (cluster_it != cluster_bundles_map.end()) {
        auto& temp_bundles = cluster_it->second;
        auto vec_it = find(temp_bundles.begin(), temp_bundles.end(), rm_bundle);
        if (vec_it != temp_bundles.end()) { temp_bundles.erase(vec_it); }
        if (temp_bundles.empty()) { cluster_bundles_map.erase(cluster_it); }
      }
    }
  }
}

void WireCell::QLMatch::QLMatching::organize_bundles(
  TimingTPCBundleSelection& results_bundles,
  std::map<std::pair<Opflash*, Cluster*>, TimingTPCBundle::pointer>& flash_cluster_bundles_map)
{
  // construct map for all results
  log->debug("organizing bundles");
  std::map<Opflash*, TimingTPCBundleSelection> eval_flash_bundles_map;
  for (auto it = results_bundles.begin(); it != results_bundles.end(); ++it) {
    auto bundle = *it;
    auto flash = bundle->get_flash();
    if (flash == nullptr) continue;
    if (eval_flash_bundles_map.find(flash) == eval_flash_bundles_map.end()) {
      TimingTPCBundleSelection bundles;
      bundles.push_back(bundle);
      eval_flash_bundles_map[flash] = bundles;
    }
    else {
      eval_flash_bundles_map[flash].push_back(bundle);
    }
  }

  TimingTPCBundleSelection second_round_bundles;
  // TimingTPCBundleSelection third_round_bundles;

  //* first round: evaluate primary matches
  for (auto it = eval_flash_bundles_map.begin(); it != eval_flash_bundles_map.end(); ++it) {
    auto& orig_bundles = it->second;
    auto flash = it->first;

    TimingTPCBundleSelection to_be_removed;
    TimingTPCBundle* best_bundle = nullptr;
    double best_strength = 0;

    // * find the bundle with the highest strength (best solution)
    for (auto jt = orig_bundles.begin(); jt != orig_bundles.end(); ++jt) {
      auto bundle = *jt;
      auto strength = bundle->get_strength();
      if (strength > best_strength) {
        best_strength = strength;
        best_bundle = bundle.get();
      }
    }
    log->debug("best bundle strength {} for flash {}", best_strength, flash->get_flash_id());
    // * see if performance improves by combining bundles
    for (auto jt = orig_bundles.begin(); jt != orig_bundles.end(); ++jt) {
      auto bundle = (*jt).get();
      if (bundle != best_bundle) {
        // * if combination of bundles passes examination
        // ! TODO: this needs to be validated, currently only performing cut on ks_dis and chi2/ndf
        if (best_bundle->examine_bundle(bundle)) {
          best_bundle->add_bundle(bundle);
          to_be_removed.push_back(*jt);
        }
        else {
          // second_round_bundles.push_back(*jt);
          // ! TODO: not doing a second round right now
          to_be_removed.push_back(*jt);
        }
      }
    } // loop over orig bundles associated with this flash

    for (auto it = to_be_removed.begin(); it != to_be_removed.end(); it++) {
      results_bundles.erase(find(results_bundles.begin(), results_bundles.end(), *it));
    }
    to_be_removed.clear();

    if (best_bundle != nullptr) {
      // * only actually merge clusters if the flash is a beam-related flash
      // if (flash->get_time() > m_beam_mintime && flash->get_time() < m_beam_maxtime){
      if (flash->get_time() > m_beam_mintime && flash->get_time() < m_beam_maxtime) {
        // best_bundle->examine_merge_clusters();
        log->debug("after merge, meas pe {}, pred pe {}, ks_dis {}, chi2/ndf {}",
                   int(flash->get_total_PE() * 100) / 100.,
                   int(best_bundle->get_total_pred_light() * 100) / 100.,
                   int(best_bundle->get_ks_dis() * 1000) / 1000.,
                   int(best_bundle->get_chi2() / best_bundle->get_ndf() * 100) / 100.);
      }
    }
  } // loop over flash

  TimingTPCBundleSelection to_be_removed;

  for (auto it = results_bundles.begin(); it != results_bundles.end(); it++) {
    auto bundle = *it;
    auto flash = bundle->get_flash();
    if (flash->get_time() < m_beam_mintime || flash->get_time() > m_beam_maxtime) {
      if (bundle->get_ks_dis() > 0.2 || bundle->get_chi2() / bundle->get_ndf() > 20) {
        to_be_removed.push_back(bundle);
        continue;
      }
      else if (abs(flash->get_total_PE() - bundle->get_total_pred_light()) >
               0.5 * flash->get_total_PE()) {
        to_be_removed.push_back(bundle);
        continue;
      }
    }
    log->debug("organize_bundles: flash {}, cluster *, strength {}, meas pe {}, pred pe {}, ks_dis "
               "{}, chi2/ndf {}",
               flash->get_flash_id(),
               // global_cluster_idx_map[bundle->get_main_cluster()],
               int(bundle->get_strength() * 100) / 100.,
               int(flash->get_total_PE() * 100) / 100.,
               int(bundle->get_total_pred_light() * 100) / 100.,
               int(bundle->get_ks_dis() * 1000) / 1000.,
               int(bundle->get_chi2() / bundle->get_ndf() * 100) / 100.);
  }
  for (auto it = to_be_removed.begin(); it != to_be_removed.end(); it++) {
    results_bundles.erase(find(results_bundles.begin(), results_bundles.end(), *it));
  }

  // // * second round
  // TimingTPCBundleSelection to_be_removed;
  // std::set<Opflash*> tried_flashes;
  // for (auto it=second_round_bundles.begin(); it!=second_round_bundles.end(); it++){
  //   TimingTPCBundle *bundle = (*it).get();
  //   automain_cluster = bundle->get_main_cluster();
  //   bool used = false;
  //   for (auto jt = results_bundles.rbegin(); jt != results_bundles.rend(); jt++){
  //     TimingTPCBundle *best_bundle = (*jt);
  //   }
  // }
}
