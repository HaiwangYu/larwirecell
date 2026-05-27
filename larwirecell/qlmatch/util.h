#ifndef WIRECELLDEV_QLMATCH_UTIL
#define WIRECELLDEV_QLMATCH_UTIL

#include "WireCellIface/ITensorSet.h"
#include "WireCellUtil/PointTree.h"

#include "TimingTPCBundle.h"

namespace WireCell::QLMatch {
  void dump_bee_3d(const PointCloud::Tree::Points::node_t& root, const std::string& fn);
  void dump_bee_flash(const ITensorSet::pointer& ts, const std::string& fn);
  void dump_bee_bundle(const FlashBundlesMap& f2bundle,
                       const std::map<WireCell::Clus::Facade::Cluster*, int>& cluster_idx_map,
                       const std::string& fn);

  // Dump every flash (matched or not). For a flash with a matched charge
  // cluster (present in f2bundle with a qualifying bundle) the "cluster_id"
  // and "op_pes_pred" are filled exactly as dump_bee_bundle does; flashes
  // with no match are still emitted with an empty "cluster_id".
  void dump_light(const std::vector<Opflash::pointer>& flashes,
                  const FlashBundlesMap& f2bundle,
                  const std::map<WireCell::Clus::Facade::Cluster*, int>& cluster_idx_map,
                  const std::string& fn);
}

#endif