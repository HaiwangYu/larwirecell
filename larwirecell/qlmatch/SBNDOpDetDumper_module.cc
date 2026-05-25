// SBNDOpDetDumper
//
// Small one-off analyzer module that prints SBND optical-detector geometry to
// stdout in CSV format, prefixed with "OPDET:" so it can be grep'd out of the
// larsoft log. Used to seed the WCT-side semi-analytical-sbnd.json data file
// for the wire-cell-toolkit/match port.
//
// Build alongside an existing larwirecell module (e.g. by dropping the .cc
// in larwirecell/qlmatch/) and add the module name to that subdir's
// CMakeLists.txt. Or build with cetbuildtools as an art module in
// sbndcode/Utilities/.
//
// One-time use. Output the CSV, hand it to build_semi_analytical_sbnd_json.py
// together with semimodel_sbnd-dump.fcl, and check the resulting JSON in.

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art_root_io/TFileService.h"

#include "larcore/Geometry/Geometry.h"
#include "larcore/CoreUtils/ServiceUtil.h"
#include "larcorealg/Geometry/BoxBoundedGeo.h"
#include "larcorealg/Geometry/CryostatGeo.h"
#include "larcorealg/Geometry/OpDetGeo.h"
#include "larcorealg/Geometry/TPCGeo.h"
#include "lardata/DetectorInfoServices/LArPropertiesService.h"
#include "larsim/IonizationScintillation/ISTPC.h"
#include "larsim/PhotonPropagation/PhotonPropagationUtils.h"

#include <iostream>

namespace sbnd {

  class SBNDOpDetDumper : public art::EDAnalyzer {
  public:
    explicit SBNDOpDetDumper(fhicl::ParameterSet const& pset) : EDAnalyzer(pset) {}
    void analyze(art::Event const&) override
    {
      auto const& geom = *(lar::providerFrom<geo::Geometry>());

      // Replicate what SemiAnalyticalModel does at init time: pull the active
      // LAr volume bounds and cathode X from larsoft so the JSON we emit
      // matches the constants the model uses internally.
      const larg4::ISTPC istpc{geom};
      auto const activeVolumes = istpc.extractActiveLArVolume(geom);
      const auto& av = activeVolumes.front();
      const double cathodeX = geom.TPC().GetCathodeCenter().X();
      const double driftDistance = geom.TPC().DriftDistance();
      // VUV absorption length: SemiAnalyticalModel::VUVAbsorptionLength()
      // pulls AbsLengthSpectrum from LArPropertiesService and interpolates at
      // 9.7 eV (Ar VUV peak). Reproduce that here so the WCT-side JSON gets
      // the exact value larsim would use.
      {
        auto const& props = *(lar::providerFrom<detinfo::LArPropertiesService>());
        auto spec = props.AbsLengthSpectrum();
        std::vector<double> ev, ab;
        ev.reserve(spec.size()); ab.reserve(spec.size());
        for (auto const& kv : spec) { ev.push_back(kv.first); ab.push_back(kv.second); }
        const double vuv_abs = phot::interpolate(ev, ab, 9.7, false);
        std::cout << "GEOM:vuv_absorption_length," << vuv_abs << "\n";
      }
      std::cout << "GEOM:active_center_y," << av.CenterY() << "\n";
      std::cout << "GEOM:active_center_z," << av.CenterZ() << "\n";
      std::cout << "GEOM:active_size_y,"   << av.SizeY()   << "\n";
      std::cout << "GEOM:active_size_z,"   << av.SizeZ()   << "\n";
      std::cout << "GEOM:cathode_x,"       << cathodeX     << "\n";
      std::cout << "GEOM:drift_distance,"  << driftDistance << "\n";
      std::cout << "GEOM:n_tpc,"           << geom.NTPC()  << "\n";

      const std::size_t n = geom.NOpDets();
      std::cout << "# OPDET CSV: idx,x_cm,y_cm,z_cm,h_cm,w_cm,type,orientation\n";
      for (std::size_t i = 0; i < n; ++i) {
        geo::OpDetGeo const& opDet = geom.OpDetGeoFromOpDet(i);
        auto const c = opDet.GetCenter();
        double h = -1, w = -1;
        int type = 1, orient = 0;
        if (opDet.isSphere()) {
          type = 1; orient = 0; h = -1; w = -1;
        }
        else if (opDet.isBar()) {
          type = 0;
          h = opDet.Height();
          w = opDet.Length();
          if (opDet.Width() > opDet.Length()) { orient = 2; w = opDet.Width(); }
          else if (opDet.Width() > opDet.Height()) { orient = 1; h = opDet.Width(); }
          else { orient = 0; }
        }
        else {
          type = 2; orient = 0; h = -1; w = -1;
        }
        std::cout << "OPDET:" << i << "," << c.X() << "," << c.Y() << "," << c.Z()
                  << "," << h << "," << w << "," << type << "," << orient << "\n";
      }
    }
  };

} // namespace sbnd

DEFINE_ART_MODULE(sbnd::SBNDOpDetDumper)
