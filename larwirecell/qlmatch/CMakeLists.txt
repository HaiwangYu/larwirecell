
cet_make_library(LIBRARY_NAME WireCellQLMatch
    SOURCE
        Opflash.cxx
        QLMatching.cxx
        TimingTPCBundle.cxx
        util.cxx
    LIBRARIES
    PUBLIC
        larsim::PhotonPropagation
        art_plugin_support::toolMaker
        WireCell::Util
        WireCell::Aux
        WireCell::Clus
        WireCell::Iface
)

# One-off analyzer module used to dump SBND OpDet geometry to stdout for the
# wire-cell-toolkit/match port. See SBNDOpDetDumper_module.cc and
# dump_sbnd_opdets.fcl in wcp-porting-img/sbnd/standalone-sample/.
cet_build_plugin(SBNDOpDetDumper art::EDAnalyzer
    LIBRARIES PRIVATE
    larcore::Geometry_Geometry_service
    larcorealg::Geometry
    larcore::headers
    lardata::DetectorInfoServices_LArPropertiesServiceStandard_service
    larsim::IonizationScintillation
    larsim::PhotonPropagation
    art_root_io::TFileService_service
)

install_fhicl()