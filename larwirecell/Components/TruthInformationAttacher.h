/**
 * @file   larwirecell/Components/TruthInformationAttacher.h
 * @brief  Stamp the art run/subrun/event -- and, on MC, the neutrino and
 *         particle-flow truth -- onto an ITensorSet.
 * @date   2026-09-29
 *
 * A SUPERSET of TensorSetMetadataAttacher (which it replaces in the SBND
 * jobs, ai-helper issue 33).  It is an art-event visitor (wcls calls
 * visit(art::Event&) at the START of the event, before the pgraph runs) and an
 * ITensorSetFilter (it sits on a tensor edge).  It never deserializes the
 * point-cloud tree: the input tensors are shared by pointer and the truth
 * tensors, if any, are appended.
 *
 * ALWAYS: "runNo", "subRunNo", "eventNo" are added to the set metadata (keys
 * configurable, as TensorSetMetadataAttacher).
 *
 * truth=true AND the event carries a neutrino MCTruth (MC): two 2D double
 * tensors are appended.  Units LArSoft native: cm, ns, GeV.  Column names are
 * in each tensor's metadata "columns"; "datatype" is "truth_nu" / "truth_pf".
 *
 *   truth_nu  (datapath nu_datapath, default "truth/%d/nu"), one row per
 *             generator MCTruth that has a neutrino (NeutrinoSet()), in MCTruth
 *             order (row 0 = the "main" interaction, as the labeler's nu_*):
 *       nu_idx    generator-MCTruth index (= the labeler's nu_idx, the Bee mc
 *                 node id 9000000+nu_idx)
 *       pdg       neutrino PDG code
 *       ccnc      0 CC, 1 NC                        (MCNeutrino::CCNC)
 *       mode      MCNeutrino::Mode()
 *       int_type  MCNeutrino::InteractionType()
 *       flavor    0 none, 1 nue, 2 numu, 3 nutau (CC), 4 NC   (labeler nu_flavor)
 *       E         neutrino energy [GeV]            (Nu().Momentum(0).E())
 *       vtx_x, vtx_y, vtx_z  interaction vertex [cm]  (Nu().Position(0))
 *       t         interaction time [ns]            (Nu().Position(0).T())
 *       edep      deposited energy [GeV]: sum of SimEnergyDeposit::Energy()
 *                 over deposits whose |trackid| descends from this interaction
 *                 (the labeler's nu_edep)
 *
 *   truth_pf  (datapath pf_datapath, default "truth/%d/pf"), one row per
 *             particle of the truth PARTICLE FLOW -- the selection of the
 *             labeler's Bee "mc" tree: MCParticles derived from a beam
 *             neutrino (MCParticle<->MCTruth Assns, Origin()==kBeamNeutrino)
 *             with KE > pf_ke_min, in MCParticle order:
 *       nu_row    row of this particle's interaction in truth_nu (-1: none)
 *       trackid, parent_trackid (nearest KEPT ancestor, 0 = attached directly
 *                 to the neutrino), mother_trackid (G4 mother), pdg,
 *       process   G4 creation-process code (the labeler's truth_per_track code)
 *       E, KE     start total / kinetic energy [GeV]
 *       start_x/y/z/t, end_x/y/z/t  [cm, ns]
 *       start_px/py/pz  [GeV]
 *
 * DATA (no neutrino MCTruth) or truth=false: RSE only, no tensors appended --
 * "the truth tensors exist" is the test for "truth is available".
 *
 * PLACEMENT (SBND): with truth=false immediately upstream of every
 * MultiAlgBlobClustering whose RSE matters (set rse_from_metadata on it), and
 * one truth=true instance after labeler_truth.  Each MUST be listed in the fcl
 * 'inputers' or it never sees an art::Event (it then warns and passes through).
 */

#ifndef LARWIRECELL_COMPONENTS_TRUTHINFORMATIONATTACHER
#define LARWIRECELL_COMPONENTS_TRUTHINFORMATIONATTACHER

#include "WireCellAux/Logger.h"

#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/ITensorSetFilter.h"
#include "larwirecell/Interfaces/IArtEventVisitor.h"

#include <string>
#include <vector>

namespace wcls {
  class TruthInformationAttacher : public WireCell::Aux::Logger,
                                   public IArtEventVisitor,
                                   public WireCell::ITensorSetFilter,
                                   public WireCell::IConfigurable {
  public:
    TruthInformationAttacher();
    virtual ~TruthInformationAttacher();

    /// IArtEventVisitor -- called at event start, before the graph runs.
    virtual void visit(art::Event& event);

    /// ITensorSetFilter
    virtual bool operator()(const input_pointer& in, output_pointer& out);

    /// IConfigurable
    virtual WireCell::Configuration default_configuration() const;
    virtual void configure(const WireCell::Configuration& config);

    static const std::vector<std::string>& nu_columns();
    static const std::vector<std::string>& pf_columns();

  private:
    // configuration
    std::string m_run_key{"runNo"};
    std::string m_sub_key{"subRunNo"};
    std::string m_evt_key{"eventNo"};
    bool m_truth{true};
    std::string m_mctruth_label{"generator"};
    std::string m_mcparticle_label{"largeant"};
    std::string m_deposet_label{"ionandscint:priorSCE"};
    double m_pf_ke_min{10.0};   // MeV, as the labeler's pf_ke_min
    std::string m_nu_datapath{"truth/%d/nu"};
    std::string m_pf_datapath{"truth/%d/pf"};

    // per-event state filled by visit()
    int m_run{0}, m_sub{0}, m_evt{0};
    bool m_seen{false};
    std::vector<double> m_nu;   // row-major [n_nu x nu_columns]
    std::vector<double> m_pf;   // row-major [n_pf x pf_columns]
    size_t m_n_nu{0}, m_n_pf{0};

    size_t m_count{0};
  };
}

#endif
