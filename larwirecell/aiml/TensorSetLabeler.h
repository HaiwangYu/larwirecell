/** TensorSetLabeler -- attach truth information to the clustering output
 * ITensorSet (a serialized PointCloud tree).
 *
 * Sits between MultiAlgBlobClustering and the terminal TensorFileSink:
 *
 *   ... -> MABC -> wclsTensorSetLabeler -> TensorFileSink
 *
 * Event level labels (added to the output ITensorSet metadata, following the
 * "frame_apply_at_caf" convention of OpFlashSource):
 *   - runNo/subRunNo/eventNo: art run/subrun/event numbers.
 *   - nu_pdg, nu_ccnc, nu_int_type, nu_energy (GeV), nu_vtx_{x,y,z} (cm),
 *     nu_flavor ("nue"/"numu"/"nc"/"none"): from the generator MCTruth
 *     (cf. Ningclover larwirecell/aiml/Truth2h5.cxx).
 *
 * Track level truth: a new 2D "truth_per_track" tensor, one row per
 * simb::MCParticle (cf. Ningclover TrackIDPIDMap2h5.cxx).  Columns are listed
 * in the tensor metadata "columns"; units are LArSoft native (cm, ns, GeV).
 *
 * Blob level truth: for each blob node of the "live" grouping the dominant
 * G4 track id is written into the blob "scalar" PC as "trackid" (int, -1 if
 * no depo matched).  The trackid <- SimEnergyDeposit -> blob association
 * follows img/src/BlobDepoFill.cxx: each depo is converted to a (slice tick,
 * u/v/w wire-in-plane index) coordinate on its anode face and matched
 * against the blob's slice_index_[min,max) and {u,v,w}_wire_index_[min,max)
 * bounds; per-blob charge (NumElectrons) is accumulated per track id and the
 * argmax wins.
 *
 * TIME OFFSET GUIDANCE.  The raw (non-t0-corrected) blob x is defined by
 * BlobSampler::time2drift: x = x_Wplane + dirx*(t_sig + time_offset)*drift_speed
 * with SBND time_offset = -205us (= sim.tick0_time in
 * cfg/pgrapher/experiment/sbnd/params.jsonnet) and drift_speed = 1.563 mm/us.
 * The WCT sim (DepoTransform start_time = tick0_time - response_plane/speed,
 * chopped back by the Reframer) is arranged so a depo at true time t_dep=0
 * reconstructs at its true x.  Inverting for a depo at (x_true, t_dep):
 *   apparent x: x_app = x_true + dirx*drift_speed*(t_dep + depo_time_offset)
 *   signal time: t_sig = (x_app - x_Wplane)*dirx/drift_speed - time_offset
 *   slice tick:  itick = t_sig/tick   (tick = 0.5us)
 * So "time_offset" here MUST equal the BlobSampler's (-205us for SBND) and
 * "depo_time_offset" absorbs any residual sim-chain shift (default 0; only
 * needed if the detsim frame was shifted w.r.t. the g4 time origin).
 * Note we use the non-t0-corrected raw coordinates and the priorSCE
 * (true position) depos; SCE displacement (<~1cm) is well below blob size.
 *
 * Debug Bee output: when "bee_sink" names a Clus::IBeeSink, a
 * "truth_trackid" Bee points set is written per event: every blob "3d" point
 * in raw coords with cluster_id = the blob's truth trackid.
 */

#ifndef LARWIRECELL_AIML_TENSORSETLABELER
#define LARWIRECELL_AIML_TENSORSETLABELER

#include "WireCellAux/Logger.h"
#include "WireCellClus/IBeeSink.h"
#include "WireCellIface/IAnodePlane.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/ITensorSetFilter.h"
#include "WireCellIface/ITerminal.h"
#include "larwirecell/Interfaces/IArtEventVisitor.h"

#include <map>
#include <string>
#include <vector>

namespace WireCell::AIML {

  class TensorSetLabeler : public Aux::Logger,
                           public wcls::IArtEventVisitor,
                           public ITensorSetFilter,
                           public IConfigurable,
                           public ITerminal {
  public:
    TensorSetLabeler();
    virtual ~TensorSetLabeler();

    // IArtEventVisitor
    void visit(art::Event& event) override;

    // ITensorSetFilter
    bool operator()(const input_pointer& in, output_pointer& out) override;

    // IConfigurable
    void configure(const WireCell::Configuration& config) override;
    WireCell::Configuration default_configuration() const override;

    // ITerminal
    void finalize() override;

  private:
    // One depo, already projected into WCT units.
    struct Depo {
      double x, y, z; // WCT length units
      double t;       // WCT time units
      int trackid;
      double weight;  // number of electrons (fallback: energy)
    };

    // Per (apa,face) geometry context for depo->(tick, wires) projection.
    struct FaceCtx {
      IAnodeFace::pointer face;
      double xw;   // collection (W) plane x, as BlobSampler::plane_x(2)
      int dirx;    // face normal sign
      double xmin, xmax; // sensitive x extent
    };

    // config
    std::string m_inpath{"pointtrees/%d"};
    std::string m_grouping{"live"};
    std::string m_truth_datapath{"truthtracks/%d"};
    std::string m_deposet_label{"ionandscint:priorSCE"};
    std::string m_mctruth_label{"generator"};
    std::string m_mcparticle_label{"largeant"};
    double m_drift_speed;      // set in ctor (units-dependent)
    double m_time_offset;      // ADDED to signal time, as BlobSampler
    double m_depo_time_offset{0.0}; // ADDED to depo times
    double m_tick;             // sampling period
    int m_wire_slop{1};        // accept depos this many wires outside blob bounds
    int m_tick_slop{2};        // accept depos this many ticks outside blob slice
    std::vector<IAnodePlane::pointer> m_anodes;
    std::map<std::pair<int, int>, FaceCtx> m_faces; // (apa,face) -> ctx

    // optional shared Bee sink for the truth_trackid debug dump
    Clus::IBeeSink::pointer m_bee_sink{nullptr};
    std::string m_bee_detector{"sbnd"};
    std::string m_bee_algorithm{"truth_trackid"};
    int m_bee_index{0};

    // per-event truth captured in visit()
    int m_run{0}, m_sub{0}, m_evt{0};
    WireCell::Configuration m_evtmd;         // nu_* metadata
    std::vector<std::vector<double>> m_tracks; // truth_per_track rows
    std::vector<Depo> m_depos;

    size_t m_count{0};
  };
}

#endif
