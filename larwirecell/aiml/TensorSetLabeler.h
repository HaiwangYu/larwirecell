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
 * With "truth_tracks_nu_only" (default true) only BEAM-neutrino primaries
 * are saved (assns MCTruth Origin()==kBeamNeutrino && primary, the larreco
 * CellTree "nuOnly" cut) -- no cosmic-muon truth.  Rockbox events can carry
 * several beam-nu interactions: primaries of ALL of them are kept and the
 * "nu_idx" column records each row's MCTruth index (0 = the interaction
 * the nu_* metadata describes; -1 = non-beam rows in the full table).  The
 * "process" column is the G4 creation-process code (CellTree convention,
 * cf. TrackIDPIDMap2h5.cxx; -1 for unknown), with a synthetic "Michel"
 * code (10001) for a decay electron of a muon (pdg==e, process=="Decay",
 * mother pdg==mu).
 *
 * Blob level truth: for each blob node of the "live" grouping the dominant
 * G4 track id is written into the blob "scalar" PC as "trackid" (int, -1 if
 * no depo matched).  The trackid <- SimEnergyDeposit -> blob association
 * follows img/src/BlobDepoFill.cxx: each depo is converted to a (slice tick,
 * u/v/w wire-in-plane index) coordinate on its anode face and matched
 * against the blob's slice_index_[min,max) and {u,v,w}_wire_index_[min,max)
 * bounds; per-blob charge (NumElectrons) is accumulated per track id and the
 * argmax wins.  Deposits from dropped (unsaved) descendants carry the
 * NEGATIVE ancestor trackid in larg4 -- they are folded into the saved
 * ancestor via abs(TrackID) so delta-ray-dominated blob sections stay
 * labeled with their parent track.
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
 * SCE CORRECTION (default on when "sce_field" is set).  The blobs are
 * reconstructed from post-SCE (spatially distorted) charge while the
 * ionandscint:priorSCE depos are at TRUE positions -- up to ~1.4 cm apart,
 * i.e. several 3mm wire pitches.  When "sce_field" names an ISCEField
 * carrying the TrueFwd (true->reco) displacement map (a second SCEFieldTH3
 * on the SBND dualmap with th3_name_* = TrueFwd_Displacement_*, sign=1),
 * each depo is shifted x += dx(x,y,z) etc. before the association, making a
 * "postSCE" SimEnergyDeposit set on the fly.  "sce_correction" (default
 * true) gates the application without unwiring the component.
 *
 * Debug Bee output: when "bee_sink" names a Clus::IBeeSink, per event:
 *   - "truth_trackid_labeled": the "3d" points of LABELED blobs only, in
 *     raw coords with cluster_id = the blob's truth trackid.  With
 *     "bee_michel_merge" (default true) a Michel electron's cluster_id is
 *     replaced by its mother muon's trackid ("trackid merging") so decay
 *     electrons render as part of the muon -- Bee display only; the blob
 *     scalar PC keeps the true (Michel) trackid.  Also applied to
 *     "truth_depo_sce",
 *   - "truth_unlabeled": only the points of UNlabeled blobs (trackid<0),
 *     cluster_id = the reco cluster ident, to eyeball what fails to match,
 *   - "truth_depo_sce" (only when the SCE correction is applied): the
 *     SimEnergyDeposit points after the true->reco SCE shift, drawn at the
 *     DRIFTED apparent position that fills the blobs (x_app = x +
 *     dirx*drift_speed*t_dep at the post-SCE y,z), cluster_id = the truth
 *     trackid, q = the truth charge (NumElectrons).  Each depo is a
 *     diffusion "ball" (drift diffusion + SP filter smearing, see the
 *     header of the diffusion config block); "n_sample_truth_depo_sce"
 *     (default 1) points are Gaussian-sampled per ball (q split evenly),
 *   - "mc" (data/{i}/{i}-mc.json): a jstree particle-flow tree of the
 *     MCParticles with KE > "pf_ke_min" (default 10 MeV) and, when
 *     "pf_fiducial" names an IFiducial, an FV cut: start or end point
 *     inside the fiducial volume OR the start-end line section crossing
 *     it.  Particles derived from a beam neutrino (assns MCTruth
 *     Origin()==kBeamNeutrino) skip the FV cut.  With "pf_nu_only"
 *     (default true) ONLY beam-nu-derived particles enter the tree --
 *     no cosmics at all; false restores the FV-crossing cosmics.
 *     Children nest under their nearest KEPT ancestor via Mother()
 *     tracing; node id = G4 trackid (cross-references the truth_trackid
 *     cluster ids).  Interaction-level particles are GROUPED under a
 *     per-interaction "initial mother neutrino" node built from the
 *     generator MCTruth (id = 9000000 + nu_idx; start = end = the
 *     interaction vertex; name/KE from the MCTruth neutrino) -- rockbox
 *     events carry several beam-nu interactions per event.
 */

#ifndef LARWIRECELL_AIML_TENSORSETLABELER
#define LARWIRECELL_AIML_TENSORSETLABELER

#include "WireCellAux/Logger.h"
#include "WireCellClus/IBeeSink.h"
#include "WireCellIface/IAnodePlane.h"
#include "WireCellIface/IFiducial.h"
#include "WireCellIface/ISCEField.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/ITensorSetFilter.h"
#include "WireCellIface/ITerminal.h"
#include "larwirecell/Interfaces/IArtEventVisitor.h"

#include <map>
#include <random>
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
      double pitch[3];   // raygrid pitch magnitude per plane (u,v,w)
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
    int m_nticks{3400};        // readout ticks (clips the depo Bee display)
    // Diffusion of the (point-like) SimEnergyDeposits before blob filling:
    // (1) drift diffusion sigma = sqrt(2*D*t_drift) with DL/DT from the
    //     detsim (sbndcode wcsimsp_sbnd.fcl), longitudinal (-> time) and
    //     transverse (-> wire pitch) w.r.t. the drift;
    // (2) signal-processing filter smearing (dunereco
    //     docs/smear-dnn-campaign.md): time sigma = 1/(2*pi*f) with f the
    //     Gaus_wide HfFilter sigma, wire sigma = 1/(2*sqrt(pi)*k) [pitch]
    //     with k from Wire_ind/Wire_col (sbnd sp-filters.jsonnet).
    // Both are added in quadrature per depo; a blob accepts the depo when
    // its center is within (slop + nsigma*sigma) of the blob bounds.
    double m_DL;               // longitudinal diffusion [area/time]
    double m_DT;               // transverse diffusion [area/time]
    double m_sp_smear_time;    // SP time smearing sigma [time]
    double m_sp_smear_wire_ind{0.26875}; // SP wire smearing, U/V [pitch units]
    double m_sp_smear_wire_col{0.07839}; // SP wire smearing, W [pitch units]
    double m_nsigma{3.0};      // Gaussian acceptance half-width
    int m_nsample_depo{1};     // truth_depo_sce Bee: samples per depo ball
    bool m_sce_correction{true};   // apply true->reco SCE shift to depos
    bool m_truth_tracks_nu_only{true}; // truth_per_track: only nu-origin particles
    bool m_pf_nu_only{true};           // "mc" tree: only beam-nu-derived particles
    bool m_bee_michel_merge{true};     // Bee: merge Michel e- cluster_id into mother muon
    double m_pf_ke_min;            // KE cut for the Bee "mc" particle tree
    ISCEField::pointer m_sce{nullptr}; // TrueFwd (true->reco) displacement map
    IFiducial::pointer m_pf_fiducial{nullptr}; // FV cut for the "mc" tree
    std::vector<IAnodePlane::pointer> m_anodes;
    std::map<std::pair<int, int>, FaceCtx> m_faces; // (apa,face) -> ctx

    // optional shared Bee sink for the truth_trackid debug dump
    Clus::IBeeSink::pointer m_bee_sink{nullptr};
    std::string m_bee_detector{"sbnd"};
    std::string m_bee_algorithm{"truth_trackid_labeled"};
    std::string m_bee_unlabeled_algorithm{"truth_unlabeled"};
    std::string m_bee_depo_algorithm{"truth_depo_sce"};
    std::string m_bee_pf_name{"mc"};
    int m_bee_index{0};

    // per-event truth captured in visit()
    int m_run{0}, m_sub{0}, m_evt{0};
    WireCell::Configuration m_evtmd;         // nu_* metadata
    std::vector<std::vector<double>> m_tracks; // truth_per_track rows
    std::vector<Depo> m_depos;
    std::map<int, int> m_michel_mother;      // Michel e- trackid -> mother muon trackid
    WireCell::Configuration m_pf_particles;  // Bee "mc" jstree node array

    size_t m_count{0};
    std::mt19937 m_rng{20260708}; // fixed seed: deterministic depo-ball sampling
  };
}

#endif
