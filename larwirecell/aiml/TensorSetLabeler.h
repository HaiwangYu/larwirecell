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
 *   - n_nu: number of beam-neutrino interactions in the event.
 *   - nu_idx, nu_pdg, nu_ccnc, nu_int_type, nu_energy (GeV),
 *     nu_vtx_{x,y,z} (cm), nu_flavor ("nue"/"numu"/"nc"/"none"), nu_edep
 *     (GeV): PARALLEL ARRAYS of length n_nu, one entry per interaction
 *     (rockbox events carry several beam-nu interactions -- in-detector
 *     plus dirt/rock).  Entry 0 is the "main" interaction.  nu_idx is the
 *     generator-MCTruth index (matches the truth_per_track "nu_idx" column
 *     and the mc-tree node id 9000000+nu_idx).  nu_* are from the generator
 *     MCTruth (cf. Ningclover larwirecell/aiml/Truth2h5.cxx).  nu_edep is
 *     the DEPOSITED (visible) energy of the interaction: sum of
 *     sim::SimEnergyDeposit::Energy() over deposits whose (abs) trackid
 *     descends from it -- well below nu_energy.
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
 * PSEUDO-SIMULATION of the SimEnergyDeposits.  To make priorSCE (true) depos
 * comparable to reconstructed blobs the labeler applies a chain of
 * "pseudo-sim" effects, each optional/configurable:
 *   1. SCE      -- shift the true position true->reco (TrueFwd map, below).
 *   2. drift    -- apparent x from the deposit time:
 *                    x_app = x + dirx*drift_speed*(t_dep + depo_time_offset)
 *                  (a later deposit reconstructs deeper into the volume).
 *   3. smear    -- each depo is a Gaussian "ball": drift diffusion (DL/DT)
 *                  plus the SP-filter smearing (below); "ball sampling"
 *                  draws n_sample_truth_depo_sce points from it.
 *   4. readout  -- keep the depo only if its pseudo-sim TIME is inside the
 *                  readout window [readout_time_min, readout_time_max].
 *
 * TIME CONVENTION (be careful -- consistent with Facade time2drift/drift2time
 * and cfg/pgrapher/experiment/sbnd/{params,clus}.jsonnet).  The raw (non-t0-
 * corrected) blob x is x = x_W + dirx*(t_sig + time_offset)*drift_speed, so
 *   signal (slice) time: t_sig = (x_app - x_W)*dirx/drift_speed - time_offset
 *   slice tick:          itick = t_sig/tick    (tick = 0.5us)
 * "time_offset" MUST equal the BlobSampler's = sim.tick0_time = -205us for
 * SBND -- the trigger-frame time that the lower edge of readout tick 0
 * corresponds to (params.jsonnet).  Define the PSEUDO-SIM TIME (a.k.a.
 * pseudo-sim-t) as the trigger-frame time of the slice:
 *   pseudo_t = (x_app - x_W)*dirx/drift_speed  =  t_sig + time_offset
 * so pseudo_t = time_offset at tick 0.  The SBND readout is nticks=3427
 * ticks, giving the DEFAULT readout window
 *   [time_offset, time_offset + nticks*tick] = [-205us, 1508.5us].
 * (drift_speed = 1.563 mm/us; "depo_time_offset", default 0, absorbs any
 * residual sim-chain shift.)  We use the non-t0-corrected raw coordinates
 * and the priorSCE (true position) depos throughout.
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
 *     scalar PC keeps the true (Michel) trackid.  Also applied to the two
 *     SED pseudo-sim sets below,
 *   - "truth_unlabeled": only the points of UNlabeled blobs (trackid<0),
 *     cluster_id = the reco cluster ident, to eyeball what fails to match,
 *   - "sed-sce_drift_smear_readout" (only when the SCE correction is
 *     applied): the SimEnergyDeposit pseudo-sim cloud with ALL FOUR effects
 *     (SCE + drift + smear + readout) -- drawn at the DRIFTED apparent
 *     position that fills the blobs (x_app at the post-SCE y,z), ball-
 *     sampled, and cut to the readout window.  cluster_id = the truth
 *     trackid (Michel-merged, see below), q = the truth charge; this is
 *     the set that should overlay the reconstructed blobs,
 *   - "sed-smear_readout": the same depos with ONLY smear + readout -- at
 *     the TRUE position (no SCE, no drift shift), ball-sampled with the
 *     diffusion sigma of the true drift distance, cut to the readout
 *     window on the (undrifted) pseudo-sim time.  Comparing the two sets
 *     visualizes the SCE + drift displacement.  "n_sample_truth_depo_sce"
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
 *     interaction vertex; name from the MCTruth neutrino, energy = the
 *     interaction's DEPOSITED energy Edep [MeV], not the neutrino total
 *     energy) -- rockbox events carry several beam-nu interactions/event.
 *
 * HDF5 OUTPUT (nugraph, "hdf5_output" default true).  A third output (besides
 * the ITensorSet and the Bee points): a heterogeneous graph HDF5 for nugraph
 * training/testing, accumulated over events and written at finalize() to
 * "hdf5_filename" as a pynuml H5DataModule container:
 *   /planes ["u","v","y"], /semantic_classes ["nu","cosmic"], /gen, /datasize
 *   [ntrain,nval,ntest], /samples/{train,val,test}, and one scalar COMPOUND
 *   record per event at /dataset/<sample_name> whose slash-named fields are:
 *   - sp (3D nodes = blobs): sp/pos [N,3] mm, sp/features [N,6]
 *     (charge, reco_cluster_id, vtx_dist, vtx_dx, vtx_dy, vtx_dz),
 *     sp/y_semantic {0 nu,1 cosmic,-1 ghost}, sp/y_instance = trackid (-1),
 *     sp/raw_vtx_dist, and supervision edges sp/edge_label_index [2,E],
 *     sp/edge_y (same-trackid), sp/edge_labelable (both non-ghost).
 *   - u,v,y (2D nodes = per-plane merged wire measurements from the grouping
 *     ctpc_a*f*p{U,V,W} PCs, grouped in drift then split on pitch gaps):
 *     {p}/pos [M,2] mm, {p}/x [M,15] (charge,charge_err,nhits,pitch_min,
 *     pitch_max, vtx_dist/dx/dy/dz, then 6 sidecar zeros), {p}/id,
 *     {p}/y_semantic, {p}/y_instance.
 *   - edges: sp_nexus_sp (blob-blob).  Attempts the WCT "ctpc" graph flavor
 *     via Facade find_graph with detector_volumes + pc_transforms, but that
 *     needs clustering-time internal maps that as_pctree() does not rebuild
 *     (it throws map::at on the restored tree), so it falls back to an
 *     intra-cluster blob-center kNN graph (also used if dv/pcts are unset).
 *     And {p}_nexus_sp (2D-hit -> blob, from the
 *     TRUE wire/slice-box overlap our labeler already uses -- not the
 *     reference's approximate corner projection).  The intra-plane
 *     {p}_plane_{p} edges are NOT produced here (added in post-processing).
 *   - evt/num_nodes, evt/y (event has a beam nu), metadata/run,subrun,event.
 * The truth (y_semantic / y_instance) comes from the exact SED->blob trackid
 * labeling, so it is exact rather than the point-distance approximation of
 * the reference pywcml/converter.py.  NOTE: the 2D nodes only cover anode/
 * faces whose ctpc_* PC reaches this all-APA grouping (the joint QLMatching's
 * root_pcs_to_merge in the premerged xin chain); extend that list to cover
 * every TPC.
 *
 * DATA MODE ("reality" = "data").  With no MC truth: visit() keeps only
 * run/subrun/event; operator() writes ONLY the RSE into the ITensorSet
 * metadata (empty nu_* arrays, no truth_per_track tensor), emits NO Bee sets
 * (they are all truth-derived), and -- if hdf5_output -- writes an
 * INPUT-ONLY HDF5 graph: the same node/edge schema with reco features
 * (charge, reco_cluster_id, ctpc hit features, geometry edges) but the truth
 * fields set to sentinels (y_semantic = -1, y_instance = -1, vtx_* = -1/0,
 * edge_y/edge_labelable = 0).  Usable to run inference on real data and
 * judge correctness by human hand-scan.  The pseudo-sim knobs are irrelevant
 * in data mode (no depos are read).
 */

#ifndef LARWIRECELL_AIML_TENSORSETLABELER
#define LARWIRECELL_AIML_TENSORSETLABELER

#include "WireCellAux/Logger.h"
#include "WireCellClus/IBeeSink.h"
#include "WireCellUtil/Units.h"
#include "WireCellClus/IPCTransform.h"
#include "WireCellIface/IAnodePlane.h"
#include "WireCellIface/IDetectorVolumes.h"
#include "WireCellIface/IFiducial.h"
#include "WireCellIface/ISCEField.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/ITensorSetFilter.h"
#include "WireCellIface/ITerminal.h"
#include "larwirecell/Interfaces/IArtEventVisitor.h"

#include <map>
#include <random>
#include <set>
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
    // Write the accumulated m_events to m_hdf5_filename as a pynuml
    // H5DataModule container (called from finalize()).
    void write_hdf5();

    // One field of a per-event nugraph HDF5 compound record.  Stored as a
    // flat buffer + dims; written as a compound member (name may contain '/',
    // e.g. "sp/pos", "u_nexus_sp/edge_index") in finalize().
    struct H5Member {
      std::string name;
      bool is_float;                       // true: float32, false: int64
      std::vector<unsigned long long> dims; // empty = scalar
      std::vector<float> f;
      std::vector<long long> i;
    };
    // One event's heterogeneous graph, ready to serialize.
    struct EventGraph {
      std::string sample_name;
      std::vector<H5Member> members;
    };

    // One depo, already projected into WCT units.
    struct Depo {
      double x, y, z;    // WCT length units, post-SCE (true->reco) if applied
      double x0, y0, z0; // WCT length units, TRUE (pre-SCE) position
      double t;          // WCT time units
      int trackid;
      double weight;     // number of electrons (fallback: energy) -- blob labeling
      double q;          // sed.NumElectrons() -- per-point charge for the sed sets
      double e;          // sed.Energy() [MeV] -- per-point energy for the sed sets
      int nu_idx;        // 0 = non-neutrino, 1,2,... = beam-nu interaction (1-based)
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
    // "sim": full truth outputs (ITensorSet nu metadata + truth_per_track,
    // Bee truth sets, HDF5 with truth).  "data": no truth -- only RSE in the
    // ITensorSet metadata and (optionally) an input-only HDF5 graph (nodes +
    // edges, truth fields set to sentinels).  The labeler's pseudo-sim knobs
    // (sce_field/sce_correction/...) are independent of this.
    std::string m_reality{"sim"};
    double m_drift_speed;      // set in ctor (units-dependent)
    double m_time_offset;      // ADDED to signal time, as BlobSampler
    double m_depo_time_offset{0.0}; // ADDED to depo times
    double m_tick;             // sampling period
    // "Readout" pseudo-sim cut (see PSEUDO-SIM section in the class header):
    // keep a depo only if its pseudo-sim time
    //   pseudo_t = (x_app - x_W)*dirx/drift_speed  ( = t_sig + time_offset )
    // lies in [m_readout_tmin, m_readout_tmax].  This is the trigger-frame
    // time of the readout slice; tick 0 is at time_offset = tick0_time.  The
    // SBND default [-205us, 1508.5us] = [tick0_time, tick0_time+nticks*tick]
    // (nticks=3427) is the full readout window.  Set in ctor (units).
    double m_readout_tmin;     // readout window lower edge (pseudo-sim time)
    double m_readout_tmax;     // readout window upper edge (pseudo-sim time)
    int m_wire_slop{1};        // accept depos this many wires outside blob bounds
    int m_tick_slop{2};        // accept depos this many ticks outside blob slice
    int m_nticks{3400};        // legacy (superseded by the readout_time_* cut)
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
    int m_nsample_depo{1};     // SED pseudo-sim Bee sets: samples per depo ball
    bool m_sce_correction{true};   // apply true->reco SCE shift to depos
    bool m_truth_tracks_nu_only{true}; // truth_per_track: only nu-origin particles
    bool m_pf_nu_only{true};           // "mc" tree: only beam-nu-derived particles
    bool m_bee_michel_merge{true};     // Bee: merge Michel e- cluster_id into mother muon
    double m_pf_ke_min;            // KE cut for the Bee "mc" particle tree
    // nugraph HDF5 output (heterogeneous graph for training/testing; see the
    // HDF5 OUTPUT section in the class header).  Accumulated per event and
    // written at finalize() as a pynuml H5DataModule container.
    bool m_hdf5_output{true};      // write the nugraph .h5 (default on)
    std::string m_hdf5_filename{"nugraph.h5"};
    int m_plane_knn{6};            // sp-sp kNN fallback (see .cxx)
    // detector geometry for the "ctpc" blob-blob graph flavor (sp_nexus_sp)
    IDetectorVolumes::pointer m_dv{nullptr};
    WireCell::Clus::IPCTransformSet::pointer m_pcts{nullptr};
    double m_ctpc_x_tol;           // 2D-node drift grouping tol (set in ctor)
    double m_ctpc_pitch_gap;       // 2D-node pitch-gap split tol (set in ctor)
    ISCEField::pointer m_sce{nullptr}; // TrueFwd (true->reco) displacement map
    IFiducial::pointer m_pf_fiducial{nullptr}; // FV cut for the "mc" tree
    std::vector<IAnodePlane::pointer> m_anodes;
    std::map<std::pair<int, int>, FaceCtx> m_faces; // (apa,face) -> ctx

    // optional shared Bee sink for the truth_trackid debug dump
    Clus::IBeeSink::pointer m_bee_sink{nullptr};
    std::string m_bee_detector{"sbnd"};
    // Coordinate array names (in each blob's "3d" PC) for the tagger_stm/tgm/fc
    // Bee sets, so they overlay clustering_global.  Set from the entry config to
    // the same corrected scope clustering_global uses: data ['x_t0cor','y_cor',
    // 'z_cor'], sim ['x_sce','y_sce','z_sce'].  Empty -> fall back to raw x,y,z.
    std::vector<std::string> m_tagger_coords{};
    // Beam gate (internal units) for the tagger Bee cluster_id encoding: a
    // main_cluster is a "beam-window candidate" (the taggers only evaluate these)
    // when its cluster_t0 is in [low, high).  MUST match the tagger's beam_window
    // (run_nusel BEAM_WINDOW="0.2,2.2" us).  Overridable via cfg "beam_window".
    double m_beam_window_low{0.2 * WireCell::units::us};
    double m_beam_window_high{2.2 * WireCell::units::us};
    std::string m_bee_algorithm{"truth_trackid_labeled"};
    std::string m_bee_unlabeled_algorithm{"truth_unlabeled"};
    // SED pseudo-sim clouds (see PSEUDO-SIM in the class header):
    std::string m_bee_depo_algorithm{"sed-sce_drift_smear_readout"}; // all 4 effects
    std::string m_bee_sr_algorithm{"sed-smear_readout"};             // smear + readout only
    std::string m_bee_ssr_algorithm{"sed-sce_smear_readout"};        // SCE + smear + readout (no drift shift)
    std::string m_bee_pf_name{"mc"};
    int m_bee_index{0};

    // per-event truth captured in visit()
    int m_run{0}, m_sub{0}, m_evt{0};
    WireCell::Configuration m_evtmd;         // nu_* metadata
    std::vector<std::vector<double>> m_tracks; // truth_per_track rows
    std::vector<Depo> m_depos;
    std::map<int, int> m_michel_mother;      // Michel e- trackid -> mother muon trackid
    std::map<int, double> m_nu_edep;         // nu_idx -> sum SED Energy() [MeV] of that interaction
    std::set<int> m_nu_trackids;             // beam-nu-derived trackids (>=0), for node semantics
    WireCell::Configuration m_pf_particles;  // Bee "mc" jstree node array
    std::vector<EventGraph> m_events;        // accumulated nugraph records (written at finalize)

    size_t m_count{0};
    std::mt19937 m_rng{20260708}; // fixed seed: deterministic depo-ball sampling
  };
}

#endif
