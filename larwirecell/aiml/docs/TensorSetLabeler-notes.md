# TensorSetLabeler — implementation summary + pitfalls

Notes from building/validating `wclsTensorSetLabeler` (2026-07, SBND
`wcls-img-clus-matching-xin` chain, MCP2025C corsika+GENIE rockbox MC).
Companion write-up with formulas and the stage-by-stage validation numbers:
`wcp-porting-validation sbnd/docs/depo-blob-smearing.md`; run recipes in
`sbnd/TensorSetLabeler/README.md` there.

# Section 1: what was implemented and tested

`wclsTensorSetLabeler` (this package, lib `WireCellAIML`; factory
`wclsTensorSetLabeler`, `wcls::IArtEventVisitor` + `ITensorSetFilter`) sits
between the all-APA MultiAlgBlobClustering and its TensorFileSink and
attaches truth to the clustering ITensorSet (a serialized PointCloud tree):

- **Event metadata**: `runNo/subRunNo/eventNo` + generator-MCTruth neutrino
  info added to the tensor-set metadata.  Because an event can hold several
  beam-nu interactions (rockbox), the neutrino fields are PARALLEL ARRAYS
  of length `n_nu`, one entry per interaction:
  `nu_idx/nu_pdg/nu_ccnc/nu_int_type/nu_energy/nu_vtx_{x,y,z}/nu_flavor/nu_edep`.
  Entry 0 is the "main" interaction; `nu_idx[k]` is the generator-MCTruth
  index (matches the `truth_per_track` nu_idx column and the mc-tree node id
  `9000000+nu_idx`).  `nu_edep` (GeV) is the DEPOSITED (visible) energy of
  the interaction — sum of `SimEnergyDeposit::Energy()` over deposits whose
  (abs) trackid descends from it, well below `nu_energy` (see 2.10).
- **`truth_per_track` tensor** `[ntracks x 23]` (columns in the tensor
  metadata; cm/ns/GeV): with `truth_tracks_nu_only` (default true) only
  BEAM-neutrino primaries (larreco CellTree "nuOnly" cut), all beam-nu
  interactions of the event kept and distinguished by the `nu_idx` column.
  The `process` column is the G4 creation-process code (CellTree
  convention; `Michel`=10001 synthetic, see 2.10).
- **Per-blob `trackid`** written into each live blob `scalar` PC (-1 = no
  match): BlobDepoFill-style association of `ionandscint:priorSCE`
  SimEnergyDeposits with the blob (slice tick, u/v/w wire) bounds in the
  raw coordinates, with (i) the TrueFwd SCE shift of the depos to reco
  positions, (ii) drift-diffusion (DL/DT) + SP-filter smearing making each
  depo a Gaussian ball (acceptance = bounds +- slop + nsigma*sigma), and
  (iii) `abs(TrackID)` folding dropped descendants (delta rays) into their
  saved ancestor.
- **Bee debug sets** into the shared mabc.zip: `truth_trackid_labeled`
  (labeled blob points, cluster_id = trackid), `truth_unlabeled`
  (unlabeled blob points, cluster_id = reco cluster ident),
  `truth_depo_sce` (the depo cloud at the drifted apparent positions,
  sampled from the diffusion balls, clipped to the readout window), and a
  `{i}-mc.json` jstree particle-flow tree: beam-nu-derived particles with
  KE > 10 MeV, nearest-kept-ancestor nesting, node id = trackid, GROUPED
  under one "initial mother neutrino" root node per beam-nu interaction
  (built from the generator MCTruth since largeant does not reliably save
  the initial neutrino; id = 9000000 + nu_idx, start = end = the
  interaction vertex, name from the MCTruth neutrino, energy = the
  interaction's DEPOSITED energy `Edep` [MeV], not the nu total energy) --
  rockbox events carry several interactions per event.
- **Michel "trackid merging"** (`bee_michel_merge`, default true): in the
  Bee `truth_trackid_labeled` and `truth_depo_sce` sets a Michel electron's
  cluster_id is replaced by its mother muon's trackid, so decay electrons
  render as part of the muon.  Bee display only — the blob `scalar` PC keeps
  the true (Michel) trackid.  The map is built over ALL largeant particles,
  independent of `pf_nu_only` (see 2.10).

**Tested** on 10 corsika+GENIE rockbox MC events (runs 32/31): blobs
labeled per event 97/74/59/80/95/88/88/76/84/91% (points labeled higher);
nearest-neighbor trackid coherence 100.0% on 8 events, 99.9%/99.6% on the
other two.  The unlabeled remainder is dominated by real ghosts, especially
isochronous tracks.  Per-event mc trees carry 1-3 mother-neutrino roots
(multi-interaction rockbox events group correctly).  Latest BEE set
(10 events, all truth sets + clustering):
https://www.phy.bnl.gov/twister/bee/set/90ee02b9-acba-4d8e-90c1-8a28df6fa240/event/list/

# Section 2: pitfalls and conventions

## 2.1 abs(TrackID): negative ids are DROPPED descendants  **(critical)**

`sim::SimEnergyDeposit::TrackID()` is NEGATIVE for deposits made by
particles that larg4 dropped from the MCParticle list (delta rays and
similar secondaries): the value is `-(saved ancestor trackid)`.  If you
treat `tid < 0` as "no label", every blob section where delta-ray charge
dominates comes out unlabeled — real muon tracks get a striped
labeled/unlabeled pattern (found via a track in evt 32/10/13 whose
delta-ray-rich stretch was entirely "unlabeled" while the depos were
plainly there).  Fold them into the parent with `abs(TrackID())` before
accumulating.  Effect was large: 4-event blob-label rates jumped
82/53/47/68% → 97/74/59/80% and NN-trackid coherence became exactly 100%.

## 2.2 Time/drift offset convention

Raw (non-t0-corrected) blob x is defined by BlobSampler::time2drift:
`x = x_W + dirx*(t_sig + time_offset)*drift_speed`, SBND
`time_offset = -205us` (= `sim.tick0_time`), `drift_speed = 1.563 mm/us`,
`tick = 0.5 us`.  The labeler MUST use the same values (threaded from the
same jsonnet locals as the BlobSampler).  Inverse map for a depo:
`x_app = x_true + dirx*v*t_dep`, `itick = ((x_app - x_W)*dirx/v + 205us)/tick`.
The WCT sim (ductor `start_time = tick0_time - response_plane/v` + Reframer
chop) makes an in-time depo reconstruct at its true x — no residual offset
was needed (`depo_time_offset = 0`).

## 2.3 SCE: priorSCE depos vs post-SCE blobs

Only `ionandscint:priorSCE` (TRUE positions) is persisted in reco1 files,
while blobs live at SCE-distorted positions — up to ~1.4 cm apart vs 3 mm
pitch.  The SBND dualmap ROOT file carries BOTH directions:
`TrueBkwd_*` = reco→true (what SCECorrection uses, `sign:+1`) and
`TrueFwd_*` = true→reco.  Use a second `SCEFieldTH3` instance with the
TrueFwd histogram names and `sign: 1` and ADD its displacement to the depo
true position.  Do not "invert" the Bkwd map — the Fwd map exists.
(evt-1 blob labeling 58% → 67%.)

## 2.4 Diffusion smearing: depos are balls, not points

Two smearings, in quadrature per depo (details + sources in the
depo-blob-smearing doc): drift diffusion `sigma = sqrt(2*D*t_drift)` with
DL=4.0 / DT=8.8 cm²/s (`wcsimsp_sbnd.fcl`), and SP-filter smearing
(time `1/(2*pi*0.10MHz)` = 3.18 ticks from `Gaus_wide`; wire
`1/(2*sqrt(pi)*k)` = 0.269 / 0.078 pitch from `Wire_ind` k=1.05 /
`Wire_col` k=3.60).  Acceptance = blob bounds ± (slop + nsigma*sigma).
Without this, ~1/4 of real blobs are missed at one-plane boundaries.
(evt-1: 67% → 82%.)

## 2.5 Wire indices: use the face RayGrid

Blob `scalar` u/v/w bounds are RayGrid strip indices.  Compute depo
wire-in-plane indices from the SAME machinery:
`face->raygrid()`, `pitch = (pos - centers()[layer]).dot(pitch_dirs()[layer])`,
`pitch_index(pitch, layer)` with layers 2/3/4 = u/v/w.  Do not assume the
Pimpos `region_binning()` index matches the strip convention.

## 2.6 Truth-particle conventions (trackid-offset scheme)

- WHY the 1e7/2e7 offsets: the SBND G4 stage runs TWO Geant4 instances per
  event (`larg4intime` / `larg4outtime`), each numbering tracks from 1;
  `MergeSimSources` (`mergesimsources_sbnd.fcl`:
  `InputSourcesLabels: [larg4intime, larg4outtime]`,
  `TrackIDOffsets: [10000000, 20000000]`) merges them into the single
  `largeant` products, adding the per-source offset to every track AND
  mother id.  So the id block encodes the SIMULATION INSTANCE (in-time vs
  out-of-time), NOT the generator — an in-time cosmic lands in the 1e7
  block next to the GENIE tracks; use the MCTruth Assns for provenance.
- A primary's `Mother()` is therefore the offset root (e.g. 10000000),
  NOT 0 — the larreco CellTree `Mother()==0` primary test finds nothing.
  Use `Process()=="primary"` (keep `Mother()==0` as a legacy OR).
- "Beam neutrino derived" = the largeant `MCParticle<->MCTruth` Assns
  MCTruth has `Origin()==simb::kBeamNeutrino` (CellTree "nuOnly").
  Do NOT compare the assns product id to the generator handle: rockbox
  `generator` products hold MULTIPLE beam-nu MCTruths (in-detector + dirt/
  rock), and Origin covers them all.  Record the assns `Ptr.key()` as a
  `nu_idx` column to distinguish interactions (0 = the MCTruth the `nu_*`
  metadata describes, i.e. `mclist.at(0)`).

## 2.7 Bee display gotchas

- Draw the depo cloud at the DRIFTED apparent position (`x_app`), not the
  true x, if it should overlay the raw-coordinate blobs.
- CLIP to the readout window (`nticks`): far-out-of-time depos
  (radiologicals at t up to 1e13 ns) otherwise fly to |x| ~ km.
- Shared `BeeSink`: every writer keeps its own per-event index starting at
  the same `initial_index` and increments once per event — all writers'
  indices stay aligned by event ordinal.
- Bee `Points::append` takes WCT-unit Points and divides by cm itself.

## 2.8 What the unlabeled remainder IS

After all of the above, unlabeled blobs are dominated by real ghosts —
especially isochronous tracks (verified visually on BEE and by checking
that no depo is consistent in >=2 of 3 wire views for ~2/3 of them; they
also carry about half the median charge of labeled blobs).

## 2.9 Build-system pitfalls (larwirecell MRB tree)

- `WireCellUtil/Bee.h` → `custard_boost.hpp` → `miniz.h`, which `wcb
  install` does NOT install: hand-copy
  `wire-cell-toolkit/util/inc/WireCellUtil/custard/miniz.h` into
  `<install>/include/WireCellUtil/custard/` (redo after a fresh install).
- gcc `-Werror` catches: don't bind `face->sensitive().bounds()` to a
  reference (dangling into a temporary BoundingBox); keep the constructor
  init list in declaration order (`-Werror=reorder`).
- After `make install`, hand-copy `libWireCellAIML.so` from `$MRB_INSTALL`
  to the runtime `opt/larwirecell/.../lib/` or the running chain silently
  uses the STALE library (a failed build + blind copy once produced
  "results" from old code — always check the build exit status first).
- TensorFileSink `dump_mode: true` DISCARDS the tensors (historical
  "trash" sink); the sbnd cfg flips it off when the labeler is in the
  pipeline so the labeled output is actually written.

## 2.10 process code, Michel electrons, and Edep

- **`process` column**: the G4 creation-process string (`MCParticle::Process()`)
  mapped to an int via the CellTree convention (Ningclover
  `TrackIDPIDMap2h5.cxx`: `primary`=0, `Decay`=1, `eIoni`=2, ... `dInelastic`=33;
  unknown → -1).  Keep the table in sync if that reference changes.
- **Michel = synthetic code 10001** (not a G4 process): a decay electron of a
  muon — `abs(pdg)==11` AND `Process()=="Decay"` AND `abs(mother_pdg)==13`.
  Applied both to the `process` column and to the Bee "trackid merging".
- **Michel merge is Bee-only and unconditional**: `bee_michel_merge` (default
  true) rewrites the Bee cluster_id of Michel-labeled points to the mother
  muon trackid; the blob `scalar` PC keeps the true (Michel) trackid.  The
  Michel→mother map is built over ALL largeant particles, so it is NOT
  filtered by `pf_nu_only` (which only prunes the mc tree).  For a COSMIC
  Michel with `pf_nu_only:true` the mother muon has no mc-tree node either
  way, so the merge is a pure point-view grouping; for a beam-nu Michel it
  folds the points under the muon node.
- **Edep** (`nu_edep`): sum of `SimEnergyDeposit::Energy()` (MeV in LArSoft)
  over deposits whose `abs(TrackID())` maps (via the MCParticle→MCTruth
  Assns) to a beam-nu interaction, accumulated PER interaction (`nu_idx`).
  It is computed in a dedicated pass over the SEDs placed BEFORE the mc-tree
  build (the tree's neutrino-node energy consumes it) and AFTER the
  MCParticle read (needs the tid→nu_idx map).  Metadata stores it in GeV
  (parallel to `nu_energy`, per interaction); the mc tree shows it in MeV.
  Edep ≪ `nu_energy` for NC / poorly-contained events (e.g. evt 32/10/6 NC:
  792 MeV total, 49.8 MeV deposited).
