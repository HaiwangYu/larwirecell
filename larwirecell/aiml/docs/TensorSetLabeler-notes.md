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
  info (`nu_pdg/nu_ccnc/nu_int_type/nu_energy/nu_vtx_{x,y,z}/nu_flavor`)
  added to the tensor-set metadata.
- **`truth_per_track` tensor** `[ntracks x 22]` (columns in the tensor
  metadata; cm/ns/GeV): with `truth_tracks_nu_only` (default true) only
  BEAM-neutrino primaries (larreco CellTree "nuOnly" cut), all beam-nu
  interactions of the event kept and distinguished by the `nu_idx` column.
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
  `{i}-mc.json` jstree particle-flow tree (beam-nu-derived particles with
  KE > 10 MeV, nearest-kept-ancestor nesting, node id = trackid).

**Tested** on 10 corsika+GENIE rockbox MC events (runs 32/31): blobs
labeled per event 97/74/59/80/95/88/88/76/84/91% (points labeled higher);
nearest-neighbor trackid coherence 100.0% on 8 events, 99.9%/99.6% on the
other two.  The unlabeled remainder is dominated by real ghosts, especially
isochronous tracks.  Latest BEE set (10 events, all truth sets + clustering):
https://www.phy.bnl.gov/twister/bee/set/50110108-50fd-4288-aa59-ca2da05e7fc2/event/list/

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

- GENIE particles carry trackids offset by 1e7, corsika by 2e7; a GENIE
  primary's `Mother()` is the offset root (e.g. 10000000), NOT 0 — the
  larreco CellTree `Mother()==0` primary test finds nothing.  Use
  `Process()=="primary"` (keep `Mother()==0` as a legacy OR).
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
