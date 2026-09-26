/**
 * @file   larwirecell/Components/OpHitSource.h
 * @brief  A WCT component reading recob::OpHit from the art::Event and emitting
 *         the "ophits" ITensorSet the toolkit flash finders consume.
 * @date   2026-09-26
 *
 * WHY THIS EXISTS
 *
 * SBND's standalone Wire-Cell chain rebuilds its optical flashes from the reco1
 * PMT OpHits (SBNDReco1OpHitSource -> Flash::SBNDOpFlashFinder, wcp-porting-img
 * doc sbnd_xin/123) instead of taking SBND's recob::OpFlash, and that is the
 * production light since 2026-09-25.  The LArSoft 1-step chain reads
 * recob::OpFlash through wclsOpFlashSource, so the two chains ran different
 * light.  This component is the art-side twin of SBNDReco1OpHitSource (the
 * bare-ROOT reader of wire-cell-sbnd-reco1, which cannot be loaded inside a
 * LArSoft process because of its mirror dictionaries): same tensor, same
 * schema, same metadata, read through art.  Wire it as
 *   wclsOpHitSource:tpc<N> -> SBNDOpFlashFinder:tpc<N> -> FlashTensorToOpticalPCs port 1
 * in place of wclsOpFlashSource:tpc<N>.
 *
 * OUTPUT, one ITensorSet per event (ident = event number):
 *   - tensor "ophits": f8 [nhit, 9], one row per kept hit
 *       0 channel (OpChannel = OpDet index for SBND)
 *       1 hit time   [WCT ns]  -- see "hit_time"
 *       2 width      [WCT ns]
 *       3 area       [as stored]
 *       4 amplitude  [as stored]
 *       5 PE
 *       6 start time [WCT ns]
 *       7 flash id   (-1; the flash finder assigns it)
 *       8 fast_to_total
 *     Times are recob::OpHit trigger-relative us x units::microsecond.
 *   - set metadata: "run", "subrun", "event" and "frame_apply_at_caf" (ns, the
 *     sbnd::timing::FrameShiftInfo::FrameApplyAtCaf() of the "frame_label"
 *     product, 0 when absent -- exactly as wclsOpFlashSource stamps it; the
 *     flash finder passes the set metadata through, so FlashTensorToOpticalPCs
 *     applies the same offset to the rebuilt flashes).
 *
 * CONFIG
 *   art_tag     : the recob::OpHit product ("ophitpmt" in SBND reco1)
 *   channels    : list of OpChannels to keep (e.g. the 60 PMTs of one TPC,
 *                 sbnd-pmt-channels.json); empty = all
 *   hit_time    : "rise" = StartTime + RiseTime (the SBNDFlashFinder /
 *                 SimpleFlashAlgo "RiseTime" convention, the default),
 *                 "peak" = PeakTime, "start" = StartTime
 *   frame_label : FrameShiftInfo product label ("frameshift")
 *   debug_frame : print when the FrameShiftInfo product is absent
 */
#ifndef LARWIRECELL_COMPONENTS_OPHITSOURCE
#define LARWIRECELL_COMPONENTS_OPHITSOURCE

#include "WireCellAux/Logger.h"
#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/ITensorSetSource.h"
#include "larwirecell/Interfaces/IArtEventVisitor.h"

#include "canvas/Utilities/InputTag.h"

#include <deque>
#include <set>
#include <string>

namespace wcls {

  class OpHitSource : public WireCell::Aux::Logger,
                      public IArtEventVisitor,
                      public WireCell::ITensorSetSource,
                      public WireCell::IConfigurable {
  public:
    OpHitSource();
    virtual ~OpHitSource();

    /// IArtEventVisitor
    virtual void visit(art::Event& event);

    /// ITensorSetSource
    virtual bool operator()(WireCell::ITensorSet::pointer& ts);

    /// IConfigurable
    virtual WireCell::Configuration default_configuration() const;
    virtual void configure(const WireCell::Configuration& config);

  private:
    size_t m_count{0};
    std::deque<WireCell::ITensorSet::pointer> m_tensorsets;

    art::InputTag m_inputTag{"ophitpmt"};
    std::set<int> m_channels;         // empty = keep all
    std::string m_hit_time{"rise"};   // rise | peak | start
    std::string m_frame_label{"frameshift"};
    bool m_debug_frame{false};
  };
}

#endif
