#include "OpHitSource.h"

#include "WireCellAux/SimpleTensor.h"
#include "WireCellAux/SimpleTensorSet.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/Units.h"

#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "lardataobj/RecoBase/OpHit.h"

// sbnd::timing::FrameShiftInfo (sbnobj) is only available in recent sbnobj
// versions.  Guard the include so older builds still compile -- when absent,
// the per-event frame_apply_at_caf shift falls back to 0 (no shift), the same
// fallback wclsOpFlashSource has.
#if __has_include("sbnobj/SBND/Timing/FrameShiftInfo.hh")
#define HAVE_SBND_FRAMESHIFTINFO 1
#include "sbnobj/SBND/Timing/FrameShiftInfo.hh"
#endif

#include <iostream>
#include <memory>
#include <vector>

WIRECELL_FACTORY(wclsOpHitSource,
                 wcls::OpHitSource,
                 wcls::IArtEventVisitor,
                 WireCell::ITensorSetSource,
                 WireCell::IConfigurable)

using namespace wcls;
using namespace WireCell;
using WireCell::Aux::SimpleTensor;
using WireCell::Aux::SimpleTensorSet;

OpHitSource::OpHitSource() : Aux::Logger("OpHitSource", "op") {}
OpHitSource::~OpHitSource() {}

WireCell::Configuration OpHitSource::default_configuration() const
{
  Configuration cfg;
  cfg["art_tag"] = m_inputTag.encode();
  cfg["channels"] = Json::arrayValue;
  cfg["hit_time"] = m_hit_time;
  cfg["frame_label"] = m_frame_label;
  cfg["debug_frame"] = m_debug_frame;
  return cfg;
}

void OpHitSource::configure(const WireCell::Configuration& cfg)
{
  if (cfg.isMember("art_tag")) {
    const std::string art_tag = cfg["art_tag"].asString();
    if (art_tag.empty()) {
      THROW(ValueError() << errmsg{"wclsOpHitSource requires a non-empty art_tag"});
    }
    m_inputTag = art_tag;
  }
  m_channels.clear();
  if (cfg.isMember("channels")) {
    for (const auto& jch : cfg["channels"]) m_channels.insert(jch.asInt());
  }
  if (cfg.isMember("hit_time")) m_hit_time = cfg["hit_time"].asString();
  if (m_hit_time != "rise" and m_hit_time != "peak" and m_hit_time != "start") {
    THROW(ValueError() << errmsg{"wclsOpHitSource: bad hit_time \"" + m_hit_time +
                                 "\" (rise | peak | start)"});
  }
  if (cfg.isMember("frame_label")) m_frame_label = cfg["frame_label"].asString();
  if (cfg.isMember("debug_frame")) m_debug_frame = cfg["debug_frame"].asBool();
  log->debug("art_tag {} channels {} hit_time {} frame_label {}",
             m_inputTag.encode(), m_channels.size(), m_hit_time, m_frame_label);
}

void OpHitSource::visit(art::Event& event)
{
  art::Handle<std::vector<recob::OpHit>> ophits;
  event.getByLabel(m_inputTag, ophits);
  if (!ophits.isValid()) {
    THROW(ValueError() << errmsg{"wclsOpHitSource failed to get recob::OpHit product " +
                                 m_inputTag.encode()});
  }

  // Same row layout and units as SBNDReco1OpHitSource (wire-cell-sbnd-reco1),
  // so Flash::SBNDOpFlashFinder sees the identical input inside LArSoft.
  const double us = units::microsecond;
  std::vector<double> rows;
  rows.reserve(ophits->size() * 9);
  for (const auto& h : *ophits) {
    if (!m_channels.empty() and !m_channels.count(h.OpChannel())) continue;
    double t = h.StartTime() + h.RiseTime();
    if (m_hit_time == "peak") t = h.PeakTime();
    else if (m_hit_time == "start") t = h.StartTime();
    rows.insert(rows.end(), {double(h.OpChannel()), t * us, h.Width() * us, h.Area(),
                             h.Amplitude(), h.PE(), h.StartTime() * us, -1.0, h.FastToTotal()});
  }
  const size_t nhit = rows.size() / 9;

  ITensor::vector* itv = new ITensor::vector;
  {
    Configuration md;
    md["name"] = "ophits";
    itv->push_back(std::make_shared<SimpleTensor>(ITensor::shape_t{nhit, (size_t)9},
                                                  rows.data(), md));
  }

  // Forward FrameShiftInfo::FrameApplyAtCaf() (ns) as "frame_apply_at_caf",
  // exactly as wclsOpFlashSource does for the reco1 flashes: 0 when the product
  // is absent (MC) or the sbnobj header was not available at compile time.
  double frame_apply_at_caf = 0.0;
#ifdef HAVE_SBND_FRAMESHIFTINFO
  {
    art::Handle<sbnd::timing::FrameShiftInfo> frameHandle;
    event.getByLabel(m_frame_label, frameHandle);
    if (!frameHandle.isValid()) {
      if (m_debug_frame) std::cout << "wclsOpHitSource: no FrameShift product found." << std::endl;
    }
    else {
      frame_apply_at_caf = frameHandle->FrameApplyAtCaf();
    }
  }
#endif

  Configuration set_md;
  set_md["run"] = event.run();
  set_md["subrun"] = event.subRun();
  set_md["event"] = event.event();
  set_md["frame_apply_at_caf"] = frame_apply_at_caf;

  auto tset = std::make_shared<SimpleTensorSet>(event.event(), set_md, ITensor::shared_vector(itv));
  log->debug("run {} subrun {} event {}: emit {} of {} hits from {} (hit_time {}, frame_apply_at_caf {} ns)",
             event.run(), event.subRun(), event.event(), nhit, ophits->size(),
             m_inputTag.encode(), m_hit_time, frame_apply_at_caf);
  m_tensorsets.push_back(tset);
  m_tensorsets.push_back(nullptr);
}

bool OpHitSource::operator()(WireCell::ITensorSet::pointer& tensorset)
{
  tensorset = nullptr;
  if (m_tensorsets.empty()) {
    log->debug("EOS at call {}", m_count++);
    return false;
  }
  tensorset = m_tensorsets.front();
  m_tensorsets.pop_front();
  m_count++;
  return true;
}
