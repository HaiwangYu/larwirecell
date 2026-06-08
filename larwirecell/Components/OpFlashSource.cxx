#include "OpFlashSource.h"
#include "TTimeStamp.h"
#include "WireCellAux/SimpleTensor.h"
#include "WireCellAux/SimpleTensorSet.h"
#include "WireCellUtil/Logging.h"
#include "WireCellUtil/NamedFactory.h"
#include "WireCellUtil/String.h"
#include "WireCellUtil/Units.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "lardataobj/RecoBase/OpFlash.h"

// sbnd::timing::FrameShiftInfo (sbnobj) is only available in recent sbnobj
// versions.  Guard the include so older builds still compile -- when absent,
// the per-event frame_apply_at_caf shift falls back to 0 (no shift).
#if __has_include("sbnobj/SBND/Timing/FrameShiftInfo.hh")
#define HAVE_SBND_FRAMESHIFTINFO 1
#include "sbnobj/SBND/Timing/FrameShiftInfo.hh"
#endif

#include <boost/multi_array.hpp>

#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

WIRECELL_FACTORY(wclsOpFlashSource,
                 wcls::OpFlashSource,
                 wcls::IArtEventVisitor,
                 WireCell::ITensorSetSource)

using namespace wcls;
using namespace WireCell;
using WireCell::Aux::SimpleTensor;
using WireCell::Aux::SimpleTensorSet;
using WireCell::String::format;

OpFlashSource::OpFlashSource() : Aux::Logger("OpFlashSource", "op") {}

OpFlashSource::~OpFlashSource() {}

WireCell::Configuration OpFlashSource::default_configuration() const
{
  Configuration cfg;
  cfg["art_tag"] = ""; // how to look up the opflashes
  cfg["frame_label"] = fFrameLabel;
  cfg["debug_frame"] = fDebugFrame;
  return cfg;
}

void OpFlashSource::configure(const WireCell::Configuration& cfg)
{
  const std::string art_tag = cfg["art_tag"].asString();
  if (art_tag.empty()) {
    THROW(ValueError() << errmsg{"WireCell::OpFlashSource requires a source_label"});
  }
  m_inputTag = cfg["art_tag"].asString();
  if (cfg.isMember("frame_label")) fFrameLabel = cfg["frame_label"].asString();
  if (cfg.isMember("debug_frame")) fDebugFrame = cfg["debug_frame"].asBool();
}

void OpFlashSource::visit(art::Event& event)
{
  log->debug("OpFlashSource::visit event{} input tag {}", event.event(), m_inputTag.encode());
  art::Handle<std::vector<recob::OpFlash>> opflashes;
  event.getByLabel(m_inputTag, opflashes);
  if (!opflashes.isValid()) {
    THROW(ValueError() << errmsg{"WireCell::OpFlashSource failed to get opflashes"});
  }
  for (auto const& opflash : *opflashes) {
    log->debug("OpFlash time: {} PEs: {}", opflash.Time(), opflash.PEs().size());
  }

  const auto nflashes = opflashes->size();
  // Create a 2D boost::multi_array with shape 16 x nflashes
  typedef boost::multi_array<double, 2> MultiArray;
  MultiArray array(boost::extents[nflashes][m_npmts + 1]);

  for (size_t iflash = 0; iflash < nflashes; ++iflash) {
    const auto& opflash = opflashes->at(iflash);
    array[iflash][0] = opflash.Time() * units::microsecond;
    const auto& pes = opflash.PEs();
    if (pes.size() > m_npmts) {
      raise<ValueError>(
        format("WireCell::OpFlashSource got unexpected number of PMTs expecting %d got %d",
               m_npmts,
               pes.size()));
    }
    for (size_t ipmt = 0; ipmt < pes.size(); ++ipmt) {
      array[iflash][ipmt + 1] = pes[ipmt];
    }
  }
  std::vector<size_t> shape = {array.shape()[0], array.shape()[1]};
  Json::Value md = Json::objectValue;
  auto tensor = std::make_shared<SimpleTensor>(shape, array.data(), md);
  ITensor::vector* itv = new ITensor::vector;
  itv->push_back(tensor);

  // Look up the SBND FrameShift product and forward FrameApplyAtCaf() (the
  // CAF-stage frame shift in ns) into the output TensorSet metadata as
  // "frame_apply_at_caf".  Defaults to 0 (no shift) when the product is
  // absent OR the sbnobj header is not available at compile time.
  double _frame_apply_at_caf = 0.0;
#ifdef HAVE_SBND_FRAMESHIFTINFO
  {
    art::Handle<sbnd::timing::FrameShiftInfo> frameHandle;
    event.getByLabel(fFrameLabel, frameHandle);
    if (!frameHandle.isValid()) {
      if (fDebugFrame) std::cout << "No FrameShift products found." << std::endl;
    }
    else {
      sbnd::timing::FrameShiftInfo const& frame(*frameHandle);
      _frame_apply_at_caf = frame.FrameApplyAtCaf();
    }
  }
#endif

  // Debug CSV: append one row per flash to ./debug.csv.  Header is written on
  // first call to this OpFlashSource process (m_count==0 across all instances
  // is hard to coordinate, so each instance checks for file existence instead).
  // All times in us; frame_apply_at_caf is converted from ns at write time.
  // tpcid is parsed from the instance name (e.g. m_inputTag.instance() looks
  // like "tpc0" or "tpc1" with the legacy fcl).
  {
    static const std::string csv_path = "debug.csv";
    const bool need_header = ![]{
      std::ifstream f(csv_path); return f.good();
    }();
    std::ofstream csv(csv_path, std::ios::app);
    if (need_header) {
      csv << "event,flashid,tpcid,opflash_time_us,opflash_abstime_us,frame_apply_at_caf_us\n";
    }
    int tpcid = -1;
    {
      // The TPC id is encoded in the producer label like "opflashtpc0" /
      // "opflashtpc1".  Fall back to the instance string for setups that
      // attach :tpc0/:tpc1 there.
      auto extract_last_digit = [](const std::string& s) {
        for (auto it = s.rbegin(); it != s.rend(); ++it) {
          if (std::isdigit(static_cast<unsigned char>(*it))) return *it - '0';
        }
        return -1;
      };
      tpcid = extract_last_digit(m_inputTag.label());
      if (tpcid < 0) tpcid = extract_last_digit(m_inputTag.instance());
    }
    const double fc_us = _frame_apply_at_caf / units::microsecond;
    int flashid = 0;
    for (auto const& opflash : *opflashes) {
      csv << event.event() << ','
          << flashid << ','
          << tpcid << ','
          << opflash.Time() << ','
          << opflash.AbsTime() << ','
          << fc_us << '\n';
      ++flashid;
    }
  }

  Configuration set_md;
  set_md["frame_apply_at_caf"] = _frame_apply_at_caf;
  auto tset = std::make_shared<SimpleTensorSet>(event.event(), set_md, ITensor::shared_vector(itv));
  m_tensorsets.push_back(tset);
  m_tensorsets.push_back(nullptr);
}

bool OpFlashSource::operator()(WireCell::ITensorSet::pointer& tensorset)
{
  log->debug("OpFlashSource::operator() m_tensorsets.size() {}", m_tensorsets.size());
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
