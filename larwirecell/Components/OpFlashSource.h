/**
 * @file   larwirecell/Components/OpFlashSource.h
 * @brief  A WCT component read in OpFlash from art::Event and produce ITensorSet.
 * @date   2024-03-20
 */

#ifndef LARWIRECELL_COMPONENTS_OPFLASHSOURCE
#define LARWIRECELL_COMPONENTS_OPFLASHSOURCE

#include "WireCellAux/Logger.h"

#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/ITensorSetSource.h"
#include "larwirecell/Interfaces/IArtEventVisitor.h"

#include "canvas/Utilities/InputTag.h"

namespace wcls {
  class OpFlashSource : public WireCell::Aux::Logger,
                        public IArtEventVisitor,
                        public WireCell::ITensorSetSource,
                        public WireCell::IConfigurable {
  public:
    OpFlashSource();
    virtual ~OpFlashSource();

    /// IArtEventVisitor
    virtual void visit(art::Event& event);

    /// ITensorSetSource
    virtual bool operator()(WireCell::ITensorSet::pointer& ts);

    /// IConfigurable
    virtual WireCell::Configuration default_configuration() const;
    virtual void configure(const WireCell::Configuration& config);

  private:
    // Count how many times we are called
    size_t m_count{0};

    std::deque<WireCell::ITensorSet::pointer> m_tensorsets;

    // label:instance:processName
    art::InputTag m_inputTag{};

    size_t m_npmts{312};

    // sbnd::timing::FrameShiftInfo lookup label and verbose flag.  The
    // FrameApplyAtCaf() shift (ns) is forwarded into the output TensorSet
    // metadata as "frame_apply_at_caf" so downstream consumers can apply it
    // when interpreting flash times.  Defaults reproduce the standard CAF
    // chain ("frameshift" producer, no debug prints).
    std::string fFrameLabel{"frameshift"};
    bool fDebugFrame{false};
  };
}

#endif