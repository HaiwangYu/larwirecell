/**
 * @file   larwirecell/Components/TensorSetMetadataAttacher.h
 * @brief  Stamp the art run/subrun/event onto an ITensorSet's metadata.
 * @date   2026-08-17
 *
 * WHY THIS EXISTS
 *
 * Nothing inside the WCT graph knows the art run/subrun/event.  The tensor
 * ident carries the EVENT number only (CookedFrameSource builds its frame as
 * SimpleFrame(event.event(), ...)), which is why MultiAlgBlobClustering's
 * `rse_from_ident` hardcodes run/subrun to 0.  Every Bee layer and every
 * Magnify-style ROOT file written from inside the graph therefore claims
 * run 0 / subrun 0 unless something art-aware tells it otherwise.
 *
 * This component is that something.  It is an art-event visitor (so wcls calls
 * visit(art::Event&) at the START of the event, before the pgraph runs) and an
 * ITensorSetFilter (so it sits on a tensor edge).  It adds "runNo",
 * "subRunNo" and "eventNo" to the set metadata and passes everything else
 * through untouched.
 *
 * IT IS O(1).  Unlike TensorSetLabeler it never deserializes the point-cloud
 * tree: the output shares the input's ITensor vector by pointer, so the cost is
 * one small JSON copy per event.  Instantiate it freely.
 *
 * PLACEMENT.  Put one immediately upstream of every MultiAlgBlobClustering
 * whose output RSE you care about, and set `rse_from_metadata: true` on that
 * MABC.  MABC also forwards set metadata to its own output, so one attacher at
 * the head of a chain covers everything downstream of it.
 */

#ifndef LARWIRECELL_COMPONENTS_TENSORSETMETADATAATTACHER
#define LARWIRECELL_COMPONENTS_TENSORSETMETADATAATTACHER

#include "WireCellAux/Logger.h"

#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/ITensorSetFilter.h"
#include "larwirecell/Interfaces/IArtEventVisitor.h"

namespace wcls {
  class TensorSetMetadataAttacher : public WireCell::Aux::Logger,
                                    public IArtEventVisitor,
                                    public WireCell::ITensorSetFilter,
                                    public WireCell::IConfigurable {
  public:
    TensorSetMetadataAttacher();
    virtual ~TensorSetMetadataAttacher();

    /// IArtEventVisitor -- called at event start, before the graph runs.
    virtual void visit(art::Event& event);

    /// ITensorSetFilter
    virtual bool operator()(const WireCell::ITensorSet::pointer& in,
                            WireCell::ITensorSet::pointer& out);

    /// IConfigurable
    virtual WireCell::Configuration default_configuration() const;
    virtual void configure(const WireCell::Configuration& config);

  private:
    size_t m_count{0};
    int m_run{0}, m_sub{0}, m_evt{0};
    bool m_seen{false};   // visit() called at least once
    // Metadata key names, configurable only so an unusual downstream consumer
    // can be accommodated; the defaults are what MultiAlgBlobClustering and
    // TensorSetLabeler already use.
    std::string m_run_key{"runNo"};
    std::string m_sub_key{"subRunNo"};
    std::string m_evt_key{"eventNo"};
  };
}

#endif
