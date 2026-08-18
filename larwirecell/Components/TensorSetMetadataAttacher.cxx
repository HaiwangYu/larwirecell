#include "TensorSetMetadataAttacher.h"

#include "WireCellAux/SimpleTensorSet.h"
#include "WireCellUtil/NamedFactory.h"

#include "art/Framework/Principal/Event.h"

WIRECELL_FACTORY(wclsTensorSetMetadataAttacher,
                 wcls::TensorSetMetadataAttacher,
                 wcls::IArtEventVisitor,
                 WireCell::ITensorSetFilter,
                 WireCell::IConfigurable)

using namespace wcls;
using namespace WireCell;

TensorSetMetadataAttacher::TensorSetMetadataAttacher()
  : Aux::Logger("TensorSetMetadataAttacher", "larwirecell")
{
}

TensorSetMetadataAttacher::~TensorSetMetadataAttacher() {}

Configuration TensorSetMetadataAttacher::default_configuration() const
{
  Configuration cfg;
  cfg["run_key"] = m_run_key;
  cfg["subrun_key"] = m_sub_key;
  cfg["event_key"] = m_evt_key;
  return cfg;
}

void TensorSetMetadataAttacher::configure(const Configuration& cfg)
{
  m_run_key = get<std::string>(cfg, "run_key", m_run_key);
  m_sub_key = get<std::string>(cfg, "subrun_key", m_sub_key);
  m_evt_key = get<std::string>(cfg, "event_key", m_evt_key);
}

void TensorSetMetadataAttacher::visit(art::Event& event)
{
  m_run = event.run();
  m_sub = event.subRun();
  m_evt = event.event();
  m_seen = true;
}

bool TensorSetMetadataAttacher::operator()(const ITensorSet::pointer& in,
                                           ITensorSet::pointer& out)
{
  out = nullptr;
  if (!in) {                    // EOS
    log->debug("EOS at call={}", m_count++);
    return true;
  }

  if (!m_seen) {
    // visit() is driven by wcls from the art event loop; if this component was
    // not listed as an inputer it never fires and we would silently stamp
    // 0/0/0 -- exactly the failure mode this component exists to remove.  Warn
    // once and pass through untouched rather than write a wrong answer.
    log->warn("call={}: visit(art::Event&) never called -- is this component "
              "listed in the fcl 'inputers'?  passing through with no RSE",
              m_count);
    out = in;
    ++m_count;
    return true;
  }

  // Preserve whatever the upstream node put here and override only the RSE.
  Configuration md = in->metadata();
  md[m_run_key] = m_run;
  md[m_sub_key] = m_sub;
  md[m_evt_key] = m_evt;

  // Share the input tensors by pointer: no copy, no pctree round trip.
  out = std::make_shared<Aux::SimpleTensorSet>(in->ident(), md, in->tensors());

  log->debug("call={} ident={}: stamped rse=({},{},{})",
             m_count, in->ident(), m_run, m_sub, m_evt);
  ++m_count;
  return true;
}
