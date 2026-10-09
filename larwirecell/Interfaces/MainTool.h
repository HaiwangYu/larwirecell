/** A wcls::MainTool is a main entry point to the WCT from LS

    See the WCLS_tool as the likely only implementation.

    Fixme: maybe need to pick a less generic name for this tool?
*/

#ifndef LARWIRECELL_INTERFACES_MAINTOOL
#define LARWIRECELL_INTERFACES_MAINTOOL

#include <string>

namespace art {
  class Event;
  class ProducesCollector;
}

namespace wcls {

  class MainTool {
  public:
    virtual ~MainTool() noexcept = default;

    /// Accept a base producer.  Typically needed in order to call
    /// prod.produces<Type>() for Type of any expected data
    /// products
    virtual void produces(art::ProducesCollector& collector) = 0;

    /// Accept an event to process.
    virtual void process(art::Event& event) = 0;

    /// art opened an input file; forwarded to the art-event visitors
    /// (ai-helper issue 39).  Default: ignore.
    virtual void respond_to_open_input_file(const std::string& /*filename*/) {}
  };
}

#endif
