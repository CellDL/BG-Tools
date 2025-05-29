"""

Load BG framework

======================

Load BG RDF into its own rdflib graph.


Construct networkx graph of PowerBonds:
    Check graph is connected.
    Check domain consistency and identify gyrators.
    Identify nodes as BondElements (BE) or JunctionStructures (JS).

The terminals of a JS network are the BEs it connects to and these
determine possible potential (u) and flow (v) symbols for JS nodes.

For each JS subgraph/network (reactions will divide JS network):
    Build flow and potential matrices to determine their equations.
        This will include transform nodes (Tf and Gy).


Each BE gets specific symbols for its parameter, state, and powerport
variables (and constants, when the same symbol has different values).
    ==> constants' registry (node, symbol, value)



"""

