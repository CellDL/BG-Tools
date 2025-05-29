#===============================================================================
#
#  CellDL and bondgraph tools
#
#  Copyright (c) 2020 - 2025 David Brooks
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#===============================================================================

from pathlib import Path

#===============================================================================

from bg2cellml.bondgraph import Bondgraph
from bg2cellml.bondgraph.framework import BondgraphFramework

#===============================================================================

def main():
    example = Path('../bg-rdf/examples/example_1.ttl')

    framework = BondgraphFramework([
        '../bg-rdf/schema/ontology.ttl',
        '../bg-rdf/schema/elements/general.ttl',
        '../bg-rdf/schema/elements/biochemical.ttl',
        '../bg-rdf/schema/elements/electrical.ttl'
    ])

    bg = Bondgraph(example, framework)

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================
