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

from bg2cellml.bondgraph import BondgraphModelSource
from bg2cellml.cellml import CellMLModel

#===============================================================================

def bg2cellml(bondgraph_rdf_source: str):
    for model in BondgraphModelSource(bondgraph_rdf_source).models:
        cellml = CellMLModel(model).to_xml()
        print(cellml)

#===============================================================================

def main():
    #bg2cellml('../examples/example_RCR.ttl')
    #bg2cellml('../examples/example_A1.ttl')
    #bg2cellml('../examples/example_A2.ttl')
    bg2cellml('../examples/example_A3.ttl')

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================
