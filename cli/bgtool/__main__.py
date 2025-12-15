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
import sys
import traceback

#===============================================================================

from bg2cellml import BondgraphFramework, BondgraphModel, CellMLModel
from bg2cellml import __version__

#===============================================================================

from bgtool.cellml import valid_cellml
from bgtool.utils import log, pretty_log

#===============================================================================

def model2cellml(model: BondgraphModel, cellml_file: Path, save_if_errors: bool=False):
#======================================================================================
    cellml = CellMLModel(model).to_xml()
    has_issues = not valid_cellml(cellml)
    if has_issues and not save_if_errors:
        log.warning('No CellML generated')
    else:
        with open(cellml_file, 'w') as fp:
            fp.write(cellml)
            log.info(f'Generated {pretty_log(cellml_file)}')

def bg2cellml(bondgraph_source: str, output_path: Path, save_if_errors: bool=False, debug: bool=False):
#======================================================================================================
    framework = BondgraphFramework()
    if framework.has_issues:
        for issue in framework.issues:
            traceback.print_exception(issue)
        sys.exit('Issues loading BG-RDF framework')

    source_path = Path(bondgraph_source).resolve()
    if not source_path.exists():
        raise IOError(f'Missing BG-RDF source file: {bondgraph_source}')

    with open(source_path) as fp:
        model_source = fp.read()

    model = BondgraphModel(framework, model_source, model_uri=None,
                           base_iri=source_path.as_uri(), debug=debug)
    if model.has_issues:
        for issue in model.issues:
            traceback.print_exception(issue)
        sys.exit('Issues loading Bondgraph Model')

    model2cellml(model, output_path / f'{source_path.stem}.cellml', save_if_errors)

#===============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert BG-RDF models to CellML')
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--debug', action='store_true', help='Show generated equations for model')
    parser.add_argument('--save-errors', action='store_true', help='Output CellML even if it has errors')
    parser.add_argument('--output', metavar='OUTPUT_DIR', required=True, help='Directory where generated files are saved')
    parser.add_argument('bg_rdf', metavar='BG-RDF', help='Input BG-RDF source file')

    args = parser.parse_args()

    bg2cellml(args.bg_rdf, Path(args.output), save_if_errors=args.save_errors, debug=args.debug)

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================
#===============================================================================
