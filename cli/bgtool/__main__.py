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

import asyncio
from pathlib import Path
import sys
import traceback
from typing import Optional

#===============================================================================

from bg2cellml import BondgraphModel
from bg2cellml import __version__
from bg2cellml.bondgraph.framework import get_framework
from bg2cellml.utils import etree_from_string

#===============================================================================

from bgtool.cellml import valid_cellml
from bgtool.utils import log, pretty_log

#===============================================================================

def get_bgrdf(celldl: str) -> Optional[str]:
#===========================================
    document = etree_from_string(celldl)
    metadata_element = document.find('.//{http://www.w3.org/2000/svg}metadata[@id="celldl-rdf-metadata"]')
    if metadata_element is not None:
        if metadata_element.attrib.get('data-content-type') == 'text/turtle':
            return metadata_element.text

def model2cellml(bgrdf_model: BondgraphModel, cellml_file: Path, save_if_errors: bool=False):
#============================================================================================
    cellml_model = bgrdf_model.make_cellml_model()
    cellml = cellml_model.to_xml()
    has_issues = not valid_cellml(cellml)
    if has_issues and not save_if_errors:
        log.warning('No CellML generated')
    else:
        with open(cellml_file, 'w') as fp:
            fp.write(cellml)
            log.info(f'Generated {pretty_log(cellml_file)}')

async def bg2cellml(source_file: str, output_path: Path, bgrdf: bool=False, save_if_errors: bool=False, debug: bool=False):
#==========================================================================================================================
    framework = await get_framework()
    if framework.has_issues:
        for issue in framework.issues:
            traceback.print_exception(issue)
        sys.exit('Issues loading BG-RDF framework')

    source_path = Path(source_file).resolve()
    if not source_path.exists():
        raise IOError(f'Missing source file: {source_file}')
    with open(source_path) as fp:
        if bgrdf:
            model_source = fp.read()
        else:
            model_source = get_bgrdf(fp.read())
    if model_source is None or model_source == '':
        raise TypeError(f"{source_file} doesn't contain BG-RDF")
    bgrdf_model = framework.make_bondgraph_model(source_path.as_uri(), model_source, debug=debug)
    if bgrdf_model.has_issues:
        for issue in bgrdf_model.issues:
            if debug:
                traceback.print_exception(issue)
            else:
                print(issue.reason)
        sys.exit('Issues loading Bondgraph Model')

    model2cellml(bgrdf_model, output_path / f'{source_path.stem}.cellml', save_if_errors)

#===============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert BG-RDF in CellDL to CellML')
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--debug', action='store_true', help='Show generated equations for model')
    parser.add_argument('--save-errors', action='store_true', help='Output CellML even if it has errors')
    parser.add_argument('--output', metavar='OUTPUT_DIR', required=True, help='Directory where generated files are saved')
    parser.add_argument('--bgrdf', action='store_true', help='Input file is BG-RDF Turtle, not CellDL')
    parser.add_argument('source', metavar='CELLDL', help='Input file')

    args = parser.parse_args()

    if args.debug:
        print(f'bg2cellml version {__version__}')
    asyncio.run(bg2cellml(args.source, Path(args.output), bgrdf=args.bgrdf, save_if_errors=args.save_errors, debug=args.debug))

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================
#===============================================================================
