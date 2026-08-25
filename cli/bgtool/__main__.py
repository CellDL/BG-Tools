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
import shutil
import sys
import tempfile
import traceback
from typing import Optional
import zipfile

#===============================================================================

import lxml.etree as etree

#===============================================================================

from bg2cellml import BondgraphModel
from bg2cellml import __version__
from bg2cellml.bondgraph.framework import get_framework
from bg2cellml.utils import etree_from_string, XMLNamespace

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

#===========================================================================

OMEX_MANIFEST_NS = XMLNamespace('http://identifiers.org/combine.specifications/omex-manifest/')
OMEX_SPEC_NS = XMLNamespace('http://identifiers.org/combine.specifications/')

class OmexMaker:
    def __init__(self):
        self.__files: list[str] = []
        self.__root = etree.Element(OMEX_MANIFEST_NS('omexManifest'), nsmap={ None: str(OMEX_MANIFEST_NS) })

    @property
    def files(self) -> list[str]:
        return self.__files

    def add_content(self, location: str, format: str, master: bool=False):
        content_element = etree.SubElement(self.__root, OMEX_MANIFEST_NS('content'), location=location, format=format)
        if master:
            content_element.attrib['master'] = 'true'
        if location == '.':
            self.__files.append('manifest.xml')
        else:
            self.__files.append(location)

    def as_xml(self):
        tree = etree.ElementTree(self.__root)
        return etree.tostring(tree, encoding='unicode', pretty_print=True)

#===========================================================================

async def model2cellml(bgrdf_model: BondgraphModel, source_path: Path, output: str|None,
                 omex: str|None=None, bgrdf: bool=False, annotate: bool=False,
                 save_if_errors: bool=False):
    cellml_model = bgrdf_model.make_cellml_model()
    cellml = cellml_model.to_xml()
    has_issues = not valid_cellml(cellml)

    if has_issues and not save_if_errors:
        log.warning('No CellML generated')
    else:
        source_name = source_path.stem
        if omex is not None:
            omex_maker = OmexMaker()
            omex_maker.add_content('.', OMEX_SPEC_NS.url('omex'))

            with tempfile.TemporaryDirectory() as build_dir:
                ##build_dir = './omex'
                build_path = Path(build_dir)

                # Copy the source CellDL SVG or BG-RDF into the archive
                shutil.copy(source_path, build_dir)
                omex_maker.add_content(source_path.name,
                                       'text/turtle' if bgrdf else 'image/svg+xml')

                # Save the generated CellML in the archive
                cellml_file = (build_path / source_name).with_suffix('.cellml')
                cellml_file_uri = cellml_file.resolve().as_uri()
                with open(cellml_file, 'w') as fp:
                    fp.write(cellml)
                omex_maker.add_content(cellml_file.name, OMEX_SPEC_NS.url('cellml'), master=True)

                # Save the CellML annotation in the archive
                annotation_file = cellml_file.with_suffix('.ttl')
                # After adjusting file paths in the serialised annotation
                cellml_annotation = await cellml_model.metadata(cellml_file_uri)
                source_uri = source_path.resolve().as_uri()
                source_omex_uri = (build_path / source_path.name).resolve().as_uri()
                cellml_annotation = cellml_annotation.replace(source_uri, source_omex_uri)
                cellml_annotation = cellml_annotation.replace(f'{build_path.resolve().as_uri()}/', '')
                with open(annotation_file, 'w') as fp:
                    fp.write(cellml_annotation)
                omex_maker.add_content(annotation_file.name, OMEX_SPEC_NS.url('omex-metadata'))

                # Add an OMEX manifest
                manifest_file = build_path / 'manifest.xml'
                with open(manifest_file, 'w') as fp:
                    fp.write(omex_maker.as_xml())

                # Zip everything up to create the OMEX archive
                omex_file = Path(output) / omex if output is not None else Path(omex)
                if omex_file.suffix == '':
                    omex_path = omex_file.with_suffix('.omex')
                with zipfile.ZipFile(omex_file, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    for file in omex_maker.files:
                        file_path = build_path / file
                        zipf.write(file_path, file_path.name)
                log.info(f'Created: {pretty_log(omex_file)}')

        else:
            assert output is not None
            cellml_file = (Path(output) / source_name).with_suffix('.cellml')
            with open(cellml_file, 'w') as fp:
                fp.write(cellml)
                log.info(f'Generated {pretty_log(cellml_file)}')
            if annotate:
                cellml_file_uri = cellml_file.resolve().as_uri()
                annotation_file = cellml_file.with_suffix('.ttl')
                with open(annotation_file, 'w') as fp:
                    fp.write(cellml_model.metadata(cellml_file_uri))
                    log.info(f'Annotation: {pretty_log(annotation_file)}')

#===============================================================================

async def bg2cellml(source_file: str, output: str|None, omex: str|None=None,
                    annotate: bool=False, bgrdf: bool=False, save_if_errors: bool=False,
                    debug: bool=False):
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

    await model2cellml(bgrdf_model, source_path, output, omex=omex, bgrdf=bgrdf,
                        annotate=annotate, save_if_errors=save_if_errors)

#===============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert BG-RDF in CellDL to CellML')
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--debug', action='store_true', help='Show generated equations for model')
    parser.add_argument('--save-errors', action='store_true', help='Output CellML even if it has errors')
    parser.add_argument('--omex', metavar='OMEX_FILE', help='Create an OMEX archive containing the generated CellML, annotations, and the source CellDL')
    parser.add_argument('--output', metavar='OUTPUT_DIR', help='Directory where generated files are saved')
    parser.add_argument('--bgrdf', action='store_true', help='Input file is BG-RDF Turtle, not CellDL')
    parser.add_argument('--annotate', action='store_true', help='Output annotations relating BG elements and their CellML variables')
    parser.add_argument('source', metavar='CELLDL', help='Input file')

    args = parser.parse_args()
    if args.output is None and args.omex is None:
        exit('Either `--output`, to specify an output directory, or `--omex`, to specify the OMEX file to create, is required.')

    # output and omex ==> create omex in output directory...

    if args.debug:
        print(f'bg2cellml version {__version__}')
    asyncio.run(bg2cellml(args.source, args.output, omex=args.omex, bgrdf=args.bgrdf,
        annotate=args.annotate, save_if_errors=args.save_errors, debug=args.debug))

#===============================================================================

if __name__ == '__main__':
    main()

#===============================================================================
#===============================================================================
