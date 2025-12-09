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
from typing import Optional

#===============================================================================

import lxml.etree as etree

#===============================================================================


##from ..utils import etree_from_string

def etree_from_string(xml: str) -> etree.Element:
    parser = etree.XMLParser(remove_blank_text=True)
    try:
        return etree.fromstring(xml, parser)
    except etree.XMLSyntaxError as error:
        raise ValueError(error)

#===============================================================================

def get_bgrdf(celldl_file: Path|str) -> str:
    with open(celldl_file) as fp:
        svg = fp.read()
    document = etree_from_string(svg)

    # Use NS helper
    md = document.find('.//{http://www.w3.org/2000/svg}metadata[@id="celldl-rdf-metadata"]')

    print(md)
    #breakpoint()

    # get data-content-type attribute
    # if not XML then get CDATA child section, else get content as XML
    #
    # Return it, along with mimetype
    #
    # "text/turtle"

    return ''
    # find celldl metadata
    # metadata id="celldl-rdf-metadata" data-content-type="text/turtle"

#===============================================================================

get_bgrdf('/Users/dbro078/CellDL/CellDLEditor/examples/Untitled.celldl')
