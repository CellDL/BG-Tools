#===============================================================================
#===============================================================================

import sigfig

from bg2cellml.cellml import base_units, CELLML_NS, CELLML_UNITS, unit_xml
from bg2cellml.units import Unit
from bg2cellml.units.parser import unit_name, unit_name_with_exponent, UnitComponent

#===============================================================================

seen_units: list[str] = []

def unit_seen(name: str):
    return name in seen_units

def seen_unit(name: str):
    seen_units.append(name)

#===============================================================================

def elements_from_units(unit: Unit) -> list[list[str]]:
#======================================================
    if unit_seen(unit.name) or unit.name in CELLML_UNITS:
        return []
    result = []

    def add_component_xml(unit_name: str, component: UnitComponent):
        elements: list[str] = [
            f'<units xmlns="{CELLML_NS}" name="{unit_name}">'
        ]
        base = base_units(component.unit)
        for (n, base_exponent) in enumerate(base[1]):
            (base_name, exponent) = base_exponent
            multiplier = sigfig.round(base[0], 5) if n == 0 else 1
            unit_key = unit_name_with_exponent(base_name, exponent)
            if multiplier != 1:
                unit_key = f'{unit_key}_{multiplier}'
            if not unit_seen(unit_key) and unit_key not in CELLML_UNITS:
                elements.append(unit_xml(base_name,
                                         exponent=exponent,
                                         multiplier=multiplier))
        seen_unit(unit_name)
        if len(elements) > 1:
            elements.append('</units>')
            result.append(elements)

    elements: list[str] = [
        f'<units xmlns="{CELLML_NS}" name="{unit.name}">'
    ]
    xml_components: dict[str, UnitComponent] = {}
    for component in unit.unit_components:
        name = unit_name(component.unit)
        elements.append(unit_xml(unit_name(component.unit),
                                 exponent=component.exponent,
                                 prefix=component.prefix,
                                 multiplier=component.multiplier))
        if not unit_seen(name):
            xml_components[name] = component
    seen_unit(unit.name)
    if len(elements) > 1:
        elements.append('</units>')
        result.append(elements)
        for name, component in xml_components.items():
            add_component_xml(name, component)
    return result

#===============================================================================

if __name__ == "__main__":
    expressions = [
#        "mJ",
#        "mJ.nl",
#        "1.mJ",
#        "1/mJ",
#        "1/J",
#        "mJ/umol",
#        "mg",
#        "mg.mg",
        "nl",
        "fmol/nl",
#        "MW",
#        "nl/fmol",
#        "1/fmol",
        "uF",
#        "kg",
#        "mg",
#        "ampere ** 2 * second ** 4 / kilogram / meter ** 2"
    ]

    for expr in expressions:
        try:
            unit = Unit.from_ucum(expr)
            print(unit.name)
            xml_list = elements_from_units(unit)
            for xml in xml_list:
              print('\n'.join(xml))
            print()
        except:
            print(f"'{expr}'  =>  INVALID")
            raise

#===============================================================================

## 'ampere ** 2 * second ** 4 / kilogram / meter ** 2'  =>  ampere * second / gram / meter: -3

"""

litre = Quantity(0.0010000000000000002, "meter ** 3")
round(math.log(0.0010000000000000002, 10), 6) == -3.0

10**-3 x "meter ** 3" == (10**-1 meter)**3

femtomole_per_nanolitre: {
    '10**-15 mole': Quantity(1, "mole"),
    '(10**-9 liter)**-1': Quantity(0.0010000000000000002, "meter ** 3")
}


femtomole_per_nanolitre: {
    '10**-15 mole': {
        'mole': Quantity(1, "mole"),
    },
    '(10**-9 liter)**-1': {
        'liter': Quantity(0.0010000000000000002, "meter ** 3")
    }
}

'mJ/umol'  =>  millijoule_per_micromole: 10**3 joule / mole
'fmol/nl'  =>  femtomole_per_nanolitre: 10**-6 mole / liter
'1/fmol'  =>  per_femtomole: (10**-15 mole)**-1
'uF'  =>  microfarad: 10**-6 farad
'kg'  =>  kilogram: kilogram
'mg'  =>  milligram: 10**-3 gram


'mJ/umol'  =>  millijoule_per_micromole: 10**3 joule / mole  ==>  [(joule, -3, 1), (mole, -6, -1)]

'fmol/nl'  =>  femtomole_per_nanolitre: 10**-6 mole / liter  ==>  [(mole, -15, 1), (litre, -9, -1)]

'1/fmol'  =>  per_femtomole: (10**-15 mole)**-1  ==>  [(mole, -15, -1)]

'uF'  =>  microfarad: 10**-6 farad  ==> [(farad, -6, 1)]

'kg'  =>  kilogram: kilogram  ==>  [(kilogram, 0, 1)]

'mg'  =>  milligram: 10**-3 gram  ==>  [(gram, -3, 1)]




from ucumvert import PintUcumRegistry
ur = PintUcumRegistry()
x = ur.from_ucum('mJ/fmol/fmol')
str(x.u) == 'millijoule / femtomole ** 2'
ur.from_ucum('mJ/fmol/fmol')
Quantity(1.0, "millijoule / femtomole ** 2")
>>>
>>> x.to_base_units()
Quantity(9.999999999999999e+26, "kilogram * meter ** 2 / second ** 2 / mole ** 2")
"""

#===============================================================================
