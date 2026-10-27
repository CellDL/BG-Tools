#===============================================================================
#===============================================================================

from bg2cellml.units import Unit

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
            result = Unit.from_ucum(expr)
            pu = result.base_units
            d = {str(p): p.base_units for p in pu}
            print(f'        =>  {result.name}: {d}')
#            print(f'    (10**{pu.prefix} {pu.unit})**{pu.exponent}')
#            print(f'    {pu.component}')
#            print(f'    {pu.base_components}')
#            print(f'    {pu.base_units}')
#            print(f'    {list(pu.base_units.unit_items())}')
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
