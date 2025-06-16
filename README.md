# Using the package

## Instantiate the Duqling interface
``` python
from duqling_interface import DuqlingInterface
duq = Duqling()
```

## List functions
``` python
duq.list_functions()
```
| fname                 |   input_dim | input_cat   | response_type   | stochastic   |
|:----------------------|------------:|:------------|:----------------|:-------------|
| const_fn              |           1 | False       | uni             | n            |
| grlee1                |           1 | False       | uni             | n            |
| banana                |           2 | False       | uni             | n            |
| dms_additive          |           2 | False       | uni             | n            |
| dms_complicated       |           2 | False       | uni             | n            |
| dms_harmonic          |           2 | False       | uni             | n            |
| dms_radial            |           2 | False       | uni             | n            |
| dms_simple            |           2 | False       | uni             | n            |
| foursquare            |           2 | False       | uni             | n            |
| grlee2                |           2 | False       | uni             | n            |
| lim_non_polynomial    |           2 | False       | uni             | n            |
| lim_polynomial        |           2 | False       | uni             | n            |
| multivalley           |           2 | False       | uni             | n            |
| ripples               |           2 | False       | uni             | n            |
| simple_poly           |           2 | False       | uni             | n            |
| squiggle              |           2 | False       | uni             | n            |
| twin_galaxies         |           2 | False       | uni             | n            |
| const_fn3             |           3 | False       | uni             | n            |
| cube3                 |           3 | False       | uni             | n            |
| cube3_rotate          |           3 | False       | uni             | n            |
| detpep_curve          |           3 | False       | uni             | n            |
| ishigami              |           3 | False       | uni             | n            |
| sharkfin              |           3 | False       | uni             | n            |
| simple_machine        |           3 | False       | func            | n            |
| vinet                 |           3 | False       | func            | n            |
| ocean_circ            |           4 | False       | uni             | y            |
| park4                 |           4 | False       | uni             | n            |
| park4_low_fidelity    |           4 | False       | uni             | n            |
| pollutant             |           4 | False       | func            | n            |
| pollutant_uni         |           4 | False       | uni             | n            |
| beam_deflection       |           5 | False       | func            | n            |
| cube5                 |           5 | False       | uni             | n            |
| friedman              |           5 | False       | uni             | n            |
| short_column          |           5 | False       | uni             | n            |
| simple_machine_cm     |           5 | False       | func            | n            |
| stochastic_piston     |           5 | False       | uni             | y            |
| cantilever_D          |           6 | False       | uni             | n            |
| cantilever_S          |           6 | False       | uni             | n            |
| circuit               |           6 | False       | uni             | n            |
| Gfunction6            |           6 | False       | uni             | n            |
| grlee6                |           6 | False       | uni             | n            |
| crater                |           7 | False       | uni             | n            |
| piston                |           7 | False       | uni             | n            |
| borehole              |           8 | False       | uni             | n            |
| borehole_low_fidelity |           8 | False       | uni             | n            |
| detpep8               |           8 | False       | uni             | n            |
| robot                 |           8 | False       | uni             | n            |
| dts_sirs              |           9 | False       | func            | y            |
| steel_column          |           9 | False       | uni             | n            |
| sulfur                |           9 | False       | uni             | n            |
| friedman10            |          10 | False       | uni             | n            |
| ignition              |          10 | False       | uni             | n            |
| wingweight            |          10 | False       | uni             | n            |
| Gfunction12           |          12 | False       | uni             | n            |
| const_fn15            |          15 | False       | uni             | n            |
| Gfunction18           |          18 | False       | uni             | n            |
| friedman20            |          20 | False       | uni             | n            |
| welch20               |          20 | False       | uni             | n            |
| onehundred            |         100 | False       | uni             | n            |
``` python
duq.list_functions(input_dim=2, stochastic="n")
#> ['banana', 'dms_additive', 'dms_complicated', 'dms_harmonic', 'dms_radial']
```

## Query function info
``` python
duq.get_function_info("borehole")
#> {'input_dim': array([8.]),
#>  'input_cat': array([0], dtype=int32),
#>  'response_type': array(['uni'], dtype='<U3'),
#>  'input_range': array([[5.0000e-02, 1.5000e-01],
#>                        [1.0000e+02, 5.0000e+04],
#>                        [6.3070e+04, 1.1560e+05],
#>                        [9.9000e+02, 1.1100e+03],
#>                        [6.3100e+01, 1.1600e+02],
#>                        [7.0000e+02, 8.0000e+01],
#>                        [1.1200e+03, 1.6800e+03],
#>                        [9.8550e+03, 1.2045e+04]])}
```

## Generate data for a specific function
``` python
duq.generate_data("borehole", n_samples=1, seed=42)
#> (array([[0.91480604, 0.93707541, 0.28613953, 0.83044763, 
#           0.64174552, 0.51909595, 0.73658831, 0.1346666 ]]),
#>  array([135.16945286]))
```