# Body oracle provenance

Premade human geometry: the MakeHuman hm08 base mesh plus two adult
macro morphs, vendored — never generated, never modified.

- Upstream: https://github.com/makehumancommunity/makehuman
- Pinned commit: `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`
- License: CC0 1.0 Universal (released September 2020 by Data
  Collection AB, Joel Palmius, Jonas Hauquier — see the `#` headers
  inside each file).

| File | Upstream path | SHA-256 |
|---|---|---|
| `base.obj` | `makehuman/data/3dobjs/base.obj` | `8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c` |
| `male-young.target` | `makehuman/data/targets/macrodetails/caucasian-male-young.target` | `70e228ba7164737dae664454394536fc5935fa48d333c1a97d77e2dc6eacc5f5` |
| `female-young.target` | `makehuman/data/targets/macrodetails/caucasian-female-young.target` | `118379f6e8ba9266247fdb8788a20e1df40a239f97ced0b9905bcbcc74f6e820` |
| `default.mhskel` | `makehuman/data/rigs/default.mhskel` | `99f179bce0aa850b45d4191a1d0d234c5851f881c057439470ded3bddf729a24` |
| `default_weights.mhw` | `makehuman/data/rigs/default_weights.mhw` | `0f3641d651ae3d00ad6b4ccee43142edb109d3bd909d27d9e4139ef1beed8625` |

Only adult morphs are vendored (young male / young female). No
baby/child targets exist anywhere in this pipeline, by policy.

Units: decimeters in-file, converted to meters at load.
Winding: as authored; the loader verifies outward normals on a
sample ring and reverses faces if the file ever changes handedness.
