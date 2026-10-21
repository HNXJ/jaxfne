# JDNA Compact Grammar — Preregistration (Private Étude)

**Status:** `PREREGISTRATION — DOCUMENTATION_ONLY` — design doc, not implementation.  
**Scope:** Private étude in `artifacts/etudes/` — not part of shipped package.  
**Frozen?** No. No `artifacts/publication/**` outputs, no `artifacts/figures/publication/**` writes.  
**Code change in this task:** **None.** This document explicitly makes **zero** changes to `jaxfne/jdna/*.py` (no new parser, no DSL execution, no new genome schema). All implementation, if pursued, is a future step after this preregistration is reviewed.  
**Authority:** `jaxfne.jdna.genome` (`PseudoGenome` / `AreaGenome` / `LayerGenome` / `ConnectionRuleGenome` / `develop` / `genome_rules_hash` / `validate_genome` / `declared_constraints`) remains the sole implemented truth at `0.4.17`. This doc is a compiler-design sketch on top of that truth.  
**H11 note:** Any executable validation of the grammar proposed here **must** use isolated `tmp_path` testing — generators write only to a test-provided temporary directory (or `tmp_path` fixture), never to a pre-existing gitignored `artifacts/etudes/.../*.png` or `*.json`. No test may depend on an untracked pre-generated artifact. See §11.  
**Task:** `F1` (READY).  
**Date:** 2026-08-26.  
**Workspace:** `C:\workspace\jaxfne` (head `0b8c22e` at task start).

---

## 1. Purpose

Propose a compact, human-writable surface grammar for JDNA `PseudoGenome` authoring that:

- removes repetition when many related column/area variants differ only by one population tweak;
- distinguishes **absolute** population edits from **relative-fraction** edits with an explicit denominator rule;
- gives inheritance a deterministic deep-merge semantics that preserves `validate_genome` invariants;
- composes with the existing `NeuronalTensor` unifier `merge_neuronal_tensors` and the existing `develop(G, K_D)` determinism;
- records provenance via the existing `genome_rules_hash` (no new hash surface).

This is the `define / inherit / use` + `A-80` + `A*0.08` compiler concept referenced as deferred in `artifacts/etudes/provenance_relocation_jdna.json` (harness synthesis Section 4). E2 does not depend on it.

---

## 2. Non-goals (explicitly deferred)

- No runtime `H_D → H_R` projection, no `simulate()`-time structural growth (`𝒩(t) → 𝒩(t⁺)`).
- No change to `pseudogenome_v1` on-disk schema or to `NeuronalTensor` / `Configuration` bridge.
- No new on-disk genome files shipped in `jaxfne/jdna/genomes/` in this task.
- No parser shipped in `jaxfne/` in this task (this file is the only artifact).
- No AGSDR-over-genomes.

---

## 3. Relation to implemented truth

```
Implemented (0.4.17)            Proposed (this doc, not implemented)
─────────────────────           ────────────────────────────────────
PseudoGenome dataclass   <───   define  { ... }          (sugar)
python-level inherit     <───   inherit Child from Parent { ... }
manual develop/merge     <───   use Genome [tweaks]
develop(G, K_D)          ───    unchanged (K_D §8)
genome_rules_hash        ───    unchanged (§9)
merge_neuronal_tensors   ───    unchanged (§7)
validate_genome          ───    governs merge result (§6)
```

Any future implementation must be a **pure desugar** to the existing `jaxfne.jdna.genome` dataclasses before calling `validate_genome` → `develop`.

---

## 4. Grammar — BNF and PEG

### 4.1 Lexical elements

```
IDENT      ::= [A-Za-z_][A-Za-z0-9_-]*          // area, layer, cell-type, genome names
INT        ::= [0-9]+
FLOAT      ::= [0-9]+ "." [0-9]+  |  "." [0-9]+  |  [0-9]+
STRING     ::= '"' ( [^"\n\\] | "\\" . )* '"'
COMMENT    ::= "#" [^\n]*
WS         ::= [ \t\r\n]+
```

Identifiers are case-sensitive. Genome, area, layer, and cell-type names share `IDENT` but live in distinct namespaces. `A` in `A-80` / `A*0.08` is a placeholder for a concrete named target (see §5).

### 4.2 BNF (context-free, for specification)

```bnf
<GenomeFile>   ::= { <Statement> }

<Statement>    ::= <DefineStmt> | <InheritStmt> | <UseStmt>

<DefineStmt>   ::= "define" <GenomeName> "{" <Body> "}"
<InheritStmt>  ::= "inherit" <GenomeName> "from" <GenomeName> "{" <OverrideBody> "}"
<UseStmt>      ::= "use" <GenomeName> { <Tweak> } [ <CompositionClause> ]

<Body>         ::= { <AreaBlock> | <AreaConnBlock> | <DevParamsBlock> }
<AreaBlock>    ::= "area" <AreaName> [ <PoseClause> ] "{" { <LayerBlock> | <InterConnBlock> } "}"
<LayerBlock>   ::= "layer" <LayerName> <LayerProps>
<LayerProps>   ::= "n" "=" <NExpr>  "depth" "=" <DepthBand>
                   "fractions" "=" <FractionsMap>
                   [ "tolerance" "=" <ToleranceMap> ]
                   [ "geometry" "=" <GeometryMap> ]
                   [ "sizes" "=" <SizesMap> ]

<DevParamsBlock> ::= "development" "{" { <DevParam> } "}"
<DevParam>       ::= IDENT "=" (INT | FLOAT | STRING)

<AreaConnBlock>  ::= "connect" <AreaConnRule>
<InterConnBlock> ::= "connect" <InterConnRule>

<OverrideBody> ::= { <AreaOverride> | <AreaConnOverride> | <DevParamsOverride> }
<AreaOverride> ::= "area" <AreaName> [ <PoseClause> ] "{" { <LayerOverride> | <InterConnOverride> } "}"
<LayerOverride>::= "layer" <LayerName> [ <LayerPropsPartial> ]
<LayerPropsPartial> ::= { ("n" "=" <NExpr>)
                        | ("depth" "=" <DepthBand>)
                        | ("fractions" "=" <FractionsMap>)
                        | ("tolerance" "=" <ToleranceMap>)
                        | ("geometry" "=" <GeometryMap>)
                        | ("sizes" "=" <SizesMap>) }

<Tweak>        ::= <AbsoluteTweak> | <FractionTweak>
<AbsoluteTweak>::= <TargetRef> "-" INT                 ; A-80
<FractionTweak>::= <TargetRef> "*" FLOAT               ; A*0.08

<TargetRef>    ::= <AreaName> [ "." <LayerName> [ "." <CellType> ] ]
                |  <AreaName> "." <LayerName>          ; most common for tweaks
                ;  bare <AreaName> in a tweak means "the default layer" only if
                ;  the area has exactly one layer; otherwise it is a parse error.

<CompositionClause> ::= "compose" "with" <GenomeName> { "," <GenomeName> }
                      [ "poses" "=" <PoseList> ] [ "as" <GenomeName> ]

<NExpr>        ::= INT | <AbsoluteTweak> | <FractionTweak>
<DepthBand>    ::= "[" FLOAT "," FLOAT "]"
<FractionsMap> ::= "{" { <CellType> ":" FLOAT [","] } "}"
<ToleranceMap> ::= "{" { <CellType> ":" "[" FLOAT "," FLOAT "]" [","] } "}"
<GeometryMap>  ::= "{" { IDENT ":" (STRING | FLOAT | "[" FLOAT "," FLOAT "]") [","] } "}"
<SizesMap>     ::= "{" { <CellType> ":" FLOAT [","] } "}"
<PoseClause>   ::= "pose" "{" { IDENT ":" (STRING | FLOAT | "[" FLOAT "," FLOAT "," FLOAT "]") [","] } "}"

<GenomeName>   ::= IDENT
<AreaName>     ::= IDENT
<LayerName>    ::= IDENT
<CellType>     ::= IDENT
```

Line comments (`# ...`) are allowed wherever `WS` is allowed. File is whitespace-insensitive beyond token boundaries; newlines are not significant.

### 4.3 PEG (for deterministic parsing; ordered choice, longest match)

```peg
GenomeFile      <- WS Statement* WS EOF
Statement       <- DefineStmt / InheritStmt / UseStmt

DefineStmt      <- "define" WS GenomeName WS "{" WS Body WS "}" WS
InheritStmt     <- "inherit" WS GenomeName WS "from" WS GenomeName WS "{" WS OverrideBody WS "}" WS
UseStmt         <- "use" WS GenomeName WS Tweak* WS CompositionClause? WS

Body            <- (AreaBlock / AreaConnBlock / DevParamsBlock)*
AreaBlock       <- "area" WS AreaName WS PoseClause? WS "{" WS (LayerBlock / InterConnBlock)* WS "}" WS
LayerBlock      <- "layer" WS LayerName WS LayerProps WS
DevParamsBlock  <- "development" WS "{" WS DevParam* WS "}" WS

OverrideBody    <- (AreaOverride / AreaConnOverride / DevParamsOverride)*
AreaOverride    <- "area" WS AreaName WS PoseClause? WS "{" WS (LayerOverride / InterConnOverride)* WS "}" WS
LayerOverride   <- "layer" WS LayerName WS LayerPropsPartial? WS

Tweak           <- FractionTweak / AbsoluteTweak   # ordered: try "*" before "-" to avoid IDENT prefix ambiguity
AbsoluteTweak   <- TargetRef "-" INT
FractionTweak   <- TargetRef "*" FLOAT

TargetRef       <- AreaName ("." LayerName ("." CellType)?)?
CompositionClause <- "compose" WS "with" WS GenomeName WS ("," WS GenomeName WS)* WS ("poses" WS "=" WS PoseList WS)? ("as" WS GenomeName WS)?

NExpr           <- (AbsoluteTweak / FractionTweak) / INT
DepthBand       <- "[" WS FLOAT WS "," WS FLOAT WS "]"

# Maps — order-insensitive; PEG helper "MapEntry" uses WS separators, trailing comma allowed
FractionsMap    <- "{" WS (CellType WS ":" WS FLOAT WS ("," WS)?)* WS "}" WS
ToleranceMap    <- "{" WS (CellType WS ":" WS "[" WS FLOAT WS "," WS FLOAT WS "]" WS ("," WS)?)* WS "}" WS

# Lexical — longest-match, keywords not IDENT
GenomeName <- IDENT
AreaName   <- IDENT
LayerName  <- IDENT
CellType   <- IDENT
IDENT      <- !Keyword [A-Za-z_][A-Za-z0-9_-]*   # negative lookahead for keywords
Keyword    <- ("define" / "inherit" / "from" / "use" / "area" / "layer" / "connect" / "compose" / "with" / "poses" / "as" / "development") !IdentChar
IdentChar  <- [A-Za-z0-9_-]
FLOAT      <- ([0-9]+ "." [0-9]+ / "." [0-9]+ / [0-9]+)   # consumed before INT in NExpr
INT        <- [0-9]+
WS         <- ([ \t\r\n] / COMMENT)*
COMMENT    <- "#" [^\n]*
EOF        <- !.
```

**Disambiguation rules (normative):**

1. `TargetRef "*" FLOAT` and `TargetRef "-" INT` are **postfix operators** on a reference. They never denote arithmetic expressions; there is no binary `+`, no precedence beyond the two operators, and no nesting (`A*0.08*0.1` is a parse error — write two tweaks).
2. `FLOAT` is tried before `INT` inside `NExpr`; bare `80` parses as `INT`, `0.08` as `FLOAT`.
3. `IDENT` not matching a `Keyword` prevents `define` being parsed as a genome name.
4. Duplicate `layer <Name>` inside one `area { ... }` is a parse-time error (mirrors `validate_genome` duplicate-layer check).

### 4.4 Desugaring sketch (future compiler, not shipped)

```
define  G { ... }                  ──▶  PseudoGenome(name="G", areas=[...], ...)
inherit C from P { overrides }     ──▶  PseudoGenome = deep_merge(P, overrides)   (§6)
use     G  A-80  B*0.08             ──▶  ephemeral PseudoGenome' = apply_tweaks(G, [A-80, B*0.08])
                                       then develop(G', K_D) or merge_neuronal_tensors(develop(...))
```

All three desugar to the existing dataclass constructor + `validate_genome` before any `develop`.

---

## 5. `A-80` and `A*0.08` semantics — denominator rule

### 5.1 Syntax recap

| Form | Spelling | Meaning | Operand domain |
|------|----------|---------|----------------|
| `A-80` | `<TargetRef> "-" <INT>` | **Absolute** cell count `80` for the named target | `INT > 0` |
| `A*0.08` | `<TargetRef> "*" <FLOAT>` | **Fraction** `0.08` of the **enclosing area** | `0 ≤ FLOAT ≤ 1` |

`A` is a concrete `TargetRef`. Bare examples in task shorthand `A-80` / `A*0.08` stand for e.g. `V1.L4-80` or `V1.L4*0.08`. The shorthand `V1-80` is legal only when `V1` has exactly one layer; otherwise the target must be qualified as `Area.Layer`.

### 5.2 Absolute semantics: `TargetRef "-" INT`

**Definition.** `V1.L4-80` means: the layer `L4` in area `V1` shall contain exactly `n_neurons = 80`.

**Rules.**

1. The `INT` must be strictly positive. `0` and negative values are rejected at parse/validation time (mirrors `validate_genome: n_neurons > 0`).
2. Scope is a single `LayerGenome.n_neurons`. `V1-80` on a bare area applies only to the single-layer special case and rewrites that layer's `n_neurons` to `80`.
3. Multiple absolute tweaks to the same `TargetRef` within one `use` statement: **last writer wins** (deterministic, order is file order).
4. `A-80` **does not** implicitly alter `cell_type_fractions` or `fraction_tolerance`; composition of counts across cell types remains governed by `cell_type_fractions` + `fraction_tolerance` + `fraction_jitter_sigma` at `develop` time (§8).
5. If both an absolute and a fractional tweak name the same layer in one statement, the **later** lexical tweak wins (no implicit addition).

**Example effects.**

```
# before: V1.L4 { n=100, fractions {E:0.8, PV:0.2} }
use Base V1.L4-80
# after:  V1.L4 { n=80,  fractions {E:0.8, PV:0.2} }
# develop with sigma=0: E=64, PV=16 (exact); with sigma>0: counts via §8 within tolerance.
```

### 5.3 Fractional semantics: `TargetRef "*" FLOAT` — "08 % of the enclosing area"

**Definition.** `V1.L4*0.08` means: the layer `L4` shall contain a count equal to `0.08` × **N_enclosing**, where **N_enclosing = N_area** = total neurons of the enclosing area `V1`.

**Denominator rule (normative, resolves prior ambiguity).**

> **Denominator = N_enclosing = sum_{layers L' in area A} n_neurons(L')**, evaluated on the **parent materialization** immediately before applying the tweak set of the current statement.

Concretely:

1. Let `G₀` be the `PseudoGenome` denoted by `use G₀` (or the child genome after `inherit`'s deep merge, before `use` tweaks). Compute `N_area(G₀, A) = Σ_{L' ∈ A} n_neurons(G₀, A, L')`.

2. For each `Area.Layer "*" f` tweak, the **fractional target count** is `target = N_area(G₀, A) * f`. This is a real number; integer allocation is deferred to step 4.

3. **Absolute tweaks take precedence over fractional base.** If a `use` statement contains both `V1.L2-80` and `V1.L3*0.08`, first fix every `A-INT` layer to its absolute count `n_abs`, then recompute the **residual denominator** for fractional allocation:
   ```
   N_fixed    = Σ_{L' with A-INT} INT(L')
   N_frac_base = N_area(G₀, A)          # denominator rule, not N_area - N_fixed
   # fractional targets are N_frac_base * f, but then renormalized:
   ```
   The desugarer computes raw fractional targets as `N_frac_base * f`, then normalizes the **fractional subset** by largest-remainder (see step 4) to fill exactly `N_area(G') = (N_area(G₀) adjusted for any net change implied by absolutes)` if the intention is to preserve the area total. Two modes are distinguished:

   - **Area-total–preserving mode (default for pure `use` tweaks):** the area total `N_area` is held constant at `N_area(G₀)`. Absolutes are carved out first; the remaining `N_area - N_fixed` neurons are distributed among the fractionally-tweaked layers proportionally to their `f` values, with any non-mentioned layers retaining their original `n_neurons` proportionally scaled to fill the remainder. This keeps `Σ n` invariant so depth/semantic area size is not silently changed by a layer tweak.
   - **Area-total–mutating mode (explicit):** writing `V1.L4-80` without other qualifiers is allowed to change `N_area` (new total = old total - old L4 + 80). A future compiler flag `area_total="mutate"` will make this explicit; until then the default is **preserving** and a single `A-80` with no siblings preserves vs mutates based on whether the tweak list names only that layer (mutate allowed, documented in provenance `development_parameters` commentary).

   For the **preregistration default**, `A*0.08` is **area-total-preserving**: `0.08` means eight percent of the area that was there before the tweak, and the sibling layers absorb the complementary `0.92`.

4. **Integer rounding (mirrors `_allocate_counts` / `_counts_from_fractions`):** fractional targets are converted to integer `n_neurons` by the **largest-remainder method** over the fractional subset, so the integer `n_neurons` values sum exactly to the intended area total. No `n_neurons` is ever 0 unless the fractional target is exactly 0 and the area total is respected.

5. **Fraction domain:** `0.0 ≤ f ≤ 1.0`. `f = 0.0` empties the layer only if explicitly intended; `validate_genome` will then fail because `n_neurons > 0` is required — so `f = 0.0` is a parse-legal but semantically invalid value unless an absolute override also covers that layer. `f = 1.0` assigns the whole area to one layer (other layers become `0`, likewise invalid) — validation will catch it.

6. **Enclosing definition:** for `V1.L4*0.08`, the enclosing area is the area named before the dot (`V1`). For a bare `V1*0.08` (single-layer area sugar), the enclosing area is the global total over all areas — but this form is discouraged; prefer qualified `Area.Layer*frac` for clarity. A future lint will warn on bare-area fractions when `areas.len() > 1`.

7. **Multiple fractional tweaks in one statement** are evaluated **atomically** against the same `N_area(G₀, A)` snapshot, not sequentially (so `V1.L2*0.2 V1.L3*0.3` means L2 = 20 % of original area, L3 = 30 % of original area, not 30 % of the post-L2 genome).

8. **Cross-area fractions never mix.** A tweak naming `V1.L4*0.08` never draws from `V2`'s count; if a genome has multiple areas, each area's fractions are resolved within that area's own denominator.

**Worked denominator examples.**

```
# Genome G0: area V1 { L2 n=200, L3 n=300, L4 n=500 } => N_V1 = 1000
use G0 V1.L4*0.08
# N_enclosing = N_V1(G0) = 1000
# raw target L4 = 1000 * 0.08 = 80
# area-total-preserving => L4=80, remaining 920 split among L2,L3 proportionally to prior 200:300 => L2≈368, L3≈552 (largest remainder)
# new G': V1 { L2≈368, L3≈552, L4=80 }  sum=1000 preserved

# With absolute present:
# use G0 V1.L2-80 V1.L4*0.10
# N_fixed = 80 (L2)
# N_enclosing = 1000
# raw fractional target L4 = 1000*0.10=100
# preserving: remaining after fixed = 920; fractional share for L4 among fractional pool = 100/100=1 => L4=100, leftover 820 for non-mentioned L3 => L3=820 => final {L2=80, L3=820, L4=100}
```

### 5.4 `n_neurons` via `NExpr` in `define` / `inherit`

Inside a `define` or `inherit` layer header, `n = 80` (bare int), `n = V1.L4-80`, and `n = V1.L4*0.08` are all sugar for the same rules but resolved **within the block**: `*` forms use the area total of the genome being defined as it stands after processing preceding layer declarations in file order. Since this is subtle, the **recommended style** is to use bare `INT` in `define` bodies and reserve `A*F` for `use` statements; the preregistration records the more general form as legal but the style guide will recommend against it.

---

## 6. Inheritance merge semantics

### 6.1 `inherit Child from Parent { ... }`

**Identity.** Creates a new `PseudoGenome(name="Child", ...)` by deep-copying `Parent` and applying the override body.

**Implemented surface it desugars to (no new runtime):**

```python
parent = load_pseudogenome(parent_dict)   # or already in registry
child  = PseudoGenome(
    name="Child",
    description=override_description or parent.description,
    development_parameters={**parent.development_parameters, **override_dev_params},
    areas=tuple(merged_areas),
    area_connections=tuple(merged_area_connections),
)
validate_genome(child)  # must pass; otherwise inheritance is invalid
```

### 6.2 Deep-merge rules — `AreaGenome` / `LayerGenome`

Merge is **by name** at each level (`area.name`, `layer.name`).

| Level | Match | Action |
|-------|-------|--------|
| `PseudoGenome` | — | `name` replaced by child name; `description` replaced if override provides it, else inherited; `schema_version` always `pseudogenome_v1`; `development_parameters` shallow-merged with child keys overriding parent keys (child can delete a key by setting it to `null` / `None` in a future explicit syntax — preregistration leaves delete as out-of-scope, only override). |
| `AreaGenome` | area name **not** in parent | **Add** new area verbatim (must have at least one layer). |
| `AreaGenome` | area name **in** parent | **Merge**: `pose` deep-merged key-by-key (child keys override); `layers` merged per layer rules below; `inter_connections` replaced by child list if child provides any `connect` lines for that area, otherwise inherited. |
| `LayerGenome` | layer name **not** in parent area | **Add** new layer (requires `n_neurons`, `depth_band`, `cell_type_fractions`; defaults for geometry/sizes as in `LayerGenome`). |
| `LayerGenome` | layer name **in** parent area | **Per-field override**: each field present in the child's `LayerPropsPartial` replaces the parent field; absent fields are inherited. Replacement is **field-granular**, not whole-object — e.g. `layer L4 n=80` replaces only `n_neurons`, keeping parent `fractions`, `tolerance`, `depth_band`, `geometry`. |

**Geometry / relative_sizes:** child map merges key-by-key over parent map (child keys override, parent keys retained).

**Pose:** `pose` map merged key-by-key similarly.

### 6.3 `fraction_tolerance` handling (normative)

`LayerGenome.fraction_tolerance: Mapping[str, tuple[float,float]]` is a **partial** map over cell types. Merge semantics:

1. Child `tolerance = { E: [0.7, 0.85] }` **replaces** the tolerance interval for `E` only; tolerance intervals for other cell types (`PV`, `SST`, `VIP`) are inherited from the parent layer unchanged.
2. Setting a tolerance for a cell type not in `cell_type_fractions` is invalid — caught by `validate_genome` ("tolerance for undeclared cell type").
3. Removing a tolerance (reverting to exact) is expressed as `tolerance = {}` for the layer (clear all), or a future `tolerance.E = null` delete syntax (deferred; not in this preregistration's minimal grammar — for now, re-declare the whole map without the entry).
4. After merge, the **joint feasibility** invariant is rechecked:
   ```
   sum(lo) <= 1 <= sum(hi)    where lo = tolerance[ct].lo else frac[ct], hi = tolerance[ct].hi else frac[ct]
   ```
   and per-entry `0 ≤ lo ≤ base ≤ hi ≤ 1`. Failure is a hard error (no silent fallback).

**Example tolerance override:**

```
define Base { area V1 { layer L4 n=800 depth=[0.4,0.6] fractions {E:0.8, PV:0.2} tolerance {E:[0.7,0.9], PV:[0.1,0.3]} } }
inherit Child from Base {
  area V1 {
    layer L4 tolerance {E:[0.78,0.82]}   # PV tolerance inherited as [0.1,0.3]
  }
}
# After merge, L4 tolerance = {E:[0.78,0.82], PV:[0.1,0.3]} — joint feasibility still holds (0.88 ≤1≤1.12)
```

### 6.4 `depth_band` sum constraints

Each `LayerGenome.depth_band: tuple[float,float]` obeys `0 ≤ lo < hi ≤ 1`. For inheritance:

1. A child's `depth=[lo,hi]` replaces the parent band for that layer only.
2. **No automatic re-tiling** is performed. After merge, the area's layers are **not** re-sorted or re-normalized; the genome author is responsible for keeping bands disjoint and inside `[0,1]`. Validation checks per-layer `0 ≤ lo < hi ≤ 1` but does **not** yet check inter-layer overlap — a future strict mode will add `_sorted non-overlapping + covers [0,1] without gap` as a warning, not a hard error in this preregistration.
3. The **sum constraint** referred to in the task is the coverage invariant for the canonical column use-case: a laminar stack that intends to tile `[0,1]` should satisfy `∪ bands = [0,1]` with no overlaps/gaps. The preregistration notes this as a **style invariant**, not a `validate_genome` hard error, to avoid breaking existing single-area single-layer genomes.
4. A child that inserts a new layer (e.g. adding `L1`) must supply its own `depth` that does not overlap an inherited layer's band — overlap is a lint error in a future tool, but for preregistration it is documented as "author must ensure non-overlap."

### 6.5 Inter-connection inheritance

- If the child's `area { ... }` block for a given area contains **no** `connect` lines, the parent's `inter_connections` for that area are inherited verbatim.
- If the child contains **any** `connect` lines for that area, they **replace** the parent's inter-connection list for that area (whole-list replacement, not per-rule merge). This avoids ambiguous partial rule identity.
- `area_connections` (between-area) at the genome level follow the same rule: child list replaces parent list if child declares any `connect` at top level; otherwise inherited.

### 6.6 Error taxonomy for inheritance

| Condition | Error |
|-----------|-------|
| Duplicate `area` name in child block | Parse error |
| Duplicate `layer` name within one `area` override | Parse error |
| `layer` override names non-existent layer but supplies only partial props without `n`/`depth`/`fractions` | Validation error (incomplete layer spec after merge) |
| `fractions` after merge don't sum to 1 (±1e-6) | `validate_genome` failure |
| Tolerance band doesn't contain base fraction | `validate_genome` failure |
| Joint tolerance feasibility `sum(lo)>1` or `sum(hi)<1` | `validate_genome` failure |
| `n_neurons ≤ 0` after `A-80` or `A*frac` desugar | `validate_genome` failure |
| `depth_band` violates `0≤lo<hi≤1` | `validate_genome` failure |

---

## 7. Composition with `merge_neuronal_tensors`

After genomes are materialized to phenotypes, composition reuses the existing `jaxfne.neuronal_tensor.merge_neuronal_tensors`.

### 7.1 Intended flow

```
G1, G2 : PseudoGenome                          # via define / inherit
T1 = develop(G1, K_D=0)                        # NeuronalTensor
T2 = develop(G2, K_D=1)                        # independent K_D
T  = merge_neuronal_tensors([T1, T2],          # existing unifier
     poses=[Pose3D(...), Pose3D(...)],
     name="composed")
model = construct(T, RuntimeConfiguration(...)) # K_S = runtime seed (distinct from K_D)
```

The compact grammar's `use ... compose with ...` sugar desugars to exactly this sequence. The grammar never merges `PseudoGenome` objects directly (no genome-level merge); composition is at the **tensor** level after `develop`, because only tensors carry the resolved integer counts and `NeuronType.fraction` that `merge_neuronal_tensors` expects.

### 7.2 Rename / provenance handling

- Area-name collisions across `T1, T2` are resolved by `merge_neuronal_tensors`'s suffix rule (`"V1" → "V1_1"`), with `AreaConnection` source/target names rewritten to match (see `jaxfne/neuronal_tensor.py:539-625`).
- Each `Ti` retains its own `provenance` (genome name, `genome_rules_hash`, `K_D`, `phenotype_sha256`) in `Ti.provenance`. After `merge_neuronal_tensors`, the merged tensor's `provenance` is `None` by construction (merge is a tensor combinator, not a develop). The **audit trail** for the composed system is the ordered list of input provenances, carried in caller-side metadata (e.g. `save_json({"inputs":[{"genome":..., "genome_sha256":..., "K_D":...}, ...], "merged": T.to_dict()})`) — not in `T.provenance` itself.
- `connectivity_mode` propagation follows `merge_neuronal_tensors`: if any input is `explicit`, the merged result is `explicit` with concatenated `area_connections`; otherwise `unspecified`.

### 7.3 Grammar surface for composition

```
use Base compose with VariantA, VariantB as StackedColumn
use Base compose with VariantA poses [{plane:"xy", translation:[0,0,0]}, {plane:"xy", translation:[1,0,0]}]
```

Poses, when supplied, must have one entry per area in flattened encounter order (tensor 0 area 0, tensor 0 area 1, ..., tensor 1 area 0, ...) exactly as `merge_neuronal_tensors(tensors, poses)` requires — the parser checks arity at desugar time.

---

## 8. `K_D` determinism via `develop(G, K_D)` — PRNG split per area/layer

This section restates implemented behavior (`jaxfne/jdna/genome.py:604-734`) that the compact grammar **inherits unchanged**.

### 8.1 Development is deterministic in `K_D`

```
develop: (G, K_D: int) -> NeuronalTensor       with   develop(G, K) == develop(G, K)   for every K
```

`K_D` is the integer `seed` argument to `develop`. It lives in the **development PRNG domain**, disjoint from `K_S` (`RuntimeConfiguration.seed`, construction + simulate noise) and `K_A` (optimizer seed):

```
K_D ≠ K_S ≠ K_A     (docs/guides/jdna.md §PRNG separation)
```

Changing `K_S` never alters the developed `NeuronalTensor`; changing `K_D` never alters the runtime noise sequence.

### 8.2 PRNG split hierarchy (implemented)

Pseudocode of the live implementation:

```python
key = jax.random.PRNGKey(int(K_D))
for area in genome.areas:
    key, area_key = jax.random.split(key)           # one split per area, in genome.areas order
    for layer in area.layers:
        area_key, layer_key = jax.random.split(area_key)  # one split per layer, in area.layers order
        counts = _allocate_counts(layer, sigma, layer_key)
        # ... assemble Layer, Geometry, NeuronType
```

- Splits are **ordered** by the declaration order of areas/layers in the `PseudoGenome`. Reordering areas or layers changes the `layer_key` assignment and therefore the realized jitter — this is intentional and documented.
- `sigma = development_parameters["fraction_jitter_sigma"]` (default `0.0`). When `sigma == 0` or `fraction_tolerance` is empty, `counts` are exact even splits of `base fractions` via largest remainder, ignoring the key (deterministic without randomness). When `sigma > 0` and tolerances exist, base fractions are jittered as `base + sigma * Normal(0,1)` (drawn with `layer_key`), then projected onto the box-constrained simplex `C = {p : Σp=1, lo≤p≤hi}` by `_project_box_simplex` (KKT bisection, deterministic), then largest-remainder integer allocation.
- `_project_box_simplex` requires joint feasibility `Σ lo ≤ 1 ≤ Σ hi` (checked by `validate_genome` and re-checked at projection); infeasible genomes fail fast.

### 8.3 Compact grammar does not alter PRNG structure

- `define` / `inherit` / `use` produce `PseudoGenome` values; only `develop(G, K_D)` introduces randomness, and only via the split scheme above.
- `A-80` (absolute) and `A*0.08` (fractional) affect **declared `n_neurons`** before `develop`; they do not inject additional PRNG draws. Two genomes differing only by an `A-80` tweak developed with the same `K_D` will differ only by that layer's count (and its downstream position sampling under `K_S`), not by a different jitter pattern.
- Future optimization over `K_D` (sampling an ensemble `{develop(G, K) : K ∈ KS}`) is the intended use-case for tolerance-driven variation, not a grammar change.

---

## 9. Provenance hash — `genome_rules_hash`

### 9.1 Definition (implemented, `jaxfne/jdna/genome.py:131-141`)

```python
def genome_rules_hash(genome: PseudoGenome) -> str:
    payload = {
        "schema_version": genome.schema_version,
        "name": genome.name,
        "development_parameters": dict(genome.development_parameters),
        "areas": [_area_to_dict(a) for a in genome.areas],
        "area_connections": [dict(c) for c in genome.area_connections],
    }
    blob = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
```

**Excludes** `description` (free prose, not a rule). **Includes** everything that affects `develop` output: `schema_version`, `name`, `development_parameters`, all `AreaGenome`/`LayerGenome` rules, and `area_connections`. Canonicalization sorts keys and normalizes tuples→lists so `load(save(G))` preserves the hash.

### 9.2 Role in compact grammar

- Every `define` / `inherit` genome, after `validate_genome`, has a stable `genome_rules_hash`. Two genomes that desugar to the same rule set have the same hash regardless of syntactic sugar used.
- The developed `NeuronalTensor.provenance` records:
  ```python
  {
    "genome": genome.name,
    "genome_sha256": genome_rules_hash(genome),
    "schema_version": genome.schema_version,
    "development_seed": int(K_D),
    "development_parameters": dict(effective_params),
    "phenotype_sha256": phenotype_sha256(tensor),  # hash of tensor without provenance
  }
  ```
  `phenotype_sha256` is recomputed after assembly; `provenance` is excluded from `save_neuronal_tensor` JSON (see `tests/test_jdna_truth_gate.py:89-90`).
- For `inherit` chains, provenance records only the **final** child genome's hash (lineage is caller-side metadata if needed). For `merge_neuronal_tensors` compositions ( §7 ), the merged tensor's `provenance` is `None`; the individual input provenances are the audit trail.

### 9.3 No new hash

This preregistration proposes **no new hash function**, no `compact_grammar_hash`, and no salt. Reuse of `genome_rules_hash` is intentional so existing tooling (`declared_constraints`, truth-gate tests, `save_pseudogenome`) continues to work.

---

## 10. Examples (3 required)

All examples are **illustrative desugar** — they would be legal inputs to a future compiler but are not executable in `0.4.17` (no parser shipped). Each block shows the compact surface and the Python-equivalent it desugars to, plus the expected `develop` behavior.

### 10.1 Example 1 — `define` base column (no inheritance)

```text
# compact grammar
define base-column {
  development { fraction_jitter_sigma = 0.01 }

  area V1 pose { plane:"xy", rotation_deg:0.0, translation:[0.0, 0.0, 0.0] } {
    layer L4 n=800 depth=[0.2, 0.5] fractions {E:0.80, PV:0.15, SST:0.05} tolerance {E:[0.75,0.85], PV:[0.10,0.20], SST:[0.03,0.08]}
             geometry {distribution:"uniform_random", x_range:[0.0,1.0], y_range:[0.0,1.0]}
    layer L2/3 n=200 depth=[0.0, 0.2] fractions {E:0.75, PV:0.25} tolerance {E:[0.70,0.80], PV:[0.20,0.30]}
    connect L4 E -> L2/3 E mechanism:"AMPA"
    connect L4 PV -> L2/3 PV mechanism:"GABA"
  }
}
```

Equivalent implemented construction (what the future compiler would emit):

```python
import jaxfne as jtfne
from jaxfne.jdna.genome import PseudoGenome, AreaGenome, LayerGenome, ConnectionRuleGenome

base = PseudoGenome(
    name="base-column",
    description="",
    development_parameters={"fraction_jitter_sigma": 0.01},
    areas=[
        AreaGenome(
            name="V1",
            pose={"plane":"xy","rotation_deg":0.0,"translation":(0.0,0.0,0.0),"value_tag":"relative"},
            layers=[
                LayerGenome(name="L4", n_neurons=800, depth_band=(0.2,0.5),
                    cell_type_fractions={"E":0.80,"PV":0.15,"SST":0.05},
                    fraction_tolerance={"E":(0.75,0.85),"PV":(0.10,0.20),"SST":(0.03,0.08)},
                    geometry={"distribution":"uniform_random","x_range":(0.0,1.0),"y_range":(0.0,1.0),"value_tag":"relative"}),
                LayerGenome(name="L2/3", n_neurons=200, depth_band=(0.0,0.2),
                    cell_type_fractions={"E":0.75,"PV":0.25},
                    fraction_tolerance={"E":(0.70,0.80),"PV":(0.20,0.30)}),
            ],
            inter_connections=[
                ConnectionRuleGenome(source_layer="L4", source_neuron_type="E",  target_layer="L2/3", target_neuron_type="E",  mechanism="AMPA"),
                ConnectionRuleGenome(source_layer="L4", source_neuron_type="PV", target_layer="L2/3", target_neuron_type="PV", mechanism="GABA"),
            ],
        )
    ],
)
jtfne.jdna.genome.validate_genome(base)
h0 = jtfne.jdna.genome.genome_rules_hash(base)  # provenance §9
t0 = jtfne.develop(base, seed=0)                 # K_D=0,  §8
# t0.areas[0].layers[0].n_neurons == 800 exactly; counts for E/PV/SST within tolerance bands via jitter+projection+largest-remainder
```

**Validations that must pass:** `Σ fractions =1` per layer, `sum(lo)=0.88 ≤1≤ sum(hi)=1.13` feasible, `0≤lo<hi≤1` for depth bands, `n>0`.

### 10.2 Example 2 — `inherit` with `A-80` absolute override (deep merge)

```text
# compact grammar — child overrides one layer's absolute count, everything else inherited
inherit sparse-L4 from base-column {
  area V1 {
    layer L4 n=80   # shorthand for V1.L4-80; replaces only n_neurons
    # fractions, tolerance, geometry, depth, inter_connections all inherited
  }
}
# Equivalent use-form sugar (also legal future syntax):
# use base-column V1.L4-80 as sparse-L4
```

Equivalent:

```python
from jaxfne.jdna.genome import PseudoGenome, AreaGenome, LayerGenome
import copy

# deep merge per §6.2
parent = base  # from Example 1
# merge: V1.L4.n_neurons = 80, all other fields retained
sparse_layers = []
for lyr in parent.areas[0].layers:
    if lyr.name == "L4":
        sparse_layers.append(LayerGenome(
            name=lyr.name, n_neurons=80, depth_band=lyr.depth_band,
            cell_type_fractions=dict(lyr.cell_type_fractions),
            fraction_tolerance=dict(lyr.fraction_tolerance),
            geometry=dict(lyr.geometry), relative_sizes=dict(lyr.relative_sizes)))
    else:
        sparse_layers.append(lyr)

sparse = PseudoGenome(
    name="sparse-L4",
    description=parent.description,
    development_parameters=dict(parent.development_parameters),
    areas=[AreaGenome(name="V1", layers=tuple(sparse_layers),
                      inter_connections=tuple(parent.areas[0].inter_connections),
                      pose=dict(parent.areas[0].pose))],
    area_connections=tuple(parent.area_connections),
)
jtfne.jdna.genome.validate_genome(sparse)
assert jtfne.jdna.genome.genome_rules_hash(sparse) != jtfne.jdna.genome.genome_rules_hash(parent)
t_sparse = jtfne.develop(sparse, seed=0)
assert sum(l.n_neurons for l in t_sparse.areas[0].layers) == 280  # 80 + 200, area total mutated by this absolute
# L4 counts: E≈64, PV≈12, SST≈4 (σ=0) or within [60,68],[8,16],[2,6] (σ=0.01 with bands)
```

**What `A-80` means here ( §5.2 ):** `V1.L4` is exactly 80 neurons — not "remove 80", not "80 %". Tolerance still governs **cell-type split** inside that 80. Depth band `[0.2,0.5]` and `x_range/y_range` unchanged. `genome_rules_hash` changes, so provenance distinguishes `base-column` from `sparse-L4` even at the same `K_D`.

### 10.3 Example 3 — `use` with `A*0.08` fraction tweak (denominator = enclosing area)

```text
# compact grammar — fractional tweak, area-total-preserving
use base-column V1.L4*0.08 as base-frac08

# With base-column: V1 { L4 n=800, L2/3 n=200 } => N_V1 = 1000
# Denominator per §5.3: N_enclosing = N_V1(G0) = 1000
# Raw target L4 = 1000 * 0.08 = 80
# Preserving mode: L4=80, remaining 920 distributed to L2/3 (only sibling) => L2/3=920
# Result: V1 { L4 80, L2/3 920 }  sum=1000 preserved
```

Equivalent:

```python
# desugar of V1.L4*0.08 under N_V1=1000 preserving mode
N_enclosing = sum(l.n_neurons for l in base.areas[0].layers)  # 1000
raw_L4 = N_enclosing * 0.08                                   # 80.0
# largest-remainder over fractional subset {L4} vs sibling pool {L2/3}
# => deterministic integer counts:
frac08 = PseudoGenome(
    name="base-frac08",
    description=base.description,
    development_parameters=dict(base.development_parameters),
    areas=[AreaGenome(
        name="V1",
        pose=dict(base.areas[0].pose),
        layers=(
            LayerGenome(name="L4",   n_neurons=80,  depth_band=(0.2,0.5),
                        cell_type_fractions=dict(base.areas[0].layers[0].cell_type_fractions),
                        fraction_tolerance=dict(base.areas[0].layers[0].fraction_tolerance),
                        geometry=dict(base.areas[0].layers[0].geometry)),
            LayerGenome(name="L2/3", n_neurons=920, depth_band=(0.0,0.2),
                        cell_type_fractions=dict(base.areas[0].layers[1].cell_type_fractions),
                        fraction_tolerance=dict(base.areas[0].layers[1].fraction_tolerance)),
        ),
        inter_connections=tuple(base.areas[0].inter_connections),
    )],
)
jtfne.jdna.genome.validate_genome(frac08)
assert frac08.areas[0].layers[0].n_neurons == 80
assert sum(l.n_neurons for l in frac08.areas[0].layers) == N_enclosing == 1000
t_frac08 = jtfne.develop(frac08, seed=0)
# composition after develop ( §7 ) — e.g. stack two fractional variants
t_other = jtfne.develop(base, seed=1)
stacked = jtfne.merge_neuronal_tensors([t_frac08, t_other], name="stacked-frac08-vs-base")
# stacked.areas = [V1 (from frac08), V1_1 (renamed collision from base)] per merge_neuronal_tensors §7.2
```

**Why 80 both times?** Example 2's `A-80` and Example 3's `A*0.08` coincide numerically because `0.08 * 1000 = 80`. They are **not semantically equal**: `A-80` is absolute (stays 80 if the area total changes), `A*0.08` is relative (scales if `N_V1` changes — e.g. if a prior `inherit` grew `L2/3` to 400, then `N_V1=1200` and `A*0.08` would give `96`, while `A-80` still gives `80`). The provenance hashes differ accordingly.

**Multi-tweak atomicity note (for a future test, not for 0.4.17):**

```text
use base-column V1.L2/3*0.20 V1.L4*0.30 as two-frac
# Both fractions evaluated against the same N_V1(G0)=1000 snapshot (§5.3.7):
# raw L2/3=200, raw L4=300, remaining 500 for any uninvolved layers (none here),
# largest-remainder fills to integer with Σ=1000 preserved atomically.
```

---

## 11. Validation, testing, and H11 compliance

### 11.1 Status of this document

- This file is **not** an executable spec. No `jaxfne/jdna/*.py` file is changed. No new test file is added in this task.
- A future implementation task will add a compiler module (proposed location `jaxfne/jdna/compact.py`, not created here) that implements the PEG, desugar, and error taxonomy of §4–§6, and a test file that validates it.

### 11.2 H11 — isolated `tmp_path` testing (normative for future tests)

Any test that exercises the grammar proposed here must satisfy:

1. **Self-generate outputs.** The test generates any genome JSON / tensor JSON / figure into the `tmp_path` fixture directory it receives. It never reads a pre-existing `artifacts/etudes/jdna-compact-grammar-*/` file as an input unless that file is a committed, tracked fixture explicitly declared as a test input.
2. **No reliance on gitignored artifacts.** Tests do not depend on untracked, gitignored, or locally-generated PNGs/JSONs. The visual review pattern in `tests/test_jdna_truth_gate.py:106-124` (`tmp_path / "jdna_visual_review.png"` via subprocess) is the exemplar.
3. **Roundtrip checks are isolated.** `genome_rules_hash` and `phenotype_sha256` assertions run on in-memory objects or on files written to `tmp_path`; a fresh clone without any prior `outputs/` or `artifacts/etudes/*.json` must still pass.
4. **Explicit invalid-input tests.** Per H5, the future test suite must include adversarial invalid and boundary inputs: duplicate area/layer names, `tolerance` for undeclared cell type, `sum(lo)>1` joint infeasibility, `depth_band` out-of-range, bare `Area*frac` with multi-area genomes, and `A*1.5` out-of-domain — not only the canonical happy paths of §10.

### 11.3 Future compiler acceptance criteria (not executed here)

| Check | Receipt |
|-------|---------|
| `define` → `PseudoGenome` → `validate_genome` passes, `genome_rules_hash` stable across `save/load` | `tmp_path` roundtrip test |
| `inherit` deep merge per §6.2–§6.4, tolerance override field-granular, joint feasibility rechecked | unit tests for each row of §6.6 error taxonomy |
| `A-80` sets `n_neurons==80` exactly, `A*0.08` obeys denominator `N_enclosing = N_area(G0)` preserving mode, largest-remainder sums to area total | parametrized tests over `N_area ∈ {100, 1000, 10000}` with both forms, assert `Σ n == N_area` (§5.3) |
| `develop(G, K_D)` determinism: same clone, same `K_D` → `phenotype_sha256` equal; different `K_D` → (when `sigma>0`) different feasible phenotype within bands | determinism test + band-containment test |
| `merge_neuronal_tensors([develop(G1,K1), develop(G2,K2)])` preserves renamed provenances, area-collision suffix rule | composition test asserting `stacked.areas[1].name == "V1_1"` |
| No `construct`/`simulate`/`neuronal_tensor.py` import of `jaxfne.jdna` | grep test mirroring `test_no_jdna_branches_in_construct_simulate` |
| H11 isolation: all generated files in `tmp_path` | `pytest --override-ini="addopts="` run from clean worktree |

---

## 12. Deferred decisions (recorded, not decided)

- **Delete syntax for tolerance/pose keys** (e.g. `tolerance.E = null` to remove a tolerance interval rather than redeclaring the whole map). Deferred; for now the whole map is redeclared without the entry (§6.3.3).
- **Bare `Area*frac` with multi-area genomes** — preregistration marks this as discouraged/linted; a future version may forbid it outright or define denominator as global total.
- **Depth-band strict mode** (non-overlapping, gap-free tiling of `[0,1]` as a hard error vs warning). Current `validate_genome` only checks per-layer `0≤lo<hi≤1`; strict tiling remains a style invariant (§6.4).
- **`area_total` preserving vs mutating default** for isolated `A-80` (§5.3.3). Preregistration defaults to **preserving** except the single-layer-single-tweak mutate allowance; a future compiler flag `area_total="preserve"|"mutate"` will make it explicit.
- **LESs-like arithmetic expressions** (e.g. `A*0.08 + 10`). Explicitly not in grammar — only bare `TargetRef-INT` and `TargetRef*FLOAT`.

---

## 13. Traceability

| Concept | Lives in |
|---------|----------|
| `PseudoGenome`, `AreaGenome`, `LayerGenome`, `ConnectionRuleGenome` | `jaxfne/jdna/genome.py:42-112` |
| `genome_rules_hash`, `phenotype_sha256`, `_canonical` | `jaxfne/jdna/genome.py:120-149` |
| `validate_genome` (+ joint feasibility, per-layer invariants) | `jaxfne/jdna/genome.py:317-447` |
| `declared_constraints` (machine-readable bands) | `jaxfne/jdna/genome.py:449-471` |
| `_project_box_simplex` / `_allocate_counts` (jitter+project+largest-remainder) | `jaxfne/jdna/genome.py:479-571` |
| `develop(G, K_D)` (PRNG split per area/layer) | `jaxfne/jdna/genome.py:604-734` |
| `merge_neuronal_tensors` (area rename, pose override, provenance handling) | `jaxfne/neuronal_tensor.py:539-625` |
| `K_D ≠ K_S ≠ K_A` doctrine | `docs/guides/jdna.md:114-124` |
| Root surface + truth gate (no JDNA in construct/simulate) | `tests/test_jdna_truth_gate.py` |
| This preregistration | `artifacts/etudes/jdna-compact-grammar-preregistration.md` (this file) |

---

## 14. Sign-off

- **Author:** Private étude draft — no `jaxfne/*.py` change, no public surface change, no frozen receipt.
- **Review:** Requires CODE authority sign-off before any compiler implementation is branched.
- **Next step (if approved):** Open a CODE-track task to implement `jaxfne/jdna/compact.py` + `tests/test_jdna_compact_grammar.py` (isolated `tmp_path` per H11), with the grammar in this file as the spec. No `jaxfne/jdna/genomes/*.json` canonical genome is modified by that task without a separate review.

*Explicitly marked: **Private étude — not a publication artifact, not a release deliverable, not a protocol execution. No `jaxfne/jdna/*.py` change in this task. Future tests for this grammar must be H11-isolated (`tmp_path`).***
