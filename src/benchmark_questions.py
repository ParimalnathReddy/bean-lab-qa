#!/usr/bin/env python3
"""
Shared benchmark question bank for the Bean Lab RAG QA system.

Used by both eval_qa.py (coarse keyword-rubric scoring) and eval_ragas.py
(LLM-judged RAGAS metrics: faithfulness, answer relevancy, context precision,
context recall) so the two evaluators always run against the exact same
question set — no drift between them.

75 questions total, across 7 categories:
  direct_lookup   (10) — fact stated verbatim in a single paper
  cross_section   (10) — fact assembled from multiple papers/sections
  inference       (10) — answer follows logically but isn't stated directly
  critique        (10) — limitations, gaps, or criticisms
  multi_part      (10) — compound questions requiring multiple sub-answers
  adversarial     (10) — designed to probe hallucination: fake gene IDs, fake
                         cultivars, and out-of-scope topics phrased like bean
                         questions. The "correct" answer is honest uncertainty,
                         not a confident (necessarily fabricated) fact.
  multi_hop       (15) — require synthesizing across 3+ distinct sub-topics /
                         papers, deeper than cross_section's 2-paper asks.

IMPORTANT CAVEATS — read before trusting results built on this file:

1. ground_truth (Change 11): 11 of 75 questions now have it populated — all
   7 direct_lookup and 4 of 10 cross_section questions with clean, specific
   corpus support (DL-01, DL-03, DL-04, DL-07, DL-08, DL-09, DL-10, CS-01,
   CS-02, CS-09, CS-10). The remaining 64 are still None, deliberately, not
   an oversight: each populated answer was verified against real chunk text
   pulled from data/processed_chunks.json (21,876 chunks, 964 papers) via
   targeted keyword search — never invented from general domain knowledge —
   and several DL/CS questions that looked promising on paper (DL-02 soil
   pH, DL-05 herbicides, DL-06 deficit-irrigation WUE, DL-06's cousin CS-03
   P/N-fixation, CS-04 heat x drought, CS-05 genetic vs. agronomic gains,
   CS-06 canopy/white mold, CS-07 yield vs. market class, CS-08 Al
   toxicity) turned up no clean, specific, bean-relevant match in this
   corpus after real searching and were deliberately left None rather than
   answered from outside knowledge — this crop-science-journal corpus
   covers some agronomy-extension-style facts (optimal soil pH, herbicide
   product names) thinly or not at all. CS-02's ground_truth is itself an
   example of this discipline applied within a single answer: it states
   what the retrieved maize-bean intercrop yield data actually shows and
   explicitly notes the nitrogen-fixation angle the original question asked
   about was NOT found in this source, rather than inventing it. The 10
   inference/10 critique/10 multi_part/10 adversarial/15 multi_hop
   questions were left entirely alone — untouched by this change. Where
   ground_truth is populated, eval_ragas.py now also computes
   context_precision and context_recall for that question, in addition to
   the three reference-free metrics; where it's still None, only the
   reference-free metrics run, exactly as before.

2. The 25 direct_lookup/cross_section/inference/critique/multi_part questions
   beyond the original 25 (i.e. *-06 through *-10), plus all 10 adversarial
   and all 15 multi_hop questions, were authored in the same style and domain
   as the original 25 — but, same as the originals, were NOT verified against
   the actual corpus content. Some may turn out to be unanswerable from what's
   actually indexed, or to have a rubric that doesn't quite match what the
   papers say. Treat rubric_coverage as a rough signal, and expect to revise
   entries once you can see how the real system answers them.

3. Adversarial questions (AD-*) reference deliberately fictional gene IDs,
   loci, and cultivar names, invented to look plausible in Phaseolus
   vulgaris nomenclature (e.g. "Phvul.011G999950") without being real
   accessions. They test whether the system hedges honestly versus
   hallucinating specifics. `expect_low_confidence=True` marks these — a
   "good" system response acknowledges the term/cultivar isn't found in the
   sources rather than inventing data about it. This is also exactly where
   RAGAS's faithfulness metric should shine over keyword rubrics: a
   fabricated answer about a fictional gene will have near-zero faithfulness
   (no supporting context exists), regardless of how confident or
   keyword-rich the fabrication sounds.
"""

from typing import Dict, List, Optional, TypedDict


class BenchmarkQuestion(TypedDict):
    id: str
    category: str
    question: str
    rubric: List[str]
    ground_truth: Optional[str]
    expect_low_confidence: bool


def _q(
    id: str,
    category: str,
    question: str,
    rubric: List[str],
    expect_low_confidence: bool = False,
    ground_truth: Optional[str] = None,  # Change 11: verified against the
    # actual indexed corpus (data/processed_chunks.json) via keyword search,
    # not invented — see this module's docstring, item 1, for why every
    # other question still has None here.
) -> BenchmarkQuestion:
    return {
        "id": id,
        "category": category,
        "question": question,
        "rubric": rubric,
        "ground_truth": ground_truth,
        "expect_low_confidence": expect_low_confidence,
    }


BENCHMARK: List[BenchmarkQuestion] = [

    # ── 1. Direct lookup (10) ─────────────────────────────────────────────────
    _q("DL-01", "direct_lookup",
        "What nitrogen fixation rates have been reported for common bean (Phaseolus vulgaris) varieties?",
        ["specific numerical fixation rates (kg N/ha or % N derived from atmosphere)",
         "mention of Rhizobium or nodulation",
         "at least one cultivar or experimental condition"],
        ground_truth=(
            "In a field comparison of 12 dry bean lines using 15N isotope methods "
            "(St. Clair et al., Crop Science 1988), the percentage of seed nitrogen "
            "derived from atmospheric N2 fixation (% Ndfa) ranged from 26% to 55% "
            "for most lines, with the cultivar Sanilac a clear outlier at only about "
            "5%. Higher N2-fixing lines had significantly more fixed N2 in their "
            "seed, and seed yield was highly rank-correlated with plant N2 fixed "
            "(rs = 0.98, P < 0.01), indicating N2-fixing ability and seed yield can "
            "be improved together through selection."
        )),
    _q("DL-02", "direct_lookup",
        "What soil pH range is considered optimal for common bean production?",
        ["pH range (typically 6.0–7.0)",
         "reference to nutrient availability or root health"]),
    _q("DL-03", "direct_lookup",
        "What are the symptoms of bean rust caused by Uromyces appendiculatus?",
        ["description of pustules or uredinia",
         "leaf or stem symptoms",
         "mention of sporulation or color"],
        ground_truth=(
            "Bean rust reactions are commonly scored on the Stavely et al. (1983) "
            "1-to-6 scale used across multiple common bean cultivar registrations: "
            "grade 1 shows no visible symptoms; grades 2/2+ are necrotic or "
            "chlorotic flecks under 1.0 mm without sporulation (a resistant, "
            "hypersensitive reaction); grade 3 shows tiny sporulating uredinia "
            "(pustules) under 0.3 mm; and grades 4 through 6 are sporulating "
            "pustules of increasing size, from 0.3-0.5 mm up to larger than "
            "0.8 mm in diameter, as the reaction becomes fully susceptible. "
            "Multiple pustule sizes can occur on the same leaf."
        )),
    _q("DL-04", "direct_lookup",
        "What seed yield advantage has been reported for indeterminate over determinate bean varieties under field conditions?",
        ["numerical yield comparison (% or kg/ha)",
         "mention of growth habit type (Type I, II, III, IV)",
         "at least one study or environment cited"],
        ground_truth=(
            "In a CIAT field trial at Quilichao, Colombia comparing determinate "
            "(Type I) with indeterminate erect (Type II) and indeterminate "
            "prostrate (Type III) common bean, indeterminate types generally "
            "out-yielded the determinate type across plant densities: yield of "
            "Type I plants only approached that of Type II at the highest density "
            "tested (30 plants/m2), while Types II and III yields were already "
            "maximized at a lower density of about 23 plants/m2. Indeterminate "
            "cultivars tended to compensate for yield over a wide range of plant "
            "densities, whereas determinate cultivars increased yield per unit "
            "area mainly by increasing density."
        )),
    _q("DL-05", "direct_lookup",
        "What herbicides are commonly used for weed control in dry bean production?",
        ["at least two herbicide names (e.g., S-metolachlor, imazethapyr, fomesafen)",
         "mention of application timing or target weed species"]),
    _q("DL-06", "direct_lookup",
        "What water use efficiency values have been reported for common bean under deficit irrigation?",
        ["numerical water use efficiency value (e.g., kg yield per mm water, or a WUE ratio)",
         "mention of deficit irrigation or water stress treatment",
         "reference to a specific study or experimental condition"]),
    _q("DL-07", "direct_lookup",
        "What is the typical days-to-flowering range reported for determinate common bean cultivars?",
        ["numerical days-to-flowering range",
         "mention of determinate growth habit",
         "reference to environmental or photoperiod effect"],
        ground_truth=(
            "Days to flowering in common bean field trials at CIAT sites in "
            "Colombia (Palmira and Popayan) has been observed to range from "
            "roughly 25 to 55 days across genotypes and environments (Sexton et "
            "al., Crop Science 1994). This variation is strongly driven by "
            "photoperiod response: most tropical common bean landraces are "
            "sensitive to long daylengths, delaying flowering, while some "
            "genotypes (especially in the Mesoamerican and Andean gene pools) "
            "are photoperiod-insensitive. Photoperiod sensitivity is controlled "
            "by a small number of genes (often reported as one to two) whose "
            "dominance behavior varies by genotype and test environment."
        )),
    _q("DL-08", "direct_lookup",
        "What protein content percentages have been reported for common bean seed?",
        ["numerical protein content percentage or range",
         "mention of cultivar or genotype variation",
         "reference to nutritional or seed quality context"],
        ground_truth=(
            "Seed protein content of common bean (Phaseolus vulgaris) usually "
            "ranges from about 18% to 30% (Mutschler and Bliss, Crop Science "
            "1981). A weak negative correlation between seed yield and protein "
            "concentration has been reported across breeding lines, meaning "
            "higher-protein selections tend to yield somewhat less, an important "
            "trade-off for breeding programs targeting nutritional quality."
        )),
    _q("DL-09", "direct_lookup",
        "What plant density or spacing recommendations have been reported for maximizing dry bean yield?",
        ["numerical plant density (plants/m2 or similar) or row spacing figure",
         "mention of planting arrangement",
         "reference to yield outcome"],
        ground_truth=(
            "In a CIAT field trial (Quilichao, Colombia) across three bean growth "
            "habits, yield of indeterminate erect and prostrate types (Types II "
            "and III) was maximized at a plant density of about 23 plants/m2, "
            "while nodes per m2 continued increasing up to the highest density "
            "tested, 30 plants/m2. The determinate type (Type I) only reached "
            "yield comparable to the indeterminate types at that highest density, "
            "consistent with determinate cultivars needing higher planting "
            "density to maximize yield per unit area, while indeterminate types "
            "compensate for yield over a wider range of densities."
        )),
    _q("DL-10", "direct_lookup",
        "What symptoms characterize common bacterial blight (CBB) caused by Xanthomonas in bean leaves?",
        ["description of water-soaked lesions or leaf spots",
         "mention of leaf margin or vein symptoms",
         "reference to bacterial ooze or angular lesions"],
        ground_truth=(
            "Common bacterial blight, caused by Xanthomonas campestris pv. "
            "phaseoli (and the fuscans variant), is a seed-borne bacterial "
            "disease causing severe yield losses in dry and snap bean. On the "
            "standard CBB rating scale used in resistance screening (van "
            "Schoonhoven and Pastor-Corrales, 1987), symptoms progress from an "
            "isolated small chlorotic zone around a necrotic lesion at the "
            "inoculation site, to chlorotic zones and lesions joining together "
            "as severity increases, to complete chlorosis of the affected leaf "
            "area and systemic infection in fully susceptible reactions."
        )),

    # ── 2. Cross-section synthesis (10) ───────────────────────────────────────
    _q("CS-01", "cross_section",
        "How has marker-assisted selection improved bean resistance to bean common mosaic virus (BCMV)?",
        ["identifies specific resistance genes or QTLs (e.g., I gene, bc-1, bc-3)",
         "describes how MAS is applied in breeding",
         "mentions genetic diversity or gene pools"],
        ground_truth=(
            "The recessive bc-3 gene confers resistance to all known strains of "
            "Bean Common Mosaic Virus. RAPD markers linked to the bc-3 locus "
            "(e.g., OAD1969, linked in coupling at about 1.9 cM, and OS1366, "
            "linked in repulsion at about 7.1 cM, identified via bulked "
            "segregant analysis) allow indirect, marker-assisted selection for "
            "this recessive resistance allele without needing to mechanically "
            "inoculate and progeny-test each plant against BCMV strains. "
            "Survey work tracing the bc-3 resistance allele's evolutionary "
            "origin placed it in the Mesoamerican gene pool. This let breeders "
            "introgress durable, broad-spectrum BCMV resistance into elite "
            "cultivars more efficiently than conventional inoculation-based "
            "selection."
        )),
    _q("CS-02", "cross_section",
        "How does intercropping common bean with maize affect bean yield and nitrogen fixation?",
        ["yield comparison (monoculture vs. intercrop)",
         "light or nutrient competition effects",
         "nitrogen fixation change (increase or decrease with rationale)"],
        ground_truth=(
            "A multi-year CIAT field trial evaluating maize-bean cropping "
            "patterns found maize yields were significantly higher in "
            "monoculture than when intercropped with bean — 4761 kg/ha in "
            "monoculture versus 4203 to 4324 kg/ha intercropped in the first "
            "experiment. The bean growth type mattered: a more competitive "
            "indeterminate climbing bean reduced maize yield more than a "
            "determinate bush-type bean, which had comparatively limited effect "
            "on maize yield. Retrieved sources in this corpus report yield and "
            "competition effects of maize-bean intercropping in detail but do "
            "not report a nitrogen-fixation-specific measurement for this "
            "particular intercrop combination."
        )),
    _q("CS-03", "cross_section",
        "What is the role of phosphorus availability in nitrogen fixation efficiency in common bean?",
        ["mechanism linking P to nodulation or nitrogenase activity",
         "threshold P level or fertilization trial results",
         "interaction with soil pH or organic matter"]),
    _q("CS-04", "cross_section",
        "How do drought stress and heat stress interact to affect bean pod set and seed filling?",
        ["separate effects of drought and heat on reproductive stage",
         "combined/interaction effects if reported",
         "mention of canopy temperature or water deficit measurement"]),
    _q("CS-05", "cross_section",
        "How have bean yield improvements been achieved through genetic improvement vs. agronomic management since 1990?",
        ["distinguishes genetic gain from management-based yield increases",
         "references breeding programs or variety releases",
         "references agronomic practices (fertilization, planting density, irrigation)"]),
    _q("CS-06", "cross_section",
        "How does canopy architecture interact with disease pressure to affect white mold incidence in dry bean?",
        ["description of canopy density or closure effect on microclimate",
         "link between humidity/microclimate and Sclerotinia infection",
         "mention of a management implication (row spacing, cultivar choice)"]),
    _q("CS-07", "cross_section",
        "How have breeding programs balanced selection for yield with selection for seed coat color and market class in common bean?",
        ["mention of trade-offs between yield and market/seed traits",
         "reference to specific market classes (e.g., navy, pinto, black)",
         "mention of breeding strategy (selection index, multi-trait selection)"]),
    _q("CS-08", "cross_section",
        "How does aluminum toxicity in acid soils interact with phosphorus deficiency to limit bean root development?",
        ["mechanism linking Al toxicity to root growth inhibition",
         "interaction with P availability or uptake",
         "reference to acid soil management or tolerant genotypes"]),
    _q("CS-09", "cross_section",
        "How do pre-flowering and post-flowering drought stress differentially affect bean pod number and seed weight?",
        ["distinguishes effects at different growth stages",
         "quantitative or comparative yield component impact",
         "reference to a specific study or trial"],
        ground_truth=(
            "In dry bean, excessive abortion of flowers, young pods, and seeds "
            "occurs from drought stress applied during pre-flowering (10 to 12 "
            "days before anthesis) as well as during later reproductive periods, "
            "and moderate to severe drought stress at these stages reduces "
            "biomass, seed yield, harvest index, pod number, seed number, seed "
            "weight, and days to maturity, with the magnitude depending on "
            "timing and intensity of the stress. In a field trial of three "
            "landraces and 13 cultivars under intermittent drought stress at "
            "Kimberly, Idaho (2003-2004), mean seed yield was reduced by about "
            "62% relative to non-stressed conditions."
        )),
    _q("CS-10", "cross_section",
        "How has integrated pest management combining resistant varieties and insecticide use addressed whitefly-transmitted virus problems in bean?",
        ["mention of resistant variety deployment",
         "mention of insecticide or vector control practice",
         "reference to virus (e.g., BGMV) management outcome"],
        ground_truth=(
            "Bean Golden Mosaic Virus (BGMV), transmitted by whitefly (Bemisia "
            "tabaci and B. argentifolii), is described as the most serious "
            "common bean disease in the lowland tropics of the Americas. The "
            "dominant management strategy documented in this corpus is "
            "resistant-cultivar deployment rather than insecticide-centered "
            "control: most BGMV resistance breeding traces back to the black "
            "bean landrace 'Porrillo Sintetico' from El Salvador, and derived "
            "lines such as 'Dorado' (DOR 364) show quantitative resistance — "
            "reduced yellow mosaic symptoms and superior yield. RAPD/QTL "
            "marker-assisted selection (via bulked segregant analysis) has been "
            "used to accelerate breeding for BGMV resistance, in some cases "
            "combined with common bacterial blight resistance in the same "
            "mapping populations. Retrieved sources in this corpus emphasize "
            "genetic resistance over insecticide-based vector control as the "
            "primary management approach."
        )),

    # ── 3. Inference / cause-effect (10) ──────────────────────────────────────
    _q("IN-01", "inference",
        "Why is pyramiding rust resistance genes from both Middle American and Andean gene pools considered important for bean breeding programs?",
        ["explanation of gene pool diversity (Middle American vs. Andean)",
         "reason for combining genes (broader spectrum resistance or durability)",
         "mention of Uromyces races or pathogen variation"]),
    _q("IN-02", "inference",
        "Why might tepary bean (Phaseolus acutifolius) be more drought-tolerant than common bean at the physiological level?",
        ["physiological mechanisms (deeper roots, reduced transpiration, osmotic adjustment)",
         "comparison with common bean",
         "at least one study or trait measurement cited"]),
    _q("IN-03", "inference",
        "What explains why biological nitrogen fixation in common bean is often insufficient to meet crop nitrogen demand?",
        ["factors limiting BNF (carbon cost, soil mineral N, P deficiency)",
         "quantitative shortfall mentioned (kg N/ha fixed vs. required)",
         "interaction with Rhizobium strain or soil conditions"]),
    _q("IN-04", "inference",
        "Why does early-season drought have a more severe effect on bean yield than late-season drought?",
        ["impact on flowering or pod set (vs. seed filling)",
         "critical period concept",
         "supporting data or study result"]),
    _q("IN-05", "inference",
        "Why might low-input farming systems in sub-Saharan Africa benefit more from improved bean varieties than high-input systems?",
        ["yield gap argument (genetic potential vs. realized yield)",
         "role of disease resistance or adaptation in low-input contexts",
         "comparison of input-response between systems"]),
    _q("IN-06", "inference",
        "Why might combining physical seed coat resistance traits with chemical seed treatments provide more durable protection against seed-borne pathogens than either approach alone?",
        ["explains complementary mechanisms (physical vs. chemical)",
         "reasoning about durability or resistance breakdown",
         "reference to a seed-borne pathogen context"]),
    _q("IN-07", "inference",
        "Why do determinate bean varieties often show more stable yields across environments than indeterminate varieties, despite lower yield potential?",
        ["explains growth habit stability mechanism",
         "trade-off between potential and stability (genotype-by-environment interaction)",
         "supporting study or data reference"]),
    _q("IN-08", "inference",
        "What might explain why nitrogen fertilization sometimes reduces biological nitrogen fixation rates in common bean?",
        ["mechanism: N fertilizer suppresses nodulation or nitrogenase activity",
         "reasoning about plant carbon allocation trade-off",
         "reference to a study or data"]),
    _q("IN-09", "inference",
        "Why could genotype-by-environment interaction complicate the identification of universally superior drought-tolerant bean lines?",
        ["explanation of the GxE concept",
         "implication for multi-environment trial design",
         "reference to a specific finding"]),
    _q("IN-10", "inference",
        "Why might early-maturing bean varieties be preferentially adopted in regions with short rainy seasons despite lower yield potential?",
        ["reasoning linking maturity to seasonal water availability",
         "trade-off between maturity and yield",
         "reference to adoption or agronomic context"]),

    # ── 4. Critique / limitations (10) ────────────────────────────────────────
    _q("CR-01", "critique",
        "What are the known challenges in breeding white mold (Sclerotinia sclerotiorum) resistance in dry beans?",
        ["complexity of host-pathogen interaction",
         "environmental variation in disease expression",
         "lack of complete resistance in germplasm",
         "difficulty of field screening"]),
    _q("CR-02", "critique",
        "What methodological limitations affect the measurement of nitrogen fixation in field bean experiments?",
        ["15N isotope dilution or acetylene reduction assay limitations",
         "spatial or temporal variability in field trials",
         "reference plant selection issues"]),
    _q("CR-03", "critique",
        "What gaps remain in understanding the genetic basis of drought tolerance in common bean?",
        ["complexity of quantitative inheritance",
         "environment-by-genotype interaction",
         "limited marker-trait associations or QTL stability"]),
    _q("CR-04", "critique",
        "What are the limitations of using yield trials as the primary method for evaluating drought tolerance in bean breeding programs?",
        ["confounding factors in field yield trials",
         "inconsistency of drought timing across environments",
         "need for physiological or secondary traits"]),
    _q("CR-05", "critique",
        "What are the barriers to adopting improved bean varieties among smallholder farmers in developing countries?",
        ["seed system or seed access issues",
         "market or cultural preference factors",
         "input cost or risk aversion"]),
    _q("CR-06", "critique",
        "What are the limitations of relying on visual disease scoring scales for evaluating rust resistance in breeding trials?",
        ["subjectivity or rater variability",
         "resolution/precision limitations of ordinal scales",
         "alternative or complementary methods suggested"]),
    _q("CR-07", "critique",
        "What challenges limit the transferability of QTLs identified for drought tolerance across different bean growing regions?",
        ["genetic background or population-specific effects",
         "environment-specific QTL expression",
         "need for validation across environments"]),
    _q("CR-08", "critique",
        "What are the shortcomings of single-location yield trials for recommending bean varieties to farmers?",
        ["lack of environmental representativeness",
         "genotype-by-environment interaction not captured",
         "need for multi-location or multi-year testing"]),
    _q("CR-09", "critique",
        "What limitations affect the use of molecular markers for marker-assisted selection in common bean breeding programs?",
        ["cost or infrastructure barriers",
         "marker-trait association reliability across populations",
         "practical breeding program integration challenges"]),
    _q("CR-10", "critique",
        "What are the criticisms of relying solely on acetylene reduction assays to estimate nitrogen fixation in field studies?",
        ["indirect measurement or conversion factor uncertainty",
         "temporal or spatial variability not captured",
         "comparison to isotope-based methods"]),

    # ── 5. Multi-part questions (10) ──────────────────────────────────────────
    _q("MP-01", "multi_part",
        "What are the most effective fungicides for bean rust management, how do they work mechanistically, and what resistance risks do they pose?",
        ["names at least two effective fungicide classes or active ingredients",
         "explains mode of action (sterol inhibition, respiration, etc.)",
         "addresses fungicide resistance risk or resistance management"]),
    _q("MP-02", "multi_part",
        "Compare the drought tolerance mechanisms of tepary bean and common bean, and explain the implications for introgression breeding programs.",
        ["mechanism comparison (tepary vs. common bean physiology)",
         "crossability barriers or reproductive isolation issues",
         "practical implications for breeding (backcrossing, marker selection)"]),
    _q("MP-03", "multi_part",
        "What soil amendments and inoculants have been used to enhance nitrogen fixation in common bean, what are their effects on yield, and what factors limit their adoption?",
        ["specific amendments or inoculants (Rhizobium, P fertilizer, organic matter)",
         "yield or fixation response data",
         "adoption barriers (cost, availability, farmer knowledge)"]),
    _q("MP-04", "multi_part",
        "How does common bacterial blight (CBB) spread in bean fields, what resistance mechanisms have been identified, and which breeding strategies are most effective for durable resistance?",
        ["describes disease spread pathway (seed, rain splash, insects)",
         "identifies resistance QTLs or genes in Middle American or Andean germplasm",
         "discusses durability strategy (pyramiding, multilines, rotation)"]),
    _q("MP-05", "multi_part",
        "What are the nutritional benefits of common beans for human health, what processing methods improve their bioavailability, and what genetic variation exists for key nutritional traits?",
        ["nutritional content (protein, iron, zinc, fiber, polyphenols)",
         "processing effects (soaking, cooking, fermentation) on anti-nutrients",
         "genetic variability in bean accessions for nutritional traits"]),
    _q("MP-06", "multi_part",
        "What role does Rhizobium strain selection play in nitrogen fixation efficiency, how is strain compatibility assessed, and what limits farmer adoption of inoculant technology?",
        ["strain-genotype compatibility or specificity",
         "assessment methods (nodulation count, competitiveness)",
         "adoption barriers (cost, storage, awareness)"]),
    _q("MP-07", "multi_part",
        "What are the main viral diseases affecting common bean, how are they transmitted, and what resistance breeding strategies have been used?",
        ["names at least two viruses (e.g., BCMV, BGMV)",
         "transmission mode (seed-borne, whitefly, aphid)",
         "resistance gene or breeding strategy mentioned"]),
    _q("MP-08", "multi_part",
        "How does intercropping bean with cereals affect light interception and nutrient competition, and what management practices mitigate negative effects?",
        ["light competition mechanism",
         "nutrient competition mechanism",
         "management mitigation (density, fertilization, arrangement)"]),
    _q("MP-09", "multi_part",
        "What genetic and environmental factors determine cooking time variability in common bean, and what practical implications does this have for consumers and breeders?",
        ["genetic factor (seed coat characteristics, genotype)",
         "environmental or storage factor (hard-to-cook phenomenon)",
         "practical implication (breeding target, consumer acceptance)"]),
    _q("MP-10", "multi_part",
        "What are the primary abiotic stresses limiting common bean production in sub-Saharan Africa, how do they interact, and what breeding priorities have been proposed to address them?",
        ["names at least two abiotic stresses (drought, heat, low phosphorus)",
         "interaction between stresses",
         "breeding priority or strategy mentioned"]),

    # ── 6. Adversarial (10) — probes hallucination, not fact recall ──────────
    # Fake gene/locus names (Phaseolus vulgaris-style IDs that don't correspond
    # to any real, known accession):
    _q("AD-01", "adversarial",
        "What is the function of the Phvul.011G999950 gene in common bean drought tolerance?",
        ["should state this gene/locus is not identified in the retrieved sources rather than inventing a function",
         "should not fabricate a specific molecular mechanism"],
        expect_low_confidence=True),
    _q("AD-02", "adversarial",
        "How does the PvDT-9X transcription factor regulate nitrogen fixation in common bean nodules?",
        ["should state this factor is not found in the retrieved sources",
         "should not invent regulatory mechanism details"],
        expect_low_confidence=True),
    _q("AD-03", "adversarial",
        "What allele frequency has been reported for the BRR-77 rust resistance locus in Andean bean germplasm?",
        ["should state this locus name is not found in the retrieved sources",
         "should not invent a numeric allele frequency"],
        expect_low_confidence=True),
    # Out-of-scope topics phrased like bean research questions:
    _q("AD-04", "adversarial",
        "What is the average wholesale price per kilogram of black bean in commodity markets in 2025?",
        ["should state that market/commodity pricing data is outside the scope of this research corpus",
         "should not invent a specific price figure"],
        expect_low_confidence=True),
    _q("AD-05", "adversarial",
        "How does bean plant root architecture influence its use as a companion planting species for roses in ornamental gardens?",
        ["should note this specific ornamental-gardening application is not covered by the indexed research",
         "should not fabricate root architecture claims specific to rose companion planting"],
        expect_low_confidence=True),
    _q("AD-06", "adversarial",
        "What machine learning algorithms have been used to predict bean yield from satellite imagery in the past two years?",
        ["should state whether this specific remote-sensing/ML application is covered, without inventing algorithm names or accuracy figures not present in the sources"],
        expect_low_confidence=True),
    # Specific cultivars not expected to be in the corpus:
    _q("AD-07", "adversarial",
        "What are the reported yield and disease resistance characteristics of the 'Sierra Crimson-9' dry bean cultivar?",
        ["should state this cultivar is not identified in the retrieved sources",
         "should not invent yield or disease resistance figures for it"],
        expect_low_confidence=True),
    _q("AD-08", "adversarial",
        "How does the cultivar 'Tanzanite Bush 22' perform under drought stress compared to standard checks?",
        ["should state this cultivar is not identified in the retrieved sources",
         "should not invent drought performance data for it"],
        expect_low_confidence=True),
    _q("AD-09", "adversarial",
        "What breeding history led to the release of the 'Highland Ember' navy bean variety?",
        ["should state this cultivar/variety is not identified in the retrieved sources",
         "should not invent a breeding history for it"],
        expect_low_confidence=True),
    _q("AD-10", "adversarial",
        "What is the recommended fungicide program specifically validated for the 'Copperfield-14' pinto bean cultivar?",
        ["should state this cultivar is not identified in the retrieved sources",
         "should not invent a cultivar-specific fungicide program"],
        expect_low_confidence=True),

    # ── 7. Multi-hop synthesis (15) — require 3+ distinct sub-topics/papers ──
    _q("MH-01", "multi_hop",
        "How do drought tolerance, nitrogen fixation capacity, and root architecture jointly determine common bean performance in low-input smallholder systems, and which trait combination has been prioritized in breeding programs targeting these systems?",
        ["addresses drought tolerance", "addresses nitrogen fixation capacity",
         "addresses root architecture", "synthesizes how the three interact or are jointly prioritized"]),
    _q("MH-02", "multi_hop",
        "Synthesize how rust resistance genes from Middle American gene pools, common bacterial blight resistance from Andean sources, and marker-assisted selection have been combined in multiline or gene-pyramided bean varieties.",
        ["rust resistance / Middle American gene pool", "CBB resistance / Andean source",
         "marker-assisted selection", "synthesis of combining multiple resistances"]),
    _q("MH-03", "multi_hop",
        "How do soil phosphorus availability, Rhizobium strain effectiveness, and aluminum toxicity in acid tropical soils together constrain nitrogen fixation potential in common bean grown in the tropics?",
        ["phosphorus availability effect", "Rhizobium strain effectiveness",
         "aluminum toxicity effect", "synthesis of combined constraint on nitrogen fixation"]),
    _q("MH-04", "multi_hop",
        "What is the combined evidence linking canopy architecture, humidity or microclimate, and fungicide timing in managing white mold risk in dry bean production?",
        ["canopy architecture", "microclimate or humidity",
         "fungicide timing", "synthesis across all three for disease management"]),
    _q("MH-05", "multi_hop",
        "How have determinate growth habit, plant density, and mechanical harvestability jointly shaped modern dry bean ideotype breeding?",
        ["determinate growth habit", "plant density",
         "mechanical harvestability", "synthesis into an ideotype concept"]),
    _q("MH-06", "multi_hop",
        "Synthesize the evidence on how drought timing (pre-flowering vs. post-flowering), genotype root depth, and osmotic adjustment capacity together determine yield loss under terminal drought in common bean.",
        ["drought timing effect", "root depth or genotype factor",
         "osmotic adjustment", "synthesis of combined yield-loss determinants"]),
    _q("MH-07", "multi_hop",
        "How do seed coat characteristics, storage conditions, and genotype interact to produce the hard-to-cook defect in stored common bean, and what breeding or postharvest solutions address all three factors?",
        ["seed coat characteristic", "storage condition",
         "genotype factor", "a solution addressing multiple factors"]),
    _q("MH-08", "multi_hop",
        "What combined role do whitefly population dynamics, virus strain variation, and host plant resistance genes play in bean golden mosaic virus epidemics?",
        ["whitefly population dynamics", "virus strain variation",
         "host resistance genes", "synthesis of epidemic dynamics"]),
    _q("MH-09", "multi_hop",
        "How do intercropping arrangement, nitrogen fertilization rate, and maize-bean competition for light jointly determine bean yield in maize-bean intercropping systems?",
        ["intercropping arrangement", "nitrogen fertilization rate",
         "light competition", "synthesis of combined yield determinants"]),
    _q("MH-10", "multi_hop",
        "Synthesize how QTL mapping population design, marker density, and multi-environment replication have collectively advanced understanding of drought tolerance genetics in common bean.",
        ["QTL mapping population design", "marker density or technology",
         "multi-environment replication", "synthesis of collective advancement"]),
    _q("MH-11", "multi_hop",
        "How do gene pool origin (Andean vs. Middle American), seed size, and cooking quality traits interact in determining market class classification of common bean cultivars?",
        ["gene pool origin", "seed size",
         "cooking quality", "synthesis into market class classification"]),
    _q("MH-12", "multi_hop",
        "What is the combined evidence on how heat stress during flowering, pollen viability, and pod set percentage determine yield loss in bean grown under high-temperature conditions?",
        ["heat stress at flowering", "pollen viability",
         "pod set percentage", "synthesis of the yield-loss mechanism"]),
    _q("MH-13", "multi_hop",
        "How have breeding for anthracnose resistance, fungicide seed treatment, and certified seed systems together contributed to managing Colletotrichum lindemuthianum in bean production?",
        ["anthracnose resistance breeding", "fungicide seed treatment",
         "certified seed systems", "synthesis of combined disease management"]),
    _q("MH-14", "multi_hop",
        "Synthesize how soil organic matter, biological nitrogen fixation, and reduced synthetic fertilizer use interact in sustainable bean cropping systems.",
        ["soil organic matter role", "biological nitrogen fixation",
         "reduced fertilizer use", "synthesis into a sustainability framing"]),
    _q("MH-15", "multi_hop",
        "How do smallholder farmer seed access, variety adoption rates, and on-farm yield gaps collectively explain the difference between research-station yield potential and farmer-realized yields for improved bean varieties?",
        ["seed access", "variety adoption rate",
         "on-farm yield gap", "synthesis explaining the potential-vs-realized gap"]),
]


def get_categories() -> List[str]:
    """Ordered list of unique categories present in BENCHMARK."""
    seen = []
    for q in BENCHMARK:
        if q["category"] not in seen:
            seen.append(q["category"])
    return seen


if __name__ == "__main__":
    from collections import Counter
    counts = Counter(q["category"] for q in BENCHMARK)
    print(f"Total questions: {len(BENCHMARK)}")
    for cat, n in counts.items():
        print(f"  {cat:<15} {n}")
    with_gt = sum(1 for q in BENCHMARK if q["ground_truth"])
    print(f"\nQuestions with ground_truth populated: {with_gt}/{len(BENCHMARK)}")
