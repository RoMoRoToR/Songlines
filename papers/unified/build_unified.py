#!/usr/bin/env python3
"""Build the unified Songlines monograph from the four current papers.

Mechanically merges (in series order):
  Part I    qrmc_measurement_framework  -- the Q/R/M/C protocol
  Part II   semantic_warp               -- provenance-conditioned completion
  Part III  route_warp                  -- routes, place identity, songs
  Part IV   collective_semantic_memory  -- CSM mechanics + categorical frame

symbolic_memory is the historical ANCESTOR of Parts I and IV (it split
into them); every experiment it contains lives in those parts, so its
body is not duplicated -- the lineage is stated in the roadmap.

Per-source transformations: strip preamble/title/abstract, prefix all
labels and refs with pN:, extract bibitems into one deduplicated
bibliography, convert \\appendix into a visible part-appendix divider
(sections stay arabic), drop NeurIPS checklists.

Usage:  python3 build_unified.py [ru]
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
RU = len(sys.argv) > 1 and sys.argv[1] == "ru"
SUF = "_ru" if RU else ""

SOURCES = [
    ("p1", "qrmc_measurement_framework/songlines_qrmc_measurement_framework"),
    ("p2", "semantic_warp/songlines_semantic_warp"),
    ("p3", "route_warp/songlines_route_warp"),
    ("p4", "collective_semantic_memory/songlines_collective_semantic_memory"),
    ("p5", "song_grammar_ucsm/songlines_song_grammar_ucsm"),
]

PART_TITLES_EN = {
    "p1": ("The Q/R/M/C Measurement Framework",
           "Stage-decomposed evaluation of memory-based navigation: "
           "one logging contract, five system classes, and the "
           "candidate-generation diagnosis."),
    "p2": ("Semantic Warp: Provenance-Conditioned Completion",
           "Navigating by someone else's memory as a measured event: "
           "the foreignness annotation, the warp distance law, and the "
           "Warp Drive coordination protocol."),
    "p3": ("The Song, Not the Pin: Routes and Meaning-Based Identity",
           "What transfers between agents is the song: edge-provenanced "
           "routes, landmark-anchored place identity without shared "
           "frames, the vocabulary ablation, and the song anatomy."),
    "p4": ("Collective Semantic Memory: Merge, Trust, and Staleness",
           "The mechanics of peer memory -- three explicit rules, the "
           "third regime on the M-by-C plane -- and the categorical "
           "frame in which commonality of interpretation is a "
           "parameter, not an assumption."),
    "p5": ("Song Grammar and Utility-Gated Memory: the Ontogenesis "
           "of What Is Remembered",
           "Memory formation as a two-axis decision (counterfactual "
           "utility times simplicity of analogy): measured "
           "deterministically, learned by a bandit that improves on "
           "its designer, evolved to a finite budget, and priced "
           "against seven policies over heterogeneous embodiments."),
}
PART_TITLES_RU = {
    "p1": ("Измерительный фреймворк Q/R/M/C",
           "Стадийная оценка навигации на основе памяти: один "
           "logging-контракт, пять классов систем и диагноз "
           "candidate generation."),
    "p2": ("Семантический варп: завершение, обусловленное провенансом",
           "Навигация по чужой памяти как измеряемое событие: "
           "аннотация чужеродности, закон варп-дистанции и "
           "координационный протокол Warp Drive."),
    "p3": ("Песня, а не точка: маршруты и смысловая идентичность",
           "Между агентами переносится песня: маршруты с провенансом "
           "рёбер, идентичность мест без общих систем координат, "
           "абляция словаря и анатомия песни."),
    "p4": ("Коллективная семантическая память: слияние, доверие, "
           "устаревание",
           "Механика peer-памяти — три явных правила, третий режим на "
           "плоскости M×C — и категорная рамка, в которой общность "
           "интерпретации — параметр, а не предположение."),
    "p5": ("Грамматика песни и Utility-Gated Memory: онтогенез "
           "запоминаемого",
           "Формирование памяти как двухосевое решение "
           "(контрфактическая полезность × простота аналогии): "
           "измерено детерминированно, выучено бандитом, обыгравшим "
           "проектировщика, эволюционировано к конечному бюджету и "
           "расценено против семи политик на гетерогенных "
           "воплощениях."),
}

REFCMDS = r"(?:label|ref|eqref|pageref|autoref|Cref|cref)"


def prefix_labels(body: str, p: str) -> str:
    return re.sub(r"\\(" + REFCMDS + r")\{([^}]*)\}",
                  lambda m: "\\%s{%s}" % (
                      m.group(1),
                      ",".join(f"{p}:{x.strip()}"
                               for x in m.group(2).split(","))),
                  body)


def extract(source: Path, p: str):
    tex = source.read_text()
    # abstract
    am = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex,
                   re.S)
    abstract = am.group(1).strip() if am else ""
    # body: after \maketitle to \end{document}
    body = tex.split("\\maketitle", 1)[1]
    body = body.split("\\end{document}")[0]
    if am:
        body = body.replace(am.group(0), "")
    # bibliography
    bibitems = []
    bm = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                   body, re.S)
    if bm:
        block = bm.group(0)
        for item in re.split(r"(?=\\bibitem)", block):
            if item.startswith("\\bibitem"):
                item = item.split("\\end{thebibliography}")[0]
                bibitems.append(item.strip())
        body = body.replace(bm.group(0), "")
    # checklist
    body = re.sub(r"\\input\{checklist[^}]*\}", "", body)
    # appendix divider (sections stay arabic inside the monograph)
    divider = ("\\FloatBarrier\n\\begin{center}\\rule{0.55\\linewidth}"
               "{0.4pt}\\\\[2pt]\\textbf{"
               + ("Приложения к этой части" if RU else
                  "Appendix material of this part")
               + "}\\end{center}\n")
    body = body.replace("\\appendix", divider)
    body = prefix_labels(body, p)
    abstract = prefix_labels(abstract, p)
    return abstract, body.strip(), bibitems


def main():
    titles = PART_TITLES_RU if RU else PART_TITLES_EN
    parts, bib, seen = [], [], set()
    for p, rel in SOURCES:
        abstract, body, items = extract(
            HERE.parent / (rel + SUF + ".tex"), p)
        for it in items:
            key = re.search(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", it)
            k = key.group(1) if key else it[:40]
            if k not in seen:
                seen.add(k)
                bib.append(it)
        title, blurb = titles[p]
        head = (f"\\part{{{title}}}\n"
                f"\\noindent\\textit{{{blurb}}}\n\n"
                + ("\\paragraph{Аннотация части.}" if RU else
                   "\\paragraph{Part abstract.}")
                + f" {abstract}\n\n\\FloatBarrier\n")
        parts.append(head + body + "\n\\FloatBarrier\n")

    front = (HERE / ("front_ru.tex" if RU else "front.tex")).read_text()
    preamble = (HERE / ("preamble_ru.tex" if RU else
                        "preamble.tex")).read_text()
    doc = (preamble + "\n\\begin{document}\n\\maketitle\n" + front
           + "\n\\clearpage\n\\tableofcontents\n\\clearpage\n"
           + "\n\\clearpage\n".join(parts)
           + "\n\\clearpage\n\\begin{thebibliography}{99}\n"
           + "\n\n".join(bib)
           + "\n\\end{thebibliography}\n\\end{document}\n")
    out = HERE / f"songlines_unified{SUF}.tex"
    out.write_text(doc)
    print(f"wrote {out} ({len(doc.splitlines())} lines, "
          f"{len(bib)} bib entries)")


if __name__ == "__main__":
    main()
