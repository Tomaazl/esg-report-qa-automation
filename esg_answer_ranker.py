"""
ESG Answer Ranker
------------------

Lightweight utility to suggest answer candidates for ESG questionnaire questions.

Usage:
    from esg_answer_ranker import ESGAnswerRanker

    ranker = ESGAnswerRanker()
    suggestions = ranker.rank_answer_candidates_for_question(
        "Do you track Scope 1 and Scope 2 emissions?",
        top_k=3,
    )
    # suggestions -> List[Tuple[str, float]] with (answer_text, score)

No external dependencies; uses simple keyword overlap and text similarity.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Sequence, Set, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into a set of lowercase word tokens."""
    if not text:
        return set()
    return {token.lower() for token in _WORD_RE.findall(text.lower())}


@dataclass(frozen=True)
class AnswerCandidate:
    label: str
    text: str
    keywords: Set[str]


def _build_candidate(label: str, text: str, keywords: Iterable[str]) -> AnswerCandidate:
    return AnswerCandidate(label=label, text=text.strip(), keywords={k.lower() for k in keywords})


# A compact, extensible default answer bank covering common ESG topics
DEFAULT_ANSWER_BANK: List[AnswerCandidate] = [
    _build_candidate(
        label="GHG Emissions (Scope 1 & 2)",
        text=(
            "We quantify Scope 1 and Scope 2 greenhouse gas emissions annually and disclose the "
            "results in our sustainability report. Our methodology aligns with the GHG Protocol, "
            "and we have third-party assurance for key metrics."
        ),
        keywords=["scope", "scope1", "scope2", "ghg", "emissions", "greenhouse", "protocol", "assurance"],
    ),
    _build_candidate(
        label="Net-Zero / Targets",
        text=(
            "We have set science-based targets to reduce absolute emissions and are committed to "
            "achieving net-zero by 2050. Interim targets are established for 2030, with annual "
            "progress tracking and board oversight."
        ),
        keywords=["science-based", "net-zero", "targets", "2050", "2030", "board", "oversight"],
    ),
    _build_candidate(
        label="Renewable Energy",
        text=(
            "We procure renewable electricity through a mix of on-site generation and energy "
            "attribute certificates. Our goal is to source 100% of electricity from renewables by 2030."
        ),
        keywords=["renewable", "electricity", "rec", "ppas", "solar", "wind", "certificates", "2030"],
    ),
    _build_candidate(
        label="Energy Efficiency",
        text=(
            "We implement energy efficiency measures across operations, including LED retrofits, "
            "HVAC optimization, and smart building controls, resulting in year-over-year intensity reductions."
        ),
        keywords=["energy", "efficiency", "led", "hvac", "optimization", "intensity", "reduction"],
    ),
    _build_candidate(
        label="Water Management",
        text=(
            "We monitor water withdrawal and consumption, prioritize water-stressed locations, and deploy "
            "recycling and reuse technologies to reduce absolute consumption."
        ),
        keywords=["water", "withdrawal", "consumption", "stress", "recycle", "reuse"],
    ),
    _build_candidate(
        label="Waste & Circularity",
        text=(
            "We track waste generation and diversion rates, prioritizing reduction, reuse, and recycling. "
            "We aim to increase landfill diversion through circular economy initiatives."
        ),
        keywords=["waste", "diversion", "recycling", "landfill", "circular", "reuse"],
    ),
    _build_candidate(
        label="Supplier ESG (Scope 3)",
        text=(
            "We assess material suppliers on ESG performance and request Scope 3 data. We engage suppliers "
            "to set emission reduction targets and improve transparency across the value chain."
        ),
        keywords=["supplier", "scope3", "value", "chain", "engage", "assessment", "procurement"],
    ),
    _build_candidate(
        label="DEI (Diversity, Equity & Inclusion)",
        text=(
            "We maintain a comprehensive DEI strategy with measurable goals for representation, pay equity, "
            "and inclusion. We publish workforce demographics and conduct regular pay equity analyses."
        ),
        keywords=["diversity", "equity", "inclusion", "dei", "representation", "pay", "equity"],
    ),
    _build_candidate(
        label="Employee Health & Safety",
        text=(
            "We uphold robust health and safety management systems aligned to recognized standards, track TRIR, "
            "and provide training to minimize incidents and promote a strong safety culture."
        ),
        keywords=["health", "safety", "hse", "trir", "training", "incidents"],
    ),
    _build_candidate(
        label="Training & Development",
        text=(
            "We offer structured learning and development programs, including compliance, leadership, and role-based "
            "training, with tracked completion rates and manager accountability."
        ),
        keywords=["training", "learning", "development", "compliance", "completion"],
    ),
    _build_candidate(
        label="Human Rights & Labor",
        text=(
            "We adhere to international human rights standards, prohibit forced and child labor, and conduct risk "
            "assessments and audits in high-risk regions."
        ),
        keywords=["human", "rights", "labor", "forced", "child", "audit", "risk"],
    ),
    _build_candidate(
        label="Anti-Corruption & Ethics",
        text=(
            "We maintain an anti-corruption compliance program with training, third-party due diligence, and a code of "
            "conduct applicable to all employees and suppliers."
        ),
        keywords=["anti-corruption", "ethics", "bribery", "code", "conduct", "due", "diligence"],
    ),
    _build_candidate(
        label="Data Privacy & Security",
        text=(
            "We implement data protection controls aligned to recognized frameworks, conduct regular security testing, "
            "and provide privacy training to employees."
        ),
        keywords=["data", "privacy", "security", "protection", "training", "testing"],
    ),
    _build_candidate(
        label="Board Governance & Oversight",
        text=(
            "Our board oversees ESG strategy and risk management through a designated committee, receiving periodic "
            "updates on progress and key metrics."
        ),
        keywords=["board", "governance", "oversight", "committee", "risk", "management"],
    ),
    _build_candidate(
        label="Whistleblower & Grievance Mechanisms",
        text=(
            "We provide confidential reporting channels for employees and stakeholders, prohibit retaliation, and "
            "track investigations to closure."
        ),
        keywords=["whistleblower", "hotline", "grievance", "retaliation", "reporting", "investigation"],
    ),
]


class ESGAnswerRanker:
    """Ranks default ESG answer candidates for given questions.

    Scoring model:
      - Keyword overlap score = matched_keywords / max(1, num_keywords)
      - Surface text similarity = difflib.SequenceMatcher ratio between question and candidate text
      - Final score = 0.7 * overlap + 0.3 * similarity

    This is intentionally simple and dependency-free. You may extend by adding domain-specific
    keywords or replacing the similarity function.
    """

    def __init__(
        self,
        answer_bank: Sequence[AnswerCandidate] | None = None,
        overlap_weight: float = 0.7,
        similarity_weight: float = 0.3,
    ) -> None:
        if overlap_weight < 0.0 or similarity_weight < 0.0:
            raise ValueError("Weights must be non-negative")
        weight_sum = overlap_weight + similarity_weight
        if weight_sum == 0.0:
            raise ValueError("At least one weight must be positive")
        # Normalize weights to sum to 1.0 for interpretability
        self.overlap_weight = overlap_weight / weight_sum
        self.similarity_weight = similarity_weight / weight_sum
        self.answer_bank: List[AnswerCandidate] = list(answer_bank or DEFAULT_ANSWER_BANK)

    def rank_answer_candidates_for_question(
        self,
        question: str,
        top_k: int = 3,
        min_keywords_to_consider: int = 0,
    ) -> List[Tuple[str, float]]:
        """Return top-K answer candidates as (answer_text, score) for a single question."""
        question = (question or "").strip()
        if not question:
            return []

        question_tokens = _tokenize(question)

        scored: List[Tuple[AnswerCandidate, float]] = []
        for candidate in self.answer_bank:
            if len(candidate.keywords) < min_keywords_to_consider:
                continue
            overlap_score = self._keyword_overlap_score(question_tokens, candidate.keywords)
            similarity_score = self._string_similarity(question, candidate.text)
            final_score = (self.overlap_weight * overlap_score) + (
                self.similarity_weight * similarity_score
            )
            scored.append((candidate, final_score))

        scored.sort(key=lambda pair: pair[1], reverse=True)

        top_candidates = scored[: max(0, top_k)] if top_k > 0 else scored
        return [(c.text, round(score, 4)) for c, score in top_candidates]

    def rank_answer_candidates(
        self,
        questions: Sequence[str],
        top_k: int = 3,
        min_keywords_to_consider: int = 0,
    ) -> List[List[Tuple[str, float]]]:
        """Return a list for each question containing (answer_text, score) tuples."""
        if not questions:
            return []
        return [
            self.rank_answer_candidates_for_question(q, top_k=top_k, min_keywords_to_consider=min_keywords_to_consider)
            for q in questions
        ]

    @staticmethod
    def _keyword_overlap_score(question_tokens: Set[str], answer_keywords: Set[str]) -> float:
        if not answer_keywords:
            return 0.0
        if not question_tokens:
            return 0.0

        matched = 0
        for keyword in answer_keywords:
            if keyword in question_tokens:
                matched += 1
                continue
            # Allow for very close matches for hyphenation/casing variants
            for token in question_tokens:
                if ESGAnswerRanker._string_similarity(keyword, token) >= 0.92:
                    matched += 1
                    break

        return matched / float(len(answer_keywords))

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        # difflib returns [0, 1]
        return SequenceMatcher(a=a.lower(), b=b.lower()).ratio()


__all__ = [
    "ESGAnswerRanker",
    "AnswerCandidate",
    "DEFAULT_ANSWER_BANK",
]

