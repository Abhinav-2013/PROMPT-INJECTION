"""
ATHS Hypothesis Engine

Converts the output of the Threat Detection Engine into
human-readable security hypotheses.

Input:
    Detection Engine analysis result

Output:
    Structured hypotheses describing:
    - What type of threat was detected
    - Why it was detected
    - Supporting evidence
    - Confidence
    - Recommended action
"""


from typing import Any, Dict, List


class HypothesisEngine:
    """
    Generates security hypotheses from ATHS detection results.
    """

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.75
    MEDIUM_CONFIDENCE = 0.50

    # Threat categories
    CATEGORY_INSTRUCTION_OVERRIDE = "Instruction Override"
    CATEGORY_SYSTEM_PROMPT = "System Prompt Reference"
    CATEGORY_PROMPT_LEAKAGE = "Prompt Leakage"
    CATEGORY_ROLE_CHANGE = "Role Change Attempt"
    CATEGORY_POLICY_BYPASS = "Policy Bypass"
    CATEGORY_DATA_EXTRACTION = "Data Extraction"
    CATEGORY_TOOL_MANIPULATION = "Tool Manipulation"
    CATEGORY_SENSITIVE_DATA = "Sensitive Data Reference"
    CATEGORY_SEMANTIC = "Semantic Prompt Injection"
    CATEGORY_GENERAL = "General Prompt Injection"

    def __init__(self):
        """Initialize the Hypothesis Engine."""

        self.rule_category_map = {
            "instruction_override": self.CATEGORY_INSTRUCTION_OVERRIDE,
            "system_prompt_reference": self.CATEGORY_SYSTEM_PROMPT,
            "prompt_leakage": self.CATEGORY_PROMPT_LEAKAGE,
            "role_change_attempt": self.CATEGORY_ROLE_CHANGE,
            "policy_bypass": self.CATEGORY_POLICY_BYPASS,
            "data_extraction_attempt": self.CATEGORY_DATA_EXTRACTION,
            "tool_manipulation": self.CATEGORY_TOOL_MANIPULATION,
            "sensitive_data_reference": self.CATEGORY_SENSITIVE_DATA,
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_bool(value: Any, default: bool = False) -> bool:
        """Safely convert a value to bool."""

        if isinstance(value, bool):
            return value

        if value is None:
            return default

        return bool(value)

    def _get_confidence(self, threat_score: float) -> str:
        """
        Convert threat score into a confidence level.
        """

        if threat_score >= self.HIGH_CONFIDENCE:
            return "HIGH"

        if threat_score >= self.MEDIUM_CONFIDENCE:
            return "MEDIUM"

        return "LOW"

    # ------------------------------------------------------------------
    # Evidence extraction
    # ------------------------------------------------------------------

    def _extract_rule_categories(
        self,
        analysis: Dict[str, Any],
    ) -> List[str]:
        """
        Extract human-readable threat categories from rule detector output.
        """

        categories = []

        rule_result = analysis.get("rules", {})

        raw_categories = rule_result.get("categories", [])

        if not isinstance(raw_categories, list):
            return categories

        for category in raw_categories:

            readable_category = self.rule_category_map.get(
                str(category),
                str(category).replace("_", " ").title(),
            )

            if readable_category not in categories:
                categories.append(readable_category)

        return categories

    def _build_evidence(
        self,
        analysis: Dict[str, Any],
    ) -> List[str]:
        """
        Build human-readable evidence from all detectors.
        """

        evidence = []

        # --------------------------------------------------------------
        # ML evidence
        # --------------------------------------------------------------

        ml = analysis.get("ml", {})

        ml_probability = self._safe_float(
            ml.get("probability_malicious", 0.0)
        )

        ml_label = ml.get("label", 0)

        if ml_label == 1 or ml_probability >= self.MEDIUM_CONFIDENCE:
            evidence.append(
                f"ML detector assigned a malicious probability "
                f"of {ml_probability:.4f}"
            )

        # --------------------------------------------------------------
        # Rule evidence
        # --------------------------------------------------------------

        rules = analysis.get("rules", {})

        rule_score = self._safe_float(
            rules.get("score", rules.get("rule_score", 0.0))
        )

        if rule_score > 0:
            matched_rules = rules.get("matched_rules", [])

            if matched_rules:
                evidence.append(
                    "Rule detector matched: "
                    + ", ".join(str(rule) for rule in matched_rules)
                )
            else:
                evidence.append(
                    f"Rule detector produced a threat score "
                    f"of {rule_score:.4f}"
                )

        # --------------------------------------------------------------
        # Semantic evidence
        # --------------------------------------------------------------

        semantic = analysis.get("semantic", {})

        semantic_score = self._safe_float(
            semantic.get(
                "threat_score",
                semantic.get("score", 0.0),
            )
        )

        semantic_similarity = self._safe_float(
            semantic.get("similarity", 0.0)
        )

        malicious_neighbors = int(
            semantic.get("malicious_neighbor_count", 0)
        )

        strong_malicious_neighbors = int(
            semantic.get("strong_malicious_neighbor_count", 0)
        )

        if semantic_score > 0:

            evidence.append(
                f"Semantic detector found malicious similarity "
                f"of {semantic_similarity:.4f}"
            )

        if malicious_neighbors > 0:

            evidence.append(
                f"{malicious_neighbors} malicious semantic "
                f"neighbor(s) found"
            )

        if strong_malicious_neighbors > 0:

            evidence.append(
                f"{strong_malicious_neighbors} strong malicious "
                f"semantic neighbor(s) found"
            )

        # --------------------------------------------------------------
        # Detector voting
        # --------------------------------------------------------------

        voting = analysis.get("detector_votes", {})

        total_votes = voting.get("total", 0)

        if total_votes:

            ml_vote = self._safe_bool(voting.get("ml"))
            rule_vote = self._safe_bool(voting.get("rules"))
            semantic_vote = self._safe_bool(voting.get("semantic"))

            vote_count = sum(
                [
                    ml_vote,
                    rule_vote,
                    semantic_vote,
                ]
            )

            evidence.append(
                f"{vote_count}/{total_votes} detectors "
                f"classified the prompt as a threat"
            )

        return evidence

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def _generate_hypotheses(
        self,
        analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on detected evidence.
        """

        hypotheses = []

        rule_categories = self._extract_rule_categories(analysis)

        semantic = analysis.get("semantic", {})

        semantic_score = self._safe_float(
            semantic.get(
                "threat_score",
                semantic.get("score", 0.0),
            )
        )

        # --------------------------------------------------------------
        # Rule-based hypotheses
        # --------------------------------------------------------------

        for category in rule_categories:

            hypotheses.append(
                {
                    "type": category,
                    "description": (
                        f"{category} behavior detected in the prompt."
                    ),
                    "source": "rule_detector",
                }
            )

        # --------------------------------------------------------------
        # Semantic hypothesis
        # --------------------------------------------------------------

        if semantic_score >= self.MEDIUM_CONFIDENCE:

            hypotheses.append(
                {
                    "type": self.CATEGORY_SEMANTIC,
                    "description": (
                        "The prompt is semantically similar to "
                        "previously identified malicious prompts."
                    ),
                    "source": "embedding_detector",
                }
            )

        # --------------------------------------------------------------
        # General hypothesis fallback
        # --------------------------------------------------------------

        if not hypotheses:

            threat_score = self._safe_float(
                analysis.get("threat_score", 0.0)
            )

            if threat_score >= self.MEDIUM_CONFIDENCE:

                hypotheses.append(
                    {
                        "type": self.CATEGORY_GENERAL,
                        "description": (
                            "Multiple detection signals indicate "
                            "potential prompt injection behavior."
                        ),
                        "source": "threat_fusion",
                    }
                )

        return hypotheses

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate hypotheses from a Threat Detection Engine result.
        """

        if not isinstance(analysis, dict):
            raise TypeError(
                "Detection analysis must be a dictionary."
            )

        threat_score = self._safe_float(
            analysis.get("threat_score", 0.0)
        )

        decision = str(
            analysis.get("decision", "ALLOW")
        ).upper()

        severity = str(
            analysis.get("severity", "LOW")
        ).upper()

        confidence = self._get_confidence(threat_score)

        evidence = self._build_evidence(analysis)

        hypotheses = self._generate_hypotheses(analysis)

        # --------------------------------------------------------------
        # Primary hypothesis
        # --------------------------------------------------------------

        if hypotheses:

            primary_hypothesis = hypotheses[0]["type"]

            primary_description = hypotheses[0]["description"]

        else:

            primary_hypothesis = "No Threat Detected"

            primary_description = (
                "No significant prompt injection behavior "
                "was identified."
            )

        # --------------------------------------------------------------
        # Recommended action
        # --------------------------------------------------------------

        if decision == "BLOCK":

            recommended_action = "BLOCK"

        elif decision == "REVIEW":

            recommended_action = "REVIEW"

        else:

            recommended_action = "ALLOW"

        return {
            "hypothesis": primary_hypothesis,
            "description": primary_description,
            "hypotheses": hypotheses,
            "evidence": evidence,
            "confidence": confidence,
            "threat_score": threat_score,
            "decision": decision,
            "severity": severity,
            "recommended_action": recommended_action,
            "evidence_count": len(evidence),
            "hypothesis_count": len(hypotheses),
        }


# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------

def main():
    """
    Test the Hypothesis Engine independently.
    """

    print("ATHS HYPOTHESIS ENGINE TEST")

    engine = HypothesisEngine()

    # --------------------------------------------------------------
    # BENIGN TEST
    # --------------------------------------------------------------

    benign_analysis = {
        "threat_score": 0.0913,
        "decision": "ALLOW",
        "severity": "LOW",

        "ml": {
            "label": 0,
            "probability_malicious": 0.1826,
        },

        "rules": {
            "rule_score": 0.0,
            "is_threat": False,
            "matched_rules": [],
            "categories": [],
        },

        "semantic": {
            "similarity": 0.4701,
            "threat_score": 0.0,
            "matched_label": 0,
            "malicious_neighbor_count": 2,
            "strong_malicious_neighbor_count": 0,
        },

        "detector_votes": {
            "ml": False,
            "rules": False,
            "semantic": False,
            "total": 3,
        },
    }

    benign_result = engine.analyze(benign_analysis)

    print("\nBENIGN PROMPT")
    print("Hypothesis       :", benign_result["hypothesis"])
    print("Confidence       :", benign_result["confidence"])
    print("Threat score     :", f"{benign_result['threat_score']:.4f}")
    print("Decision         :", benign_result["decision"])
    print("Recommended action:", benign_result["recommended_action"])

    print("Evidence:")

    if benign_result["evidence"]:

        for item in benign_result["evidence"]:
            print("  -", item)

    else:

        print("  None")

    # --------------------------------------------------------------
    # MALICIOUS TEST
    # --------------------------------------------------------------

    malicious_analysis = {
        "threat_score": 0.9249,
        "decision": "BLOCK",
        "severity": "HIGH",

        "ml": {
            "label": 1,
            "probability_malicious": 0.9847,
        },

        "rules": {
            "rule_score": 0.9,
            "is_threat": True,
            "matched_rules": [
                "instruction_override",
            ],
            "categories": [
                "instruction_override",
            ],
        },

        "semantic": {
            "similarity": 0.7838,
            "threat_score": 0.7838,
            "matched_label": 1,
            "malicious_neighbor_count": 5,
            "strong_malicious_neighbor_count": 2,
        },

        "detector_votes": {
            "ml": True,
            "rules": True,
            "semantic": True,
            "total": 3,
        },
    }

    malicious_result = engine.analyze(malicious_analysis)

    print("\nMALICIOUS PROMPT")
    print("Hypothesis       :", malicious_result["hypothesis"])
    print("Confidence       :", malicious_result["confidence"])
    print("Threat score     :", f"{malicious_result['threat_score']:.4f}")
    print("Decision         :", malicious_result["decision"])
    print("Recommended action:", malicious_result["recommended_action"])

    print("Hypotheses:")

    for hypothesis in malicious_result["hypotheses"]:

        print(
            f"  - {hypothesis['type']}: "
            f"{hypothesis['description']}"
        )

    print("Evidence:")

    for item in malicious_result["evidence"]:

        print("  -", item)

    # --------------------------------------------------------------
    # REVIEW TEST
    # --------------------------------------------------------------

    review_analysis = {
        "threat_score": 0.4900,
        "decision": "REVIEW",
        "severity": "MEDIUM",

        "ml": {
            "label": 1,
            "probability_malicious": 0.55,
        },

        "rules": {
            "rule_score": 0.40,
            "is_threat": False,
            "matched_rules": [
                "policy_bypass",
            ],
            "categories": [
                "policy_bypass",
            ],
        },

        "semantic": {
            "similarity": 0.50,
            "threat_score": 0.0,
            "matched_label": 0,
            "malicious_neighbor_count": 0,
            "strong_malicious_neighbor_count": 0,
        },

        "detector_votes": {
            "ml": True,
            "rules": False,
            "semantic": False,
            "total": 3,
        },
    }

    review_result = engine.analyze(review_analysis)

    print("\nREVIEW PROMPT")
    print("Hypothesis       :", review_result["hypothesis"])
    print("Confidence       :", review_result["confidence"])
    print("Threat score     :", f"{review_result['threat_score']:.4f}")
    print("Decision         :", review_result["decision"])
    print("Recommended action:", review_result["recommended_action"])

    print("\nATHS HYPOTHESIS ENGINE TEST COMPLETE")


if __name__ == "__main__":
    main()