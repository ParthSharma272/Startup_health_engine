import json
import os
from typing import Dict, Any, List
from src.core.config_loader import ConfigLoader
from src.core.kpi_normalizer import KPINormalizer
from src.utils.logger_config import logger
import openai # Added for OpenAI API calls

class ScoringEngine:
    def __init__(self, config_loader: ConfigLoader, openai_api_key: str = None):
        self.config_loader = config_loader
        self.kpi_weights = self.config_loader.load_config(self.config_loader.kpi_weights_path)
        self.kpi_benchmarks = self.config_loader.load_config(self.config_loader.kpi_benchmarks_path)
        self.kpi_benchmark_map = self.config_loader.get_kpi_benchmark_map(self.kpi_benchmarks)
        self.kpi_normalizer = KPINormalizer(self.kpi_benchmark_map) # KPINormalizer now takes kpi_benchmark_map
        
        # Load percentile thresholds for score categorization
        self.percentile_thresholds = self.config_loader.load_config(self.config_loader.percentile_thresholds_path)
        
        self.openai_api_key = openai_api_key # Store the API key
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            self.llm_model_for_suggestions = "gpt-3.5-turbo" # Or "gpt-4o"
            logger.info(f"OpenAI client initialized for ScoringEngine with model: {self.llm_model_for_suggestions}.")
        else:
            self.openai_client = None
            logger.warning("OPENAI_API_KEY not provided to ScoringEngine. LLM-based suggestions will be skipped.")

        logger.info("ScoringEngine initialized with KPI weights, benchmarks, and percentile thresholds.")

    def calculate_scores(self, extracted_kpis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates normalized KPI scores, category scores, and total health score.
        Also identifies missing mandatory/non-mandatory KPIs.
        """
        normalized_kpis = {}
        missing_mandatory_kpis = []
        missing_non_mandatory_kpis = []

        # 1. Normalize KPIs and identify missing ones
        for category_data in self.kpi_weights['kpi_weights_within_category'].values():
            for kpi_config in category_data: # kpi_config is a dict like {"kpi": "MRR", "weight": 0.25}
                kpi_name = kpi_config['kpi']
                is_mandatory = kpi_config.get('mandatory', False)
                
                if kpi_name in extracted_kpis and extracted_kpis[kpi_name] is not None:
                    try:
                        # Pass all extracted_kpis to normalizer if needed for compound KPIs
                        normalized_score = self.kpi_normalizer.normalize_kpi(kpi_name, extracted_kpis[kpi_name], extracted_kpis)
                        normalized_kpis[kpi_name] = normalized_score
                        logger.debug(f"Normalized {kpi_name}: {extracted_kpis[kpi_name]} -> {normalized_score:.2f}")
                    except ValueError as e:
                        logger.warning(f"Could not normalize KPI '{kpi_name}' with value '{extracted_kpis[kpi_name]}': {e}")
                        if is_mandatory:
                            missing_mandatory_kpis.append(kpi_name)
                        else:
                            missing_non_mandatory_kpis.append(kpi_name)
                        normalized_kpis[kpi_name] = 0 # Assign 0 if normalization fails
                else:
                    if is_mandatory:
                        missing_mandatory_kpis.append(kpi_name)
                        logger.warning(f"Mandatory KPI '{kpi_name}' is missing.")
                    else:
                        missing_non_mandatory_kpis.append(kpi_name)
                        logger.info(f"Non-mandatory KPI '{kpi_name}' is missing.")
                    normalized_kpis[kpi_name] = 0 # Assign 0 for missing KPIs

        # If any mandatory KPIs are missing, the total score cannot be reliably calculated.
        # However, we still proceed to calculate what we can and flag the issue.
        if missing_mandatory_kpis:
            logger.error(f"Cannot calculate full health score due to missing mandatory KPIs: {missing_mandatory_kpis}")

        # 2. Calculate Category Scores
        category_scores = {}
        for category, kpis_in_category_list in self.kpi_weights['kpi_weights_within_category'].items():
            category_total_score = 0
            category_total_weight = 0
            
            for kpi_config in kpis_in_category_list: # Iterate through the list of KPI dicts
                kpi_name = kpi_config['kpi']
                weight = kpi_config['weight']
                
                category_total_weight += weight
                
                normalized_score = normalized_kpis.get(kpi_name, 0) # Use 0 if KPI was missing/failed normalization
                category_total_score += normalized_score * weight
            
            # Prevent division by zero if a category has no KPIs or weights
            if category_total_weight > 0:
                category_scores[category] = category_total_score / category_total_weight
            else:
                category_scores[category] = 0
            logger.debug(f"Calculated score for category '{category}': {category_scores[category]:.2f}")

        # 3. Calculate Total Health Score
        total_health_score = 0
        overall_total_weight = 0
        # FIX: Access the 'weight' key from the category_weights dictionary
        for category, category_info in self.kpi_weights['category_weights'].items():
            weight = category_info.get('weight', 0) # Get the actual numerical weight
            overall_total_weight += weight
            total_health_score += category_scores.get(category, 0) * weight
        
        if overall_total_weight > 0:
            total_health_score = total_health_score / overall_total_weight
        else:
            total_health_score = 0
        logger.info(f"Calculated total health score: {total_health_score:.2f}")

        # 4. Determine Health Category (Percentile-based)
        health_category = self._determine_health_category(total_health_score)
        logger.info(f"Startup Health Category: {health_category['name']} ({health_category['emoji']})")

        # 5. Generate Suggestions and Warnings (now includes LLM-based)
        suggestions_and_warnings = self._generate_suggestions_and_warnings(
            total_health_score,
            category_scores,
            normalized_kpis,
            missing_mandatory_kpis,
            missing_non_mandatory_kpis,
            health_category
        )

        return {
            "total_score": total_health_score,
            "category_scores": category_scores,
            "normalized_kpis": normalized_kpis,
            "missing_mandatory_kpis": missing_mandatory_kpis,
            "missing_non_mandatory_kpis": missing_non_mandatory_kpis,
            "health_category": health_category,
            "suggestions_and_warnings": suggestions_and_warnings
        }

    def _determine_health_category(self, total_score: float) -> Dict[str, Any]:
        """
        Determines the health category based on the total score and predefined thresholds.
        """
        for category in self.percentile_thresholds['score_categories']:
            if category['min_score'] <= total_score <= category['max_score']:
                return category
        
        # Fallback for scores outside defined ranges (shouldn't happen with 0-100)
        return {
            "name": "Undetermined",
            "min_score": 0, "max_score": 100,
            "emoji": "❓",
            "description": "Could not determine health category."
        }

    def _generate_suggestions_and_warnings(
        self,
        total_score: float,
        category_scores: Dict[str, float],
        normalized_kpis: Dict[str, float],
        missing_mandatory_kpis: List[str],
        missing_non_mandatory_kpis: List[str],
        health_category: Dict[str, Any]
    ) -> List[str]:
        """
        Generates rule-based and LLM-based suggestions and warnings.
        """
        suggestions = []

        # Rule-based suggestions (kept as a baseline)
        suggestions.append(f"**Overall Health:** {health_category['emoji']} {health_category['name']} - {health_category['description']}")

        if missing_mandatory_kpis:
            suggestions.append(f"🚨 **Critical Warning:** The following mandatory KPIs were missing: {', '.join(missing_mandatory_kpis)}. This significantly impacts the accuracy and completeness of the score. Please ensure these metrics are provided.")
        
        if missing_non_mandatory_kpis:
            suggestions.append(f"⚠️ **Data Gap:** Consider providing the following non-mandatory KPIs for a more comprehensive assessment: {', '.join(missing_non_mandatory_kpis)}.")

        sorted_categories = sorted(category_scores.items(), key=lambda item: item[1])
        if sorted_categories:
            lowest_category, lowest_score = sorted_categories[0]
            if lowest_score < 50:
                suggestions.append(f"📉 **Area for Improvement:** '{lowest_category.replace('_', ' ').title()}' category scored lowest ({lowest_score:.2f}/100). Focus on metrics within this area.")
                
                kpis_in_lowest_category = self.kpi_weights['kpi_weights_within_category'].get(lowest_category, [])
                low_kpis_in_category = []
                for kpi_config in kpis_in_lowest_category:
                    kpi_name = kpi_config['kpi']
                    if normalized_kpis.get(kpi_name, 0) < 30:
                        low_kpis_in_category.append(kpi_name.replace('_', ' ').title())
                if low_kpis_in_category:
                    suggestions.append(f"   - Specifically, review performance in: {', '.join(low_kpis_in_category)}.")

        # LLM-based suggestions
        if self.openai_client: # Check if client was successfully initialized
            llm_structured_insights = self._call_llm_for_suggestions(
                total_score,
                category_scores,
                normalized_kpis,
                missing_mandatory_kpis,
                missing_non_mandatory_kpis,
                health_category
            )
            if llm_structured_insights:
                suggestions.append("---")
                suggestions.append("✨ **AI-Powered Insights & Recommendations:**")
                
                # Format and add overall assessment
                if llm_structured_insights.get("overall_assessment"):
                    suggestions.append(f"**Overall Assessment:** {llm_structured_insights['overall_assessment']}")

                # Format and add strengths
                if llm_structured_insights.get("strengths"):
                    suggestions.append("\n**Strengths by Category:**")
                    for category, strength_list in llm_structured_insights["strengths"].items():
                        suggestions.append(f"- **{category.replace('_', ' ').title()}:**")
                        for strength in strength_list:
                            suggestions.append(f"  - {strength}")

                # Format and add weaknesses
                if llm_structured_insights.get("weaknesses"):
                    suggestions.append("\n**Weaknesses by Category:**")
                    for category, weakness_list in llm_structured_insights["weaknesses"].items():
                        suggestions.append(f"- **{category.replace('_', ' ').title()}:**")
                        for weakness in weakness_list:
                            suggestions.append(f"  - {weakness}")

                # Format and add actionable recommendations
                if llm_structured_insights.get("recommendations"):
                    suggestions.append("\n**Actionable Recommendations:**")
                    for rec in llm_structured_insights["recommendations"]:
                        suggestions.append(f"- {rec}")

            else:
                suggestions.append("⚠️ Could not generate AI-powered insights. Check LLM logs.")
        else:
            suggestions.append("ℹ️ LLM API key not configured for advanced AI insights.")
        
        return suggestions

    def _call_llm_for_suggestions(
        self,
        total_score: float,
        category_scores: Dict[str, float],
        normalized_kpis: Dict[str, float],
        missing_mandatory_kpis: List[str],
        missing_non_mandatory_kpis: List[str],
        health_category: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calls the LLM (OpenAI GPT) to generate nuanced suggestions and warnings.
        Returns a structured dictionary of insights.
        """
        logger.info("Calling LLM for AI-powered suggestions...")
        try:
            # Prepare a concise summary for the LLM
            summary_for_llm = {
                "total_score": f"{total_score:.2f}",
                "health_category": health_category['name'],
                "category_scores": {k: f"{v:.2f}" for k, v in category_scores.items()},
                "normalized_kpis": {k: f"{v:.2f}" for k, v in normalized_kpis.items()},
                "missing_mandatory_kpis": missing_mandatory_kpis,
                "missing_non_mandatory_kpis": missing_non_mandatory_kpis
            }

            prompt = (
                "Based on the following startup health analysis, provide a structured assessment. "
                "Your response MUST be a valid JSON object with the following keys:\n"
                "- `overall_assessment`: A concise paragraph summarizing the overall health and key takeaways.\n"
                "- `strengths`: A dictionary where keys are category names (e.g., 'Financial Health') and values are lists of bullet-point strings describing strengths in that category.\n"
                "- `weaknesses`: A dictionary structured similarly to `strengths`, describing areas for improvement.\n"
                "- `recommendations`: A list of 3-5 concise, actionable recommendations. Each recommendation should start with an emoji (e.g., '💡', '📈', '📉', '⚠️').\n\n"
                "Here is the analysis data:\n"
                f"{json.dumps(summary_for_llm, indent=2)}\n\n"
                "JSON Output:"
            )

            chat_completion = self.openai_client.chat.completions.create(
                model=self.llm_model_for_suggestions,
                messages=[
                    {"role": "system", "content": "You are an expert business analyst providing structured insights on startup health. Your output must be a valid JSON object as specified in the prompt."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, # Instruct OpenAI to return JSON
                max_tokens=1024, # Increased max tokens for more detailed response
                temperature=0.4 # Slightly higher temperature for more nuanced analysis, but still focused
            )
            
            if chat_completion.choices and chat_completion.choices[0].message and chat_completion.choices[0].message.content:
                raw_text_content = chat_completion.choices[0].message.content
                try:
                    llm_insights = json.loads(raw_text_content)
                    # Basic validation of the expected structure
                    if isinstance(llm_insights, dict) and \
                       "overall_assessment" in llm_insights and \
                       "strengths" in llm_insights and \
                       "weaknesses" in llm_insights and \
                       "recommendations" in llm_insights:
                        logger.info("Successfully received and parsed structured LLM insights.")
                        return llm_insights
                    else:
                        logger.warning(f"LLM response structure was unexpected: {llm_insights}")
                        return {"overall_assessment": "LLM returned unexpected format.", "strengths": {}, "weaknesses": {}, "recommendations": []}
                except json.JSONDecodeError as e:
                    logger.error(f"LLM response was not valid JSON for insights: {raw_text_content}. Error: {e}")
                    return {"overall_assessment": f"LLM returned invalid JSON: {e}", "strengths": {}, "weaknesses": {}, "recommendations": []}
            else:
                logger.warning(f"LLM returned no candidates or empty response for insights: {chat_completion}")
                return {"overall_assessment": "LLM returned no content for insights.", "strengths": {}, "weaknesses": {}, "recommendations": []}

        except openai.APIError as e:
            logger.error(f"OpenAI API error for insights: {e}")
            return {"overall_assessment": f"LLM API call failed: {e}", "strengths": {}, "weaknesses": {}, "recommendations": []}
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM insight generation: {e}", exc_info=True)
            return {"overall_assessment": f"Unexpected error during AI insight generation: {e}", "strengths": {}, "weaknesses": {}, "recommendations": []}

