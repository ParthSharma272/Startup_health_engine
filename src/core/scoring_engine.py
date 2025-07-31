# src/core/scoring_engine.py (updated sections)
import json
import os
from typing import Dict, Any, List
from src.core.config_loader import ConfigLoader
from src.core.kpi_normalizer import KPINormalizer
from src.ml_models.health_predictor import HealthPredictor  # Import the ML model
from src.utils.logger_config import logger
import openai

class ScoringEngine:
    def __init__(self, config_loader: ConfigLoader, openai_api_key: str = None):
        self.config_loader = config_loader
        self.kpi_weights = self.config_loader.load_config(self.config_loader.kpi_weights_path)
        self.kpi_benchmarks = self.config_loader.load_config(self.config_loader.kpi_benchmarks_path)
        self.kpi_benchmark_map = self.config_loader.get_kpi_benchmark_map(self.kpi_benchmarks)
        self.kpi_normalizer = KPINormalizer(self.kpi_benchmark_map)
        
        # Load percentile thresholds for score categorization
        self.percentile_thresholds = self.config_loader.load_config(self.config_loader.percentile_thresholds_path)
        
        self.openai_api_key = openai_api_key
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            self.llm_model_for_suggestions = "gpt-3.5-turbo"
            logger.info(f"OpenAI client initialized for ScoringEngine with model: {self.llm_model_for_suggestions}.")
        else:
            self.openai_client = None
            logger.warning("OPENAI_API_KEY not provided to ScoringEngine. LLM-based suggestions will be skipped.")
        
        # Initialize ML model for confidence prediction
        self.health_predictor = HealthPredictor()
        logger.info("ScoringEngine initialized with ML confidence predictor.")

    def calculate_scores(self, extracted_kpis: Dict[str, Any], document_name: str = "unknown") -> Dict[str, Any]:
        """
        Calculates normalized KPI scores, category scores, total health score, and confidence score.
        Also identifies missing mandatory/non-mandatory KPIs.
        """
        normalized_kpis = {}
        missing_mandatory_kpis = []
        missing_non_mandatory_kpis = []

        # 1. Normalize KPIs and identify missing ones
        for category_data in self.kpi_weights['kpi_weights_within_category'].values():
            for kpi_config in category_data:
                kpi_name = kpi_config['kpi']
                is_mandatory = kpi_config.get('mandatory', False)
                
                if kpi_name in extracted_kpis and extracted_kpis[kpi_name] is not None:
                    try:
                        normalized_score = self.kpi_normalizer.normalize_kpi(kpi_name, extracted_kpis[kpi_name], extracted_kpis)
                        normalized_kpis[kpi_name] = normalized_score
                        logger.debug(f"Normalized {kpi_name}: {extracted_kpis[kpi_name]} -> {normalized_score:.2f}")
                    except ValueError as e:
                        logger.warning(f"Could not normalize KPI '{kpi_name}' with value '{extracted_kpis[kpi_name]}': {e}")
                        if is_mandatory:
                            missing_mandatory_kpis.append(kpi_name)
                        else:
                            missing_non_mandatory_kpis.append(kpi_name)
                        normalized_kpis[kpi_name] = 0
                else:
                    if is_mandatory:
                        missing_mandatory_kpis.append(kpi_name)
                        logger.warning(f"Mandatory KPI '{kpi_name}' is missing.")
                    else:
                        missing_non_mandatory_kpis.append(kpi_name)
                        logger.info(f"Non-mandatory KPI '{kpi_name}' is missing.")
                    normalized_kpis[kpi_name] = 0

        if missing_mandatory_kpis:
            logger.error(f"Cannot calculate full health score due to missing mandatory KPIs: {missing_mandatory_kpis}")

        # 2. Calculate Category Scores
        category_scores = {}
        for category, kpis_in_category_list in self.kpi_weights['kpi_weights_within_category'].items():
            category_total_score = 0
            category_total_weight = 0
            
            for kpi_config in kpis_in_category_list:
                kpi_name = kpi_config['kpi']
                weight = kpi_config['weight']
                
                category_total_weight += weight
                normalized_score = normalized_kpis.get(kpi_name, 0)
                category_total_score += normalized_score * weight
            
            if category_total_weight > 0:
                category_scores[category] = category_total_score / category_total_weight
            else:
                category_scores[category] = 0
            logger.debug(f"Calculated score for category '{category}': {category_scores[category]:.2f}")

        # 3. Calculate Total Health Score
        total_health_score = 0
        overall_total_weight = 0
        for category, category_info in self.kpi_weights['category_weights'].items():
            weight = category_info.get('weight', 0)
            overall_total_weight += weight
            total_health_score += category_scores.get(category, 0) * weight
        
        if overall_total_weight > 0:
            total_health_score = total_health_score / overall_total_weight
        else:
            total_health_score = 0
        logger.info(f"Calculated total health score: {total_health_score:.2f}")

        # 4. Determine Health Category
        health_category = self._determine_health_category(total_health_score)
        logger.info(f"Startup Health Category: {health_category['name']} ({health_category['emoji']})")

        # 5. Generate Confidence Score using ML model
        confidence_score, prediction_method = self.health_predictor.predict_confidence(
            normalized_kpis=normalized_kpis,
            category_scores=category_scores,
            total_score=total_health_score,
            missing_mandatory_kpis=missing_mandatory_kpis,
            missing_non_mandatory_kpis=missing_non_mandatory_kpis
        )
        logger.info(f"Confidence score: {confidence_score:.2f} (method: {prediction_method})")

        # 6. Log prediction data for future retraining
        self.health_predictor.log_prediction(
            document_name=document_name,
            normalized_kpis=normalized_kpis,
            category_scores=category_scores,
            total_score=total_health_score,
            confidence_score=confidence_score,
            prediction_method=prediction_method,
            missing_mandatory_kpis=missing_mandatory_kpis,
            missing_non_mandatory_kpis=missing_non_mandatory_kpis
        )

        # 7. Generate Suggestions and Warnings
        suggestions_and_warnings_output = self._generate_suggestions_and_warnings(
            total_health_score,
            category_scores,
            normalized_kpis,
            missing_mandatory_kpis,
            missing_non_mandatory_kpis,
            health_category,
            confidence_score  # Pass confidence score to suggestions
        )

        return {
            "total_score": total_health_score,
            "category_scores": category_scores,
            "normalized_kpis": normalized_kpis,
            "missing_mandatory_kpis": missing_mandatory_kpis,
            "missing_non_mandatory_kpis": missing_non_mandatory_kpis,
            "health_category": health_category,
            "confidence_score": confidence_score,  # Add confidence score to output
            "prediction_method": prediction_method,  # Add prediction method to output
            "suggestions_and_warnings": suggestions_and_warnings_output
        }

    def _generate_suggestions_and_warnings(
        self,
        total_score: float,
        category_scores: Dict[str, float],
        normalized_kpis: Dict[str, float],
        missing_mandatory_kpis: List[str],
        missing_non_mandatory_kpis: List[str],
        health_category: Dict[str, Any],
        confidence_score: float  # Add confidence score parameter
    ) -> List[Any]:
        """
        Generates rule-based and LLM-based suggestions and warnings.
        """
        suggestions_list = []

        # Add confidence score information
        if confidence_score >= 80:
            confidence_emoji = "🟢"
            confidence_text = "High"
        elif confidence_score >= 60:
            confidence_emoji = "🟡"
            confidence_text = "Medium"
        else:
            confidence_emoji = "🔴"
            confidence_text = "Low"
            
        suggestions_list.append(f"**Assessment Confidence:** {confidence_emoji} {confidence_text} ({confidence_score:.1f}/100)")

        # Rule-based suggestions
        suggestions_list.append(f"**Overall Health:** {health_category['emoji']} {health_category['name']} - {health_category['description']}")

        if missing_mandatory_kpis:
            suggestions_list.append(f"🚨 **Critical Warning:** The following mandatory KPIs were missing: {', '.join(missing_mandatory_kpis)}. This significantly impacts the accuracy and completeness of the score. Please ensure these metrics are provided.")
        
        if missing_non_mandatory_kpis:
            suggestions_list.append(f"⚠️ **Data Gap:** Consider providing the following non-mandatory KPIs for a more comprehensive assessment: {', '.join(missing_non_mandatory_kpis)}.")

        sorted_categories = sorted(category_scores.items(), key=lambda item: item[1])
        if sorted_categories:
            lowest_category, lowest_score = sorted_categories[0]
            if lowest_score < 50:
                suggestions_list.append(f"📉 **Area for Improvement:** '{lowest_category.replace('_', ' ').title()}' category scored lowest ({lowest_score:.2f}/100). Focus on metrics within this area.")
                
                kpis_in_lowest_category = self.kpi_weights['kpi_weights_within_category'].get(lowest_category, [])
                low_kpis_in_category = []
                for kpi_config in kpis_in_lowest_category:
                    kpi_name = kpi_config['kpi']
                    if normalized_kpis.get(kpi_name, 0) < 30:
                        low_kpis_in_category.append(kpi_name.replace('_', ' ').title())
                if low_kpis_in_category:
                    suggestions_list.append(f"   - Specifically, review performance in: {', '.join(low_kpis_in_category)}.")

        # LLM-based suggestions
        llm_structured_insights = {}
        if self.openai_client:
            llm_structured_insights = self._call_llm_for_suggestions(
                total_score,
                category_scores,
                normalized_kpis,
                missing_mandatory_kpis,
                missing_non_mandatory_kpis,
                health_category,
                confidence_score  # Pass confidence score to LLM
            )
            if llm_structured_insights:
                suggestions_list.append(llm_structured_insights)
            else:
                suggestions_list.append({"overall_assessment": "⚠️ Could not generate AI-powered insights. Check LLM logs.", "strengths": {}, "weaknesses": {}, "recommendations": []})
        else:
            suggestions_list.append({"overall_assessment": "ℹ️ LLM API key not configured for advanced AI insights.", "strengths": {}, "weaknesses": {}, "recommendations": []})
        
        return suggestions_list

    def _call_llm_for_suggestions(
        self,
        total_score: float,
        category_scores: Dict[str, float],
        normalized_kpis: Dict[str, float],
        missing_mandatory_kpis: List[str],
        missing_non_mandatory_kpis: List[str],
        health_category: Dict[str, Any],
        confidence_score: float  # Add confidence score parameter
    ) -> Dict[str, Any]:
        """
        Calls the LLM (OpenAI GPT) to generate nuanced suggestions and warnings.
        Returns a structured dictionary of insights.
        """
        logger.info("Calling LLM for AI-powered suggestions...")
        try:
            summary_for_llm = {
                "total_score": f"{total_score:.2f}",
                "health_category": health_category['name'],
                "confidence_score": f"{confidence_score:.2f}",
                "category_scores": {k: f"{v:.2f}" for k, v in category_scores.items()},
                "normalized_kpis": {k: f"{v:.2f}" for k, v in normalized_kpis.items()},
                "missing_mandatory_kpis": missing_mandatory_kpis,
                "missing_non_mandatory_kpis": missing_non_mandatory_kpis
            }

            prompt = (
                "Based on the following startup health analysis, provide a structured assessment. "
                "Your response MUST be a valid JSON object with the following keys:\n"
                "- `overall_assessment`: A concise paragraph summarizing the overall health, confidence, and key takeaways.\n"
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
                response_format={"type": "json_object"},
                max_tokens=1024,
                temperature=0.4
            )
            
            if chat_completion.choices and chat_completion.choices[0].message and chat_completion.choices[0].message.content:
                raw_text_content = chat_completion.choices[0].message.content
                try:
                    llm_insights = json.loads(raw_text_content)
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

    def _determine_health_category(self, total_score: float) -> Dict[str, Any]:
        """
        Determines the health category based on the total score using percentile thresholds.
        
        Args:
            total_score: The calculated total health score (0-100)
            
        Returns:
            Dictionary with category information including name, emoji, and description
        """
        try:
            # Load percentile thresholds
            thresholds = self.percentile_thresholds.get('health_categories', [])
            
            # Sort thresholds by min_score in descending order to check from highest to lowest
            sorted_thresholds = sorted(thresholds, key=lambda x: x.get('min_score', 0), reverse=True)
            
            # Find the appropriate category
            for category in sorted_thresholds:
                min_score = category.get('min_score', 0)
                if total_score >= min_score:
                    return {
                        'name': category.get('name', 'Unknown'),
                        'emoji': category.get('emoji', '❓'),
                        'description': category.get('description', 'No description available')
                    }
            
            # Fallback if no category matches (shouldn't happen with proper thresholds)
            return {
                'name': 'Unknown',
                'emoji': '❓',
                'description': 'Unable to determine health category'
            }
            
        except Exception as e:
            logger.error(f"Error determining health category for score {total_score}: {e}")
            return {
                'name': 'Error',
                'emoji': '⚠️',
                'description': 'Error occurred while determining health category'
            }