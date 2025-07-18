import json
import os
import argparse
from src.core.config_loader import ConfigLoader
from src.core.kpi_extractor import KPIExtractor
from src.core.kpi_rag_extractor import KPIRAGExtractor
from src.core.scoring_engine import ScoringEngine
from src.utils.logger_config import logger, setup_logging

def main(): # Changed back to synchronous
    """
    Main function to run the full startup health scoring pipeline locally.
    Processes a document from the 'uploads/' directory, extracts KPIs using RAG,
    and calculates health scores.
    """
    # Setup logging for the main script
    setup_logging()

    logger.info("Starting Full Startup Health Scoring Pipeline (Local Mode).")

    # Define paths relative to the script's execution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, 'config')
    uploads_dir = os.path.join(base_dir, 'uploads')
    processed_data_dir = os.path.join(base_dir, 'processed_data')
    
    # Ensure processed_data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    extracted_text_output_path = os.path.join(processed_data_dir, 'extracted_document_content.txt')
    extracted_kpis_output_path = os.path.join(processed_data_dir, 'extracted_kpis.json')
    final_score_output_path = os.path.join(processed_data_dir, 'startup_score_output.json')


    # Setup argument parser
    parser = argparse.ArgumentParser(description="Startup Health Scoring Pipeline.")
    parser.add_argument(
        "--document_path",
        type=str,
        required=True,
        help="Path to the document file in the 'uploads/' directory (e.g., 'sample_document.txt', 'image.png', 'report.pdf')."
    )
    args = parser.parse_args()

    full_document_path = os.path.join(uploads_dir, args.document_path)
    raw_extracted_content = None
    kpi_dict = None
    scoring_results = None

    try:
        # --- Step 1: Extract Raw Text (OCR/PDF/Text) ---
        text_extractor = KPIExtractor()
        raw_extracted_content = text_extractor.extract_text_from_document(full_document_path)

        if not raw_extracted_content:
            logger.error(f"Failed to extract any content from {full_document_path}. Cannot proceed.")
            return

        with open(extracted_text_output_path, 'w', encoding='utf-8') as f:
            f.write(raw_extracted_content)
        logger.info(f"Successfully extracted raw text and saved to: {extracted_text_output_path}")
        logger.info("\n--- Extracted Content Preview (first 500 chars) ---")
        logger.info(raw_extracted_content[:500] + ("..." if len(raw_extracted_content) > 500 else ""))
        logger.info("----------------------------------------------------")

        # --- Step 2: Extract KPIs using RAG (LLM) ---
        config_loader = ConfigLoader(config_dir=config_dir)
        kpi_benchmarks = config_loader.load_config(config_loader.kpi_benchmarks_path)
        
        rag_extractor = KPIRAGExtractor(kpi_benchmarks)
        kpi_dict = rag_extractor.extract_kpis(raw_extracted_content) # No await needed

        if not kpi_dict:
            logger.error("Failed to extract KPIs using RAG. Cannot proceed to scoring.")
            return

        with open(extracted_kpis_output_path, 'w', encoding='utf-8') as f:
            json.dump(kpi_dict, f, indent=2)
        logger.info(f"Successfully extracted KPIs via RAG and saved to: {extracted_kpis_output_path}")
        logger.info(f"Extracted KPI Dict: {json.dumps(kpi_dict, indent=2)}")


        # --- Step 3: Calculate Health Scores ---
        scoring_engine = ScoringEngine(config_loader=config_loader)
        scoring_results = scoring_engine.calculate_scores(kpi_dict)

        with open(final_score_output_path, 'w', encoding='utf-8') as f:
            json.dump(scoring_results, f, indent=2)
        logger.info(f"Scoring results saved to: {final_score_output_path}")

        logger.info("\n--- Scoring Results ---")
        if "error" in scoring_results:
            logger.error(f"Scoring failed: {scoring_results['error']}")
            if "missing_kpis" in scoring_results:
                logger.error(f"Missing mandatory KPIs: {', '.join(scoring_results['missing_kpis'])}")
        else:
            logger.info("Normalized KPIs:")
            for kpi, score in scoring_results["normalized_kpis"].items():
                logger.info(f"  {kpi}: {score:.2f}")

            logger.info("\nCategory Scores:")
            for category, score in scoring_results["category_scores"].items():
                logger.info(f"  {category}: {score:.2f}")

            logger.info(f"\nTotal Score: {scoring_results['total_score']:.2f}")

    except FileNotFoundError as e:
        logger.critical(f"A required file was not found: {e}")
    except json.JSONDecodeError as e:
        logger.critical(f"Error parsing JSON configuration or data file: {e}")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during pipeline execution: {e}", exc_info=True)

    logger.info("Full Startup Health Scoring Pipeline finished.")

if __name__ == "__main__":
    main() # Call synchronous main function
