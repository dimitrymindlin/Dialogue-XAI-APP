# Experiment Analysis Pipeline Makefile
# 
# This Makefile orchestrates the 4-step analysis pipeline:
# 1. Get raw experiment data from database and Prolific CSVs
# 2. Extract information and transform data 
# 3. Filter and clean data using all enabled filters
# 4. Generate overview plots and statistics
#
# Usage:
#   make all          - Run complete pipeline (steps 1-4)
#   make step1        - Run step 1 only (raw data collection)
#   make step2        - Run step 2 only (data extraction) 
#   make step3        - Run step 3 only (filtering and cleaning)
#   make step4        - Run step 4 only (overview generation)
#   make clean        - Remove all generated results
#   make clean-step2  - Remove step 2 results (forces re-extraction)
#   make clean-step3  - Remove step 3 results (forces re-filtering)
#   make clean-step4  - Remove step 4 results (forces re-overview)
#   make info         - Show current analysis configuration

# Conda environment to use for all Python scripts
CONDA_ENV = Dialogue-Experiment-Analysis

# Command to run Python with conda environment activated locally (for config)
PYTHON_LOCAL = conda run -n $(CONDA_ENV) python
# Command to run Python scripts with correct environment and parent directory access
PYTHON_CMD = PYTHONPATH=.. conda run -n $(CONDA_ENV) python

# Get current configuration values using conda environment (from current dir)
EXPERIMENT_HANDLE := $(shell $(PYTHON_LOCAL) -c "from analysis_config import EXPERIMENT_HANDLE; print(EXPERIMENT_HANDLE)")
EXPERIMENT_DATE := $(shell $(PYTHON_LOCAL) -c "from analysis_config import EXPERIMENT_DATE; print(EXPERIMENT_DATE)")
DATASET := $(shell $(PYTHON_LOCAL) -c "from analysis_config import DATASET; print(DATASET)")
GROUP := $(shell $(PYTHON_LOCAL) -c "from analysis_config import GROUP; print(GROUP)")
RESULTS_DIR := $(shell $(PYTHON_LOCAL) -c "from analysis_config import RESULTS_DIR; print(RESULTS_DIR)")
PROLIFIC_CSV_FOLDER := $(shell $(PYTHON_LOCAL) -c "from analysis_config import PROLIFIC_CSV_FOLDER; print(PROLIFIC_CSV_FOLDER)")

# Define stage directories
RAW_DIR = $(RESULTS_DIR)/1_raw
UNPACKED_DIR = $(RESULTS_DIR)/2_unpacked  
FILTERED_DIR = $(RESULTS_DIR)/3_filtered
OVERVIEW_DIR = $(RESULTS_DIR)/4_overview

# Define key output files for dependency tracking  
RAW_COMPLETE = $(RAW_DIR)/users_raw.csv
UNPACKED_COMPLETE = $(UNPACKED_DIR)/users_unpacked.csv
FILTERED_COMPLETE = $(FILTERED_DIR)/users_filtered.csv
OVERVIEW_COMPLETE = $(OVERVIEW_DIR)/score_distributions.png

# Default target
.PHONY: all
all: step4
	@echo ""
	@echo "🎉 Analysis pipeline complete!"
	@echo "Results saved in: $(RESULTS_DIR)"
	@echo ""
	@echo "Generated files:"
	@echo "  Raw data:      $(RAW_DIR)/"
	@echo "  Unpacked data: $(UNPACKED_DIR)/"  
	@echo "  Filtered data: $(FILTERED_DIR)/"
	@echo "  Overview:      $(OVERVIEW_DIR)/"

# Step 1: Get raw experiment data
.PHONY: step1
step1: $(RAW_COMPLETE)

$(RAW_COMPLETE): 1_get_raw_experiment_data.py analysis_config.py
	@echo "📥 Step 1: Collecting raw experiment data..."
	@echo "  Environment: $(CONDA_ENV)"
	@echo "  Experiment: $(DATASET)-$(GROUP) ($(EXPERIMENT_HANDLE)-$(EXPERIMENT_DATE))"
	@echo "  Prolific folder: $(PROLIFIC_CSV_FOLDER)"
	$(PYTHON_LOCAL) 1_get_raw_experiment_data.py
	@echo "✅ Step 1 complete"

# Step 2: Extract information to columns  
.PHONY: step2
step2: $(UNPACKED_COMPLETE)

$(UNPACKED_COMPLETE): $(RAW_COMPLETE) 2_extract_information_to_columns.py analysis_config.py
	@echo "🔄 Step 2: Extracting and transforming data..."
	@echo "  Environment: $(CONDA_ENV)"
	$(PYTHON_CMD) 2_extract_information_to_columns.py
	@echo "✅ Step 2 complete"

# Step 3: Filter and clean data
.PHONY: step3  
step3: $(FILTERED_COMPLETE)

$(FILTERED_COMPLETE): $(UNPACKED_COMPLETE) 3_filter_and_clean_data.py filter_users.py analysis_config.py
	@echo "🔍 Step 3: Filtering and cleaning data..."
	@echo "  Environment: $(CONDA_ENV)"
	@echo "  Applying filters including new invalid scores filter..."
	$(PYTHON_CMD) 3_filter_and_clean_data.py
	@echo "✅ Step 3 complete"

# Step 4: Generate overview
.PHONY: step4
step4: $(OVERVIEW_COMPLETE)

$(OVERVIEW_COMPLETE): $(FILTERED_COMPLETE) 4_get_raw_overview.py analysis_config.py
	@echo "📊 Step 4: Generating overview plots and statistics..."
	@echo "  Environment: $(CONDA_ENV)"
	$(PYTHON_CMD) 4_get_raw_overview.py
	@echo "✅ Step 4 complete"

# Information target
.PHONY: info
info:
	@echo "Current Analysis Configuration:"
	@echo "  Dataset:     $(DATASET)"
	@echo "  Group:       $(GROUP)"
	@echo "  Handle:      $(EXPERIMENT_HANDLE)"
	@echo "  Date:        $(EXPERIMENT_DATE)"
	@echo "  Results dir: $(RESULTS_DIR)"
	@echo "  Prolific:    $(PROLIFIC_CSV_FOLDER)"

# Clean targets
.PHONY: clean
clean:
	@echo "🧹 Cleaning all results..."
	rm -rf $(RESULTS_DIR)
	@echo "✅ All results cleaned"

.PHONY: clean-step2
clean-step2:
	@echo "🧹 Cleaning step 2 results (will force re-extraction)..."
	rm -rf $(UNPACKED_DIR) $(FILTERED_DIR) $(OVERVIEW_DIR)
	@echo "✅ Step 2+ results cleaned"

.PHONY: clean-step3  
clean-step3:
	@echo "🧹 Cleaning step 3 results (will force re-filtering)..."
	rm -rf $(FILTERED_DIR) $(OVERVIEW_DIR)
	@echo "✅ Step 3+ results cleaned"

.PHONY: clean-step4
clean-step4:
	@echo "🧹 Cleaning step 4 results (will force re-overview)..."
	rm -rf $(OVERVIEW_DIR)
	@echo "✅ Step 4 results cleaned"

# Help target
.PHONY: help
help:
	@echo "Experiment Analysis Pipeline Makefile"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Conda environment '$(CONDA_ENV)' must exist and be configured"
	@echo "  - Database access for raw data collection (step 1)"
	@echo ""
	@echo "Targets:"
	@echo "  all           Run complete analysis pipeline (steps 1-4)"
	@echo "  step1         Collect raw data from database and Prolific CSVs"
	@echo "  step2         Extract and transform data, unpack JSON columns"
	@echo "  step3         Filter and clean data (includes invalid scores filter)"
	@echo "  step4         Generate overview plots and statistics"
	@echo "  info          Show current analysis configuration"
	@echo "  clean         Remove all generated results"
	@echo "  clean-step2   Remove step 2+ results (forces re-extraction)"
	@echo "  clean-step3   Remove step 3+ results (forces re-filtering)"
	@echo "  clean-step4   Remove step 4 results (forces re-overview)"
	@echo "  help          Show this help message"
	@echo ""
	@echo "The pipeline uses analysis_config.py for all configuration."
	@echo "All Python scripts run in conda environment: $(CONDA_ENV)"
	@echo "Current config: $(DATASET)-$(GROUP) ($(EXPERIMENT_HANDLE)-$(EXPERIMENT_DATE))"