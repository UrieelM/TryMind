# AI-Powered CLAHE Parameter Optimizer (TryTest)

This project implements a self-improving AI agent system to find the optimal parameters (`clip_limit`, `tile_size`) for the **CLAHE** (Contrast Limited Adaptive Histogram Equalization) algorithm for image enhancement.



## 1\. Problem Description

Applying CLAHE requires tuning two key parameters: `clip_limit` and `tile_size`. Finding the optimal combination is a challenge because:

  * Incorrect parameters may have no effect or, worse, introduce visual artifacts and excessive noise.
  * The ideal combination depends on the characteristics of each individual image.
  * There is a **mathematical constraint** (`formula_result = (clip_limit * tile_size²) / 256`) that must be met for the parameters to be effective, making manual search even more complex.

This system **solves the problem by automating the search** for these parameters. It uses a loop of AI agents that iteratively propose, validate, execute, and evaluate parameters until they converge on an optimal solution based on image quality metrics.

## 2\. System Architecture

The system consists of an orchestrator (`SistemaAutoMejorable`) and three specialized agents:

### Agent 1: `AgenteEjecutor` (Executor)

  * **Role:** To execute the CLAHE operation and act as a "guardian" for the parameters.
  * **Key Function:** Before applying CLAHE, it validates that the proposed parameters comply with the critical validation formula:
      * $formula = (clip\_limit \times tile\_size^2) / 256$
  * **Failure Rules:** It rejects parameters if:
    1.  $formula < 1$ (the clipping threshold is ineffective).
    2.  $formula > clip\_limit$ (no actual clipping occurs).

### Agent 2: `AgenteEvaluador` (Evaluator)

  * **Role:** To quantify the "quality" of the processed image.
  * **Metrics:** It calculates a 2-dimensional feature vector:
    1.  **Shannon Entropy:** Measures the richness of information and detail in the image.
    2.  **Variance of Laplacian:** Measures sharpness and the presence of edges.
  * **Success Metric:** The goal is not to maximize a single metric, but to minimize the **Euclidean distance** between the current image's vector and the **average vector** of all previous valid runs. This helps the system converge toward a stable and balanced result.

### Agent 3: `AgenteOptimizador` (The Brain)

  * **Role:** To analyze historical results and propose new parameters.
  * **Technology:** Uses an LLM (`gpt-4o`) via the OpenAI API.
  * **Decision Process:** It is fed the complete optimization history:
      * **Valid Attempts:** Which parameters produced which metrics and what `distancia_promedio` (average distance).
      * **Invalid Attempts:** Which parameters failed and *why* (the reason from the `AgenteEjecutor`).
  * **Output:** Generates a JSON analysis with its reasoning and the new parameters to try, balancing **exploration** (trying new ideas) and **exploitation** (refining the best results).

## 3\. Execution Instructions

### Prerequisites

  * Python 3.7+
  * Python dependencies

### Installation

1.  Clone the repository.
2.  Install the necessary dependencies:
    ```bash
    pip install requirements.txt
    ```
3.  Create a `.env` file in the project root to store your OpenAI API key:
    ```
    API_KEY="sk-..."
    ```

### Running the Project

1.  Ensure you have a test image (e.g., in `src/test1.jpg`).
2.  Run the main script:
    ```bash
    python your_script_name.py
    ```
3.  The system will run the 10 optimization iterations. Upon completion, you will find:
      * The images (`01_original.jpg`, `02_optimizado.jpg`, `03_comparacion.jpg`) in the `outputs/` folder.
      * A detailed report of each cycle in `reporte_optimizacion.json`.

## 4\. Self-Improvement Loop Explained

The system learns and adapts with each iteration. This is the exact flow based on the report:

1.  **Run 1 (Learning from Failure):**

      * **Initial Proposal:** `clip_limit=2.0`, `tile_size=8`.
      * **Executor:** Fails. Validation determines `Formula result = 0`, which is `< 1`.
      * **Optimizer (AI):** Receives the error: "ineffective threshold". It analyzes the formula and reasons that `tile_size` was too small. It proposes a correction: `clip=3.0, tile=16`.

2.  **Run 2 (First Baseline):**

      * **Proposal (from AI):** `clip=3.0, tile=16`.
      * **Executor:** Success\! `Formula result = 3`, which is valid.
      * **Evaluator:** Calculates the vector. The `distancia_promedio` (compared to the original) is **587.25**.
      * **Optimizer (AI):** Receives the success and the distance. It decides to explore by increasing the `clip_limit` to `4.0`.

3.  **Run 3-7 (Exploration):**

      * The system tries different `clip_limit` values (`4.0`, `4.5`, `5.0`, `3.5`, `3.25`).
      * The `average_vector` is updated at each step, making the target more stable.
      * The AI observes that `clip_limit` values that are too high (like `5.0`) worsen the distance (Score: -278.6), while lower values improve it (Run 6, `clip=3.5`, Score: -152.3).

4.  **Run 8 (Convergence / Exploitation):**

      * **Proposal (from AI):** The agent reasons that the optimum is between `3.5` and `4.0`. It proposes `clip=3.75`.
      * **Evaluator:** Great improvement\! The distance drops dramatically to **17.03**.
      * **Optimizer (AI):** Confirms it is in the right zone and decides to "exploit" (refine) this value, proposing `3.6`.

5.  **Run 9-10 (Fine-Tuning):**

      * Run 9 (`clip=3.6`) turns out to be worse (Distance: `70.59`).
      * The AI analyzes this and determines the optimum must be *above* `3.75`. It proposes `clip=3.8`.
      * **Run 10 (Optimum):** `clip=3.8, tile=16`. The best distance of the cycle is achieved: **5.77**.

## 5\. Improvement Metrics

The quantifiable evidence of improvement is found in the objective metric: `distancia_promedio` (distance to the average vector `[Entropy, Laplacian Var.]`). **A lower value is better.**

Based on the `reporte_optimizacion.json`, the system demonstrated significant learning and improvement:

| Run | Parameters (Clip, Tile) | `distancia_promedio` (Error) | AI Analysis (Strategy) |
| :--- | :--- | :--- | :--- |
| 1 | (2.0, 8) | **INVALID** (`Formula < 1`) | Exploration (Error Correction) |
| 2 | (3.0, 16) | 587.25 | Exploration |
| 8 | (3.75, 16) | 17.03 | Exploitation (Refinement) |
| 10 | **(3.8, 16)** | **5.77** (Optimum) | Exploitation (Fine-Tuning) |

### Quantifiable Conclusion

The self-improvement system successfully:

  * **Identified and corrected** mathematically invalid parameters (Run 1).
  * **Reduced the error metric (distance) from 587.25 to 5.77** over 9 valid iterations.
  * **Successfully converged** on the optimal parameters (`clip_limit=3.8`, `tile_size=16`) for this image.