import synalinks
import asyncio
import json
import os

FINAL_SUBMISSION = {}

SUBMISSION_PROGRAM_LIBRARY_FOLDER = "submission_programs"

###############################################################################
## Reward Function
###############################################################################

@synalinks.saving.register_synalinks_serializable()
async def grid_similarity(y_true, y_pred):
    """
    Compute similarity between grids by counting matching cells.
    Returns a value between 0.0 and 1.0 where 1.0 is perfect match.
    """
    try:
        true_array = (
            np.array(y_true.get("output_grid"), dtype=np.int32)
            if y_true.get("output_grid")
            else np.array([], dtype=np.int32).reshape(0, 0)
        )
        pred_array = (
            np.array(y_pred.get("output_grid"), dtype=np.int32)
            if y_pred.get("output_grid")
            else np.array([], dtype=np.int32).reshape(0, 0)
        )

        true_shape = true_array.shape
        pred_shape = pred_array.shape

        min_height = min(true_shape[0], pred_shape[0])
        min_width = min(true_shape[1], pred_shape[1])

        # Handle empty grids
        if min_height == 0 or min_width == 0:
            total_cells = max(
                true_shape[0] * true_shape[1], pred_shape[0] * pred_shape[1]
            )
            if total_cells == 0:
                return 1.0
            return 0.0

        # Calculate overlap errors
        true_overlap = true_array[:min_height, :min_width]
        pred_overlap = pred_array[:min_height, :min_width]
        diff = np.abs(true_overlap - pred_overlap)
        overlap_errors = np.count_nonzero(diff)
        
        # Calculate size difference penalties
        true_only_cells = (true_shape[0] * true_shape[1]) - (min_height * min_width)
        pred_only_cells = (pred_shape[0] * pred_shape[1]) - (min_height * min_width)
        total_errors = overlap_errors + true_only_cells + pred_only_cells
        total_cells = max(true_shape[0] * true_shape[1], pred_shape[0] * pred_shape[1])
        total_cells = max(total_cells, min_height * min_width)

        reward = 1.0 - (total_errors / total_cells)
        return float(min(max(0.0, reward), 1.0))
    except Exception as e:
        # print(f"⚠️ Error in grid_similarity: {e}")
        return 0.0

async def main():
    
    for task_name in synalinks.datasets.arcagi.get_arcagi2_evaluation_task_names():
        
        task_checkpoint_filepath = os.path.join(SUBMISSION_PROGRAM_LIBRARY_FOLDER, f"{task_name}.json")
        
        _, (x_test, y_test) = synalinks.datasets.arcagi.load_data(
            task_name=task_name,
            arc_version=2,
            one_leave_out=False,
            permutation=False,
            repeat=1,
            curriculum_learning=True,
        )
            
        if task_name not in FINAL_SUBMISSION:
            FINAL_SUBMISSION[task_name] = []
            
        if os.path.exists(task_checkpoint_filepath):
            program = synalinks.Program.load(task_checkpoint_filepath)
            results = await program.predict(x=x_test)
        
            for i, result in enumerate(results):
                if len(FINAL_SUBMISSION[task_name]) > i:
                    FINAL_SUBMISSION[task_name][i] = {
                        "attempt_1": result.get("output_grid"),
                        "attempt_2": result.get("output_grid"),
                    }
                else:
                    FINAL_SUBMISSION[task_name].append(
                        {
                            "attempt_1": result.get("output_grid"),
                            "attempt_2": result.get("output_grid"),
                        }
                    )
        else:
            for i, _ in enumerate(x_test):
                if len(FINAL_SUBMISSION[task_name]) > i:
                    FINAL_SUBMISSION[task_name][i] = {
                        "attempt_1": [[]],
                        "attempt_2": [[]],
                    }
                else:
                    FINAL_SUBMISSION[task_name].append(
                        {
                            "attempt_1": [[]],
                            "attempt_2": [[]],
                        }
                    )
    
    with open("submission.json", "w") as f:
        f.write(json.dumps(FINAL_SUBMISSION))
    
if __name__ == "__main__":
    asyncio.run(main())