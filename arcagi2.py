
import synalinks
import asyncio
import os
from enum import Enum
from typing import List
import numpy as np
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

PROGRAM_LIBRARY = {}
SUBMISSION_PROGRAM_LIBRARY = {}

# Mutexes for shared resources
PROGRAM_LIBRARY_LOCK = threading.Lock()
SUBMISSION_PROGRAM_LIBRARY_LOCK = threading.Lock()

###############################################################################
## Parameters
###############################################################################

# Hyperparameters
POPULATION_SIZE = 10
K_NEAREST_FITTER = 5
MUTATION_TEMPERATURE = 1.0
CROSSOVER_TEMPERATURE = 1.0
MERGING_RATE = 0.02

NB_MAX_SEEDS = 4
SEED_THRESHOLD = 0.7

# Parameters for building the datasets
ONE_LEAVE_OUT = False
CURRICULUM = True
PERMUTATION = False
REPEAT = 1

# Where to store the learned programs
PROGRAM_LIBRARY_FOLDER = "program_library"
SUBMISSION_PROGRAM_LIBRARY_FOLDER = "submission_programs"

###############################################################################
## Language & Embedding Models
###############################################################################

language_model = synalinks.LanguageModel(
    model="xai/grok-code-fast-1",
    caching=False,
)

embedding_model = synalinks.EmbeddingModel(
    model="ollama/mxbai-embed-large",
    caching=True,
)

###############################################################################
## Data Models
###############################################################################


class Color(int, Enum):
    """ARC-AGI color palette"""
    BLACK: int = 0
    BLUE: int = 1
    RED: int = 2
    GREEN: int = 3
    YELLOW: int = 4
    GRAY: int = 5
    MAGENTA: int = 6
    ORANGE: int = 7
    LIGHT_BLUE: int = 8
    DARK_RED: int = 9


class ARCAGITask(synalinks.DataModel):
    """Single transformation example"""
    input_grid: List[List[Color]] = synalinks.Field(
        description="The input grid (list of integer list)",
    )
    output_grid: List[List[Color]] = synalinks.Field(
        description="The output grid (list of integer list)",
    )


class ARCAGIInput(synalinks.DataModel):
    """Input for the ARC-AGI solver"""
    examples: List[ARCAGITask] = synalinks.Field(
        description="A set of transformation examples",
    )
    input_grid: List[List[Color]] = synalinks.Field(
        description="The input grid (list of integer list)",
    )


class ARCAGIOutput(synalinks.DataModel):
    """Output from the ARC-AGI solver"""
    output_grid: List[List[Color]] = synalinks.Field(
        description="The output grid (list of integer list)",
    )


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


###############################################################################
## Program Creation
###############################################################################


def get_default_python_script() -> str:
    """Return the default python script template for transformation"""
    return """
# TODO implement utility functions

def transform(inputs):
    # TODO implement the python function to transform the input grid into the output grid
    # Don't hesitate to implement utility functions to help you
    return {"output_grid": inputs.get("input_grid")}
    
result = transform(inputs)
"""


async def build_and_compile_solver(
    language_model: synalinks.LanguageModel,
    embedding_model: synalinks.EmbeddingModel,
    python_script: str,
    seed_scripts: List[str],
    task_name: str,
    verbose: bool = False,
) -> synalinks.Program:
    """
    Build and compile a solver program for an ARC-AGI task.
    
    Args:
        language_model: The language model to use for synthesis
        embedding_model: The embedding model for optimization
        python_script: The default python script
        seed_scripts: List of seed scripts from similar tasks
        task_name: Name of the task
        verbose: Whether to print debug info
        
    Returns:
        A compiled synalinks Program
    """
    if verbose:
        print(f"🔧 Building solver for task {task_name}...")
    
    inputs = synalinks.Input(data_model=ARCAGIInput)
    
    outputs = await synalinks.PythonSynthesis(
        data_model=ARCAGIOutput,
        python_script=get_default_python_script(),
        seed_scripts=seed_scripts if seed_scripts else None,
        # If the python script raises an exception, return empty grid
        default_return_value={"output_grid": [[]]},
        name="python_synthesis_"+task_name
    )(inputs)
    
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"arcagi_task_{task_name}",
        description=f"Program to solve ARC-AGI task {task_name}",
    )
    
    if verbose:
        print(f"🪛 Compiling solver for task {task_name}...")
    
    program.compile(
        reward=synalinks.rewards.RewardFunctionWrapper(
            in_mask=["output_grid"],
            fn=grid_similarity,
        ),
        optimizer=synalinks.optimizers.OMEGA(
            language_model=language_model,
            embedding_model=embedding_model,
            population_size=POPULATION_SIZE,
            k_nearest_fitter=K_NEAREST_FITTER,
            mutation_temperature=MUTATION_TEMPERATURE,
            crossover_temperature=CROSSOVER_TEMPERATURE,
            merging_rate=MERGING_RATE,
            name="omega_"+task_name
        ),
        metrics=[
            synalinks.metrics.MeanMetricWrapper(
                fn=synalinks.rewards.exact_match,
                in_mask=["output_grid"],
                name="exact_match",
            ),
        ],
    )
    
    return program


###############################################################################
## Program Library Management
###############################################################################


async def load_library(task_names: list, program_library: dict, library_folder: str):
    print(f"📚 Loading the '{library_folder}'...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task_progress = progress.add_task("Loading programs...", total=len(task_names))

        for task_name in task_names:
            task_checkpoint_filepath = os.path.join(library_folder, f"{task_name}.json")
            if os.path.exists(task_checkpoint_filepath):
                program = synalinks.Program.load(task_checkpoint_filepath)
                program_library[task_name] = program
            else:
                program = await build_and_compile_solver(
                    task_name=task_name,
                    language_model=language_model,
                    embedding_model=embedding_model,
                    python_script=get_default_python_script(),
                    seed_scripts=None,
                )
                program_library[task_name] = program
            
            progress.update(task_progress, advance=1)

    print(f"✅ Loaded {len(program_library)} programs")


async def is_training_task_completed(task_name: str, program_library: dict, x, y, verbose=False):
    metrics = await program_library[task_name].evaluate(x=x, y=y, verbose=0 if not verbose else "auto")
    if metrics["exact_match"] == 1.0:
        if verbose:
            print(f"✅ {task_name} completed")
        return True
    if verbose:
        print(f"❌ {task_name} not completed yet")
    return False


async def find_best_seed_scripts(task_name:str, program_library:dict, x, y, k=3, threshold=0.7):
    task_names = synalinks.datasets.arcagi.get_arcagi2_training_task_names()
    
    best_candidates = []
    
    print(f"🧠 Find seeds for {task_name}...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task_progress = progress.add_task("Finding seeds...", total=len(task_names))
        
        for task in task_names:
            metrics = await program_library[task].evaluate(x=x, y=y, verbose=0)
            python_script = program_library[task].trainable_variables[0].get("python_script")
            if metrics["reward"] > threshold:
                best_candidates.append(
                    {
                        "python_script": python_script,
                        "task_name": task,
                        **metrics,
                    }
                )
                
            progress.update(task_progress, advance=1)
    
    sorted_candidates = sorted(
        best_candidates,
        key=lambda x: x.get("reward", 0.0),
        reverse=True,
    )
    
    best_seed_scripts = [
        candidate.get("python_script")
        for candidate in sorted_candidates[:k]
    ]
    
    best_task_names = [
        (candidate.get("task_name"), round(candidate.get("reward"), 2))
        for candidate in sorted_candidates[:k]
    ]
    
    if best_seed_scripts:
        print(f"🧠 Found {len(best_seed_scripts)} seeds for {task_name} ({best_task_names})!")
    else:
        print(f"🧠 Found {len(best_seed_scripts)} seeds for {task_name}!")
    return best_seed_scripts
    
            
async def pretrain(epochs: int, batch_size:int, patience: int, repeat:int, concurrency: int) -> None:
    
    task_names = synalinks.datasets.arcagi.get_arcagi2_training_task_names()
    
    await load_library(task_names, PROGRAM_LIBRARY, PROGRAM_LIBRARY_FOLDER)
    
    tasks_to_learn = []
    
    nb_tasks_completed = 0
    
    print("🧠 Evaluating the remaining tasks to learn...")
        
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task_progress = progress.add_task("Evaluating tasks...", total=len(task_names))
        
        for task_name in task_names:
            (x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(task_name=task_name, arc_version=2)
            completed = await is_training_task_completed(
                task_name=task_name,
                program_library=PROGRAM_LIBRARY,
                x=x_test,
                y=y_test,
                verbose=False,
            )
            if not completed:
                tasks_to_learn.append(task_name)
            else:
                nb_tasks_completed += 1
            
            progress.update(task_progress, advance=1)
    
    completed_percentage = nb_tasks_completed / len(task_names) * 100.0
    print(f"🔥 {nb_tasks_completed} ({completed_percentage} %) training tasks completed")
    
    semaphore = threading.Semaphore(concurrency)
    
    def train_task_wrapper(task_name):
        """Wrapper to run async train_task in a thread"""
        with semaphore:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(train_task_async(task_name, epochs, batch_size, patience, repeat))
            finally:
                loop.close()
    
    async def train_task_async(task_name, epochs, batch_size, patience, repeat):
        task_checkpoint_filepath = os.path.join(PROGRAM_LIBRARY_FOLDER, f"{task_name}.json")

        program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
            filepath=task_checkpoint_filepath,
            monitor="val_reward",
            mode="max",
            save_best_only=True,
        )
        
        early_stopping_callback = synalinks.callbacks.EarlyStopping(
            monitor="val_reward",
            patience=patience,
        )

        (x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(
            task_name=task_name,
            arc_version=2,
            one_leave_out=ONE_LEAVE_OUT,
            permutation=PERMUTATION,
            curriculum_learning=CURRICULUM,
            repeat=repeat,
        )
        
        seed_scripts = await find_best_seed_scripts(
            task_name=task_name,
            program_library=PROGRAM_LIBRARY,
            x=x_test,
            y=y_test,
            k=NB_MAX_SEEDS,
            threshold=SEED_THRESHOLD,
        )
        
        if len(seed_scripts) < NB_MAX_SEEDS:
            seed_scripts.append(get_default_python_script())
        
        with PROGRAM_LIBRARY_LOCK:
            program = PROGRAM_LIBRARY[task_name]
        
        program.trainable_variables[0].update(
            {
                "seed_candidates": [{"python_script":seed_script} for seed_script in seed_scripts]
            }
        )
        
        print(f"🧠 Start learning {task_name}...")
        await program.fit(
            x=x_train,
            y=y_train,
            shuffle=not CURRICULUM,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                program_checkpoint_callback,
                early_stopping_callback,
            ]
        )
        await is_training_task_completed(
            task_name=task_name,
            program_library=PROGRAM_LIBRARY,
            x=x_test,
            y=y_test,
            verbose=True,
        )

    print(f"🧠 Learning (again) the {len(tasks_to_learn)} remaining tasks...")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(train_task_wrapper, task_name) for task_name in tasks_to_learn]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error training task: {e}")
    
    
async def solve(epochs: int, batch_size:int, patience: int, repeat:int, concurrency:int):
    
    task_names = synalinks.datasets.arcagi.get_arcagi2_training_task_names()
    
    await load_library(task_names, PROGRAM_LIBRARY, PROGRAM_LIBRARY_FOLDER)
    
    task_names = synalinks.datasets.arcagi.get_arcagi2_evaluation_task_names()
    
    await load_library(task_names, SUBMISSION_PROGRAM_LIBRARY, SUBMISSION_PROGRAM_LIBRARY_FOLDER)
    
    tasks_to_solve = []
    
    nb_tasks_completed = 0
    
    print("🧠 Evaluating the remaining tasks to solve...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task_progress = progress.add_task("Evaluating tasks...", total=len(task_names))
        
        for task_name in task_names:
            (x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(task_name=task_name, arc_version=2)
            
            completed = await is_training_task_completed(
                task_name=task_name,
                program_library=SUBMISSION_PROGRAM_LIBRARY,
                x=x_test,
                y=y_test,
                verbose=False,
            )
            if not completed:
                tasks_to_solve.append(task_name)
            else:
                nb_tasks_completed += 1
                
            progress.update(task_progress, advance=1)
    
    completed_percentage = nb_tasks_completed / len(task_names) * 100.0
    print(f"🔥 {nb_tasks_completed} ({completed_percentage} %) evaluation tasks completed")
    
    semaphore = threading.Semaphore(concurrency)
    
    def train_task_wrapper(task_name):
        """Wrapper to run async train_task in a thread"""
        with semaphore:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(train_task_async(task_name, epochs, batch_size, patience, repeat))
            finally:
                loop.close()
    
    async def train_task_async(task_name, epochs, batch_size, patience, repeat):
        task_checkpoint_filepath = os.path.join(SUBMISSION_PROGRAM_LIBRARY_FOLDER, f"{task_name}.json")

        program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
            filepath=task_checkpoint_filepath,
            monitor="val_reward",
            mode="max",
            save_best_only=True,
        )
        
        early_stopping_callback = synalinks.callbacks.EarlyStopping(
            monitor="val_reward",
            patience=patience,
        )

        (x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(
            task_name=task_name,
            arc_version=2,
            one_leave_out=ONE_LEAVE_OUT,
            permutation=PERMUTATION,
            curriculum_learning=CURRICULUM,
            repeat=repeat,
        )
        
        seed_scripts = await find_best_seed_scripts(
            task_name=task_name,
            program_library={**PROGRAM_LIBRARY, **SUBMISSION_PROGRAM_LIBRARY},
            x=x_test,
            y=y_test,
            k=NB_MAX_SEEDS,
            threshold=SEED_THRESHOLD,
        )
        
        if len(seed_scripts) < NB_MAX_SEEDS:
            seed_scripts.append(get_default_python_script())
        
        with SUBMISSION_PROGRAM_LIBRARY_LOCK:
            program = SUBMISSION_PROGRAM_LIBRARY[task_name]
        
        program.trainable_variables[0].update(
            {
                "seed_candidates": [{"python_script":seed_script} for seed_script in seed_scripts]
            }
        )
        
        print(f"🧠 Start solving {task_name}...")
        await program.fit(
            x=x_train,
            y=y_train,
            shuffle=not CURRICULUM,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                program_checkpoint_callback,
                early_stopping_callback,
            ]
        )
        
        await is_training_task_completed(
            task_name=task_name,
            program_library=SUBMISSION_PROGRAM_LIBRARY,
            x=x_test,
            y=y_test,
            verbose=True,
        )

    print(f"🧠 Solving the {len(tasks_to_solve)} tasks...")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(train_task_wrapper, task_name) for task_name in tasks_to_solve]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error solving task: {e}")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC-AGI Solver")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "solve"],
        required=True,
        help="Select mode: 'pretrain' or 'solve'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples per batch (default: 1)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times duplicating the trainset (default: 1 - no duplication)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent tasks (default: 1)",
    )
    args = parser.parse_args()
    
    print(f"🧠🔗 synalinks version: {synalinks.version()}")
    synalinks.clear_session()
    
    if args.mode == "pretrain":
        asyncio.run(
            pretrain(
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                repeat=args.repeat,
                concurrency=args.concurrency,
            ),
        )
    if args.mode == "solve":
        asyncio.run(
            solve(
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                repeat=args.repeat,
                concurrency=args.concurrency,
            ),
        )