import click
import asyncio
import json
import os

from rich.progress import Progress
import time

from enum import Enum
from typing import List

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from dotenv import load_dotenv

# import litellm
# litellm._turn_on_debug()

import synalinks

FINAL_SUBMISSION = {}

PROGRAM_LIBRARY = []

# Hyperparameters

POPULATION_SIZE = 5
K_NEAREST_FITTER = 5
MUTATION_TEMPERATURE = 1.0
CROSSOVER_TEMPERATURE = 1.0
MERGING_RATE = 0.02

PATIENCE = 10
EPOCHS = 50
BATCH_SIZE = 1

NB_MAX_SEEDS = 3
SEED_THRESHOLD = 0.7

# Parameters for building the datasets

PERMUTATION = False
CURRICULUM = True
MAX_TRAIN_SAMPLES = 10
REPEAT = 1

# Where to store the learned programs
PROGRAM_LIBRARY_FOLDER = "program_library"

SUBMISSION_PROGRAM_LIBRARY_FOLDER = "submission_programs"

fast_language_model = synalinks.LanguageModel(
    model="xai/grok-code-fast-1",
    caching=False,
)

smart_language_model = synalinks.LanguageModel(
    model="xai/grok-4-0709",
    caching=False,
)

embedding_model = synalinks.EmbeddingModel(
    model="ollama/mxbai-embed-large",
    caching=True, # cache it to speed up DNS
)


###############################################################################
## Data Models
###############################################################################


class Color(int, Enum):
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
    input_grid: List[List[Color]] = synalinks.Field(
        description="The input grid (list of integer list)",
    )
    output_grid: List[List[Color]] = synalinks.Field(
        description="The output grid (list of integer list)",
    )


class ARCAGIInput(synalinks.DataModel):
    examples: List[ARCAGITask] = synalinks.Field(
        description="A set of transformation examples",
    )
    input_grid: List[List[Color]] = synalinks.Field(
        description="The input grid (list of integer list)",
    )


class ARCAGIOutput(synalinks.DataModel):
    output_grid: List[List[Color]] = synalinks.Field(
        description="The output grid (list of integer list)",
    )


class PythonScript(synalinks.DataModel):
    python_script: str = synalinks.Field(
        description="The python script to solve ARCAGI task",
    )
    

###############################################################################
## Reward Function
###############################################################################


@synalinks.saving.register_synalinks_serializable()
async def grid_similarity(y_true, y_pred):
    """Compute a similarity between grids by counting the number of cells in common"""
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

        if min_height == 0 or min_width == 0:
            total_cells = max(
                true_shape[0] * true_shape[1], pred_shape[0] * pred_shape[1]
            )
            if total_cells == 0:
                return 1.0
            return 0.0

        true_overlap = true_array[:min_height, :min_width]
        pred_overlap = pred_array[:min_height, :min_width]
        diff = np.abs(true_overlap - pred_overlap)
        overlap_errors = np.count_nonzero(diff)
        true_only_cells = (true_shape[0] * true_shape[1]) - (min_height * min_width)
        pred_only_cells = (pred_shape[0] * pred_shape[1]) - (min_height * min_width)
        total_errors = overlap_errors + true_only_cells + pred_only_cells
        total_cells = max(true_shape[0] * true_shape[1], pred_shape[0] * pred_shape[1])
        total_cells = max(total_cells, min_height * min_width)

        reward = 1.0 - (total_errors / total_cells)

        return float(min(max(0.0,reward), 1.0))
    except Exception:
        return 0.0
    

###############################################################################
## Program library related utils
###############################################################################


def load_program_library():
    if not os.path.exists(PROGRAM_LIBRARY_FOLDER):
        os.mkdir(PROGRAM_LIBRARY_FOLDER)
    serialized_programs = [f for f in os.listdir(PROGRAM_LIBRARY_FOLDER) if f.endswith('.json')]
    print(f"📚 {len(serialized_programs)} programs detected in the library.")
    if len(serialized_programs) > 0:
        print("📚 Loading them...")
    for filename in serialized_programs:
        filepath = os.path.join(PROGRAM_LIBRARY_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            PROGRAM_LIBRARY.append(synalinks.Program.load(filepath))
    print("📚 Done.")
            

def program_exists(task_name, folder_path):
    return os.path.exists(
        os.path.join(
            folder_path,
            f"{task_name}.json",
        )
    )

async def find_best_seed_scripts(task_name, x, y, k=3, threshold=0.7):
    seed_candidates = []
    with Progress() as p:
        t = p.add_task(f"🧠 Look for seeds for {task_name}...", total=len(PROGRAM_LIBRARY))
        for program in PROGRAM_LIBRARY:
            p.update(t, advance=1)
            metrics = await program.evaluate(x=x, y=y, verbose=0)
            python_script = program.trainable_variables[0].get("python_script")
            if metrics.get("reward") >= threshold:
                seed_candidates.append(
                    {
                        "python_script": python_script,
                        **metrics
                    }
                )
    sorted_candidates = sorted(
        seed_candidates,
        key=lambda x: x.get("reward"),
        reverse=True,
    )
    
    # Take top k candidates
    best_seed_scripts = []
    for program in sorted_candidates[:k]:
        python_script = program.get("python_script")
        best_seed_scripts.append(python_script)
    
    print(f"🧠 Found {len(best_seed_scripts)} seeds for {task_name}!")
    return best_seed_scripts


###############################################################################
## Programs
###############################################################################


def get_default_python_script():
    return \
"""
def transform(inputs):
    # TODO implement the python function to transform the input grid into the output grid
    return {"output_grid": inputs.get("input_grid")}
    
result = transform(inputs)
"""

async def build_and_compile_solver(
    language_model,
    embedding_model,
    python_script,
    seed_scripts,
    task_name,
    verbose=False,
):
    if verbose:
        print(f"🔧 Building solver for task {task_name}...")
    inputs = synalinks.Input(
        data_model=ARCAGIInput,
    )
    outputs = await synalinks.PythonSynthesis(
        data_model=ARCAGIOutput,
        python_script=get_default_python_script(),
        seed_scripts=seed_scripts if seed_scripts else None,
        # If the python script raise an exception, it returns an empty grid
        default_return_value={"output_grid": [[]]},
        name="python_synthesis",
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
            name="reward",
        ),
        optimizer=synalinks.optimizers.OMEGA(
            language_model=language_model,
            embedding_model=embedding_model,
            population_size=POPULATION_SIZE,
            k_nearest_fitter=K_NEAREST_FITTER,
            mutation_temperature=MUTATION_TEMPERATURE,
            crossover_temperature=CROSSOVER_TEMPERATURE,
            merging_rate=MERGING_RATE,
            name="omega",
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


async def learn_task(task_name, epochs, patience):
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(
        task_name=task_name,
        curriculum_learning=CURRICULUM,
        arc_version=2,
    )

    seed_scripts = await find_best_seed_scripts(task_name, x=x_test, y=y_test, k=NB_MAX_SEEDS, threshold=SEED_THRESHOLD)
    
    program = await build_and_compile_solver(
        task_name=task_name,
        python_script=get_default_python_script(),
        seed_scripts=seed_scripts,
        language_model=fast_language_model,
        embedding_model=embedding_model,
    )

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
    
    print(f"🧠 Start learning task {task_name}...")
    history = await program.fit(
        x=x_train,
        y=y_train,
        shuffle = not CURRICULUM,
        validation_data=(x_test, y_test), 
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[
            program_checkpoint_callback,
            early_stopping_callback,
        ]
    )
    program.load(task_checkpoint_filepath)
    PROGRAM_LIBRARY.append(program)


async def solve_task(task_name, epochs, patience):
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(
        task_name=task_name,
        curriculum_learning=CURRICULUM,
        arc_version=2,
    )

    seed_scripts = await find_best_seed_scripts(task_name=task_name, x=x_train, y=y_train, k=NB_MAX_SEEDS, threshold=SEED_THRESHOLD)
    
    program = await build_and_compile_solver(
        task_name=task_name,
        python_script=get_default_python_script(),
        seed_scripts=seed_scripts,
        language_model=fast_language_model,
        embedding_model=embedding_model,
    )

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
    
    print(f"🧠 Start learning task {task_name}...")
    history = await program.fit(
        x=x_train,
        y=y_train,
        shuffle = not CURRICULUM,
        validation_split=0.2,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[
            program_checkpoint_callback,
            early_stopping_callback,
        ]
    )
    
    program.load(task_checkpoint_filepath)
    
    results = program.predict(x_test)
    
    FINAL_SUBMISSION[task_name] = []
    
    for result in results:
        FINAL_SUBMISSION[task_name].append(
            {
                "attempt_1": result.get("output_grid"),
                "attempt_2": result.get("output_grid"),
            }
        )


async def pretraining(epochs, patience, concurrency):
    load_program_library()
    non_visited_task_names = synalinks.datasets.arcagi.get_arcagi2_training_task_names()
    non_solved_tasks = [task_name for task_name in non_visited_task_names if not program_exists(task_name, PROGRAM_LIBRARY_FOLDER)]
    
    print(f"🧠 {len(non_solved_tasks)} tasks not solved yet!")
    
    task_queue = non_solved_tasks.copy()
    running_tasks = {}
    
    for _ in range(min(concurrency, len(task_queue))):
        task_name = task_queue.pop(0)
        running_tasks[asyncio.create_task(learn_task(task_name, epochs, patience))] = task_name
    
    while running_tasks:
        done, pending = await asyncio.wait(running_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        
        for completed_task in done:
            task_name = running_tasks.pop(completed_task)
            print(f"🧠 Completed {task_name}, {len(task_queue)} tasks remaining")
            
            if task_queue:
                new_task_name = task_queue.pop(0)
                running_tasks[asyncio.create_task(learn_task(new_task_name, epochs, patience))] = new_task_name


async def solving(epochs, patience, concurrency):
    load_program_library()
    task_queue = synalinks.datasets.arcagi.get_arcagi2_evaluation_task_names()
    
    print(f"🧠 {len(task_queue)} tasks to solve!")
    
    running_tasks = {}
    
    # Start initial batch
    for _ in range(min(concurrency, len(task_queue))):
        task_name = task_queue.pop(0)
        running_tasks[asyncio.create_task(solve_task(task_name, epochs, patience))] = task_name
    
    # As tasks complete, start new ones
    while running_tasks:
        done, pending = await asyncio.wait(running_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        
        for completed_task in done:
            task_name = running_tasks.pop(completed_task)
            print(f"🧠 Completed {task_name}, {len(task_queue)} tasks remaining")
            
            with open("submission.json", "w") as f:
                f.write(json.dumps(FINAL_SUBMISSION))
            
            # Start a new task if any remain
            if task_queue:
                new_task_name = task_queue.pop(0)
                running_tasks[asyncio.create_task(solve_task(new_task_name, epochs, patience))] = new_task_name


async def get_stats():
    task_names = synalinks.datasets.arcagi.get_arcagi2_evaluation_task_names()
    
    exact_match = 0.0
    grid_similarity = 0.0
    arc_score = 0.0
    
    for task_name in task_names:
        if program_exists(task_name, SUBMISSION_PROGRAM_LIBRARY_FOLDER):
            task_checkpoint_filepath = os.path.join(SUBMISSION_PROGRAM_LIBRARY_FOLDER, f"{task_name}.json")
            program = synalinks.Program.load(task_checkpoint_filepath)
            
            _, (x_test, y_test) = synalinks.arcagi.load_data()
            metrics = await program.evaluate(x=x_test, y=y_test)
            exact_match += metrics["exact_match"]
            grid_similarity += metrics["reward"]
            if metrics["exact_match"] == 1.0:
                arc_score += 1.0
    
    exact_match = exact_match / len(task_names)
    grid_similarity = grid_similarity / len(task_names)
    arc_score = arc_score / len(task_names)
    print(f"🧠🔗 Synalinks achieved: {arc_score} (arc_score), {exact_match} (exact_match), {grid_similarity} (grid_similarity)")


@click.group()
def arcagi():
    """Synalinks ARCAGI2 submission"""
    click.echo(f"🧠🔗 synalinks version: {synalinks.version()}")
    synalinks.clear_session()


@arcagi.command()
@click.option('--epochs', default=10, help='Number of epochs.')
@click.option('--patience', default=5, help='Number of epochs to wait before stopping if no progress.')
@click.option('--concurrency', default=20, help='Number of tasks to solve in parrallel.')
def pretrain(epochs, patience, concurrency):
    """run ARCAGI2 pretraining"""
    asyncio.run(pretraining(epochs, patience, concurrency))


@arcagi.command()
@click.option('--epochs', default=10, help='Number of epochs.')
@click.option('--patience', default=5, help='Number of epochs to wait before stopping if no progress.')
@click.option('--concurrency', default=1, help='Number of tasks to solve in parrallel.')
def pretrain(epochs, patience, concurrency):
    """run ARCAGI2 pretraining"""
    asyncio.run(pretraining(epochs, patience, concurrency))


@arcagi.command()
@click.option('--epochs', default=20, help='Number of epochs.')
@click.option('--patience', default=7, help='Number of epochs to wait before stopping if no progress.')
@click.option('--concurrency', default=1, help='Number of tasks to solve in parrallel.')
def solve(epochs, patience, concurrency):
    """run ARCAGI2 solver"""
    asyncio.run(solving(epochs, patience, concurrency))


@arcagi.command()
def stats():
    asyncio.run(get_stats())
    

if __name__ == '__main__':
    arcagi()