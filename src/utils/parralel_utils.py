# NOTE: `multiprocessing` is imported lazily inside the functions that use it so
# that this module remains importable on Emscripten/Pyodide (which has no
# multiprocessing). Native behaviour is unchanged — the imports happen the first
# time a multiprocessing-backed function is called.
import numpy as np
import time
import src.utils.CurveUtils as CurveUtils

# global pools
_GLOBAL_POOL = None
_GLOBAL_NUM_THREADS = None




def _find_min_shape_eval_worker(args):
    """
    Worker wrapper for pool.map to avoid Queue pickling
    and slow down from starting processes so many times over and over again.
    """
    from multiprocessing import shared_memory

    func, shm_names, shapes, dtypes, start_idx, end_idx, other_args = args

    shm_list = [shared_memory.SharedMemory(name=n) for n in shm_names]
    clark_coords = np.ndarray(shapes[0], dtype=dtypes[0], buffer=shm_list[0].buf)
    shape_points = np.ndarray(shapes[1], dtype=dtypes[1], buffer=shm_list[1].buf)
    endpoints = np.ndarray(shapes[2], dtype=dtypes[2], buffer=shm_list[2].buf)
    tip_positions = np.ndarray(shapes[3], dtype=dtypes[3], buffer=shm_list[3].buf)
    arc_lengths = np.ndarray(shapes[4], dtype=dtypes[4], buffer=shm_list[4].buf)
    arc_length_cumulative = np.ndarray(shapes[5], dtype=dtypes[5], buffer=shm_list[5].buf)
    
    
    min_value = np.inf
    best_idx = -1
    best_base_pos = None
    best_base_rot = None

    for i in range(start_idx, end_idx):
        shape_dict = {
            'clark_coords': clark_coords[i],
            'shape_points': shape_points[i],
            'endpoints': endpoints[i],
            'tip_position': tip_positions[i],
            'arc_length': arc_lengths[i],
            'arc_length_cumulative': arc_length_cumulative[i],
        }

        score, base_pos, base_rot = func(shape_dict, *other_args)
        if score < min_value:
            min_value = score
            best_idx = i
            best_base_pos = base_pos
            best_base_rot = base_rot

    for shm in shm_list:
        shm.close()
    
    

    return {
        "min_value": min_value,
        "best_idx": best_idx,
        "best_base_pos": best_base_pos,
        "best_base_rot": best_base_rot,
    }


def get_global_pool(num_threads):
    """
    Initializes or reuses a global process pool.
    Uses fork context for faster startup on Unix.

    This may be slow the first time but since we call the shape evaluator so many times,
    every suceeding call will be faster as we dont have to recreate the process pool each time.
    """
    from multiprocessing import get_context

    global _GLOBAL_POOL, _GLOBAL_NUM_THREADS
    if _GLOBAL_POOL is None or _GLOBAL_NUM_THREADS != num_threads:
        #ctx = get_context("fork") if hasattr(get_context("fork"), "Pool") else multiprocessing
        # Above line is recommended, but causes a shared memory issues on MacOS
        ctx = get_context('spawn')
        # print(f"[Init] Creating global pool with {num_threads} workers...")
        _GLOBAL_POOL = ctx.Pool(processes=num_threads)
        _GLOBAL_NUM_THREADS = num_threads
    return _GLOBAL_POOL



def parallel_find_min_shape_eval(func, shape_lib, other_args, num_threads=12):
    from multiprocessing import shared_memory

    pool = get_global_pool(num_threads)

    clark_coords = np.stack([s['clark_coords'] for s in shape_lib])
    shape_points = np.stack([s['shape_points'] for s in shape_lib])
    endpoints = np.stack([s['endpoints'] for s in shape_lib])
    tip_positions = np.stack([s['tip_position'] for s in shape_lib])
    arc_lengths = np.array([s['arc_length'] for s in shape_lib])
    arc_length_cumulative = np.stack([s['arc_length_cumulative'] for s in shape_lib])

    # Create shared memory for each field
    arrays = [clark_coords, shape_points, endpoints, tip_positions, arc_lengths, arc_length_cumulative]
    shms = []
    for arr in arrays:
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shm_np = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(shm_np, arr)
        shms.append(shm)
        

    shm_names = [s.name for s in shms]
    shapes = [a.shape for a in arrays]
    dtypes = [a.dtype for a in arrays]

    # chunk
    chunk_size = len(shape_lib) // num_threads
    tasks = []
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = len(shape_lib) if i == num_threads - 1 else (i + 1) * chunk_size
        tasks.append((func, shm_names, shapes, dtypes, start_idx, end_idx, other_args))

    # Pool the procs
    t0 = time.perf_counter()
    results = pool.map(_find_min_shape_eval_worker, tasks)
    # print(f"[Pool] Completed {num_threads} workers in {time.perf_counter() - t0:.3f}s")

    # Shape eval finds the min score
    min_value = np.inf
    best_return = None
    for res in results:
        if res["min_value"] < min_value:
            min_value = res["min_value"]
            best_return = res

    # Cleanup shared memory
    for shm in shms:
        shm.close()
        shm.unlink()

    best_idx = best_return["best_idx"]
    best_shape = {
        'clark_coords': clark_coords[best_idx],
        'shape_points': shape_points[best_idx],
        'endpoints': endpoints[best_idx],
        'tip_position': tip_positions[best_idx],
        'arc_length': arc_lengths[best_idx],
        'arc_length_cumulative': arc_length_cumulative[best_idx],
    }
    best_return["best_shape"] = best_shape

    return best_return



def _generate_single_shape_normal(args):
    """
    Helper function to generate a single shape configuration.
    This must be defined at top-level (not as a class method)
    for ProcessPoolExecutor pickling.
    """
    i, num_segments, std_dev, robot_state, shape_matcher_state, seed = args


    # Set unique random seed for this thread to avoid duplicates in parallel execution
    # np.random.seed(seed)
    
    # Reconstruct or use references as needed if robot and shape_matcher are stateless
    robot, shape_matcher = robot_state, shape_matcher_state

    # Generate Clark coordinates with cx1 fixed at 0
    clark_coords = np.zeros(num_segments * 2)
    clark_coords[:] = np.random.normal(0, std_dev, num_segments * 2)
    
    clark_coords = np.clip(clark_coords, -0.2, 0.2)

    # Compute shape
    endpoints, shape_points = robot.forward_kinematics(clark_coords)

    robot_cumulative = [0]
    for i in range(1, len(shape_points)):
        segment_length = np.linalg.norm(shape_points[i] - shape_points[i-1])
        robot_cumulative.append(robot_cumulative[-1] + segment_length)
    

    return {
        'clark_coords': clark_coords,
        'shape_points': shape_points,
        'endpoints': endpoints,
        'tip_position': endpoints[-1],
        'arc_length': CurveUtils.compute_arc_length(shape_points),
        'arc_length_cumulative': robot_cumulative
    }


def _generate_single_shape_uniform(args):
    """
    Helper function to generate a single shape configuration.
    This must be defined at top-level (not as a class method)
    for ProcessPoolExecutor pickling.
    """
    i, num_segments, bounds, robot_state, shape_matcher_state, seed = args

    # Set unique random seed for this thread to avoid duplicates in parallel execution
    np.random.seed(seed)
    
    # Reconstruct or use references as needed if robot and shape_matcher are stateless
    robot, shape_matcher = robot_state, shape_matcher_state

    # Generate Clark coordinates with cx1 fixed at 0
    clark_coords = np.zeros(num_segments * 2)
    #clark_coords[1:] = np.random.normal(0, std_dev, num_segments * 2 - 1)
    
    #clark_coords = np.clip(clark_coords, -0.2, 0.2)
    clark_coords[:] = np.random.uniform(-bounds, bounds, num_segments * 2)
    #clark_coords[:2] = np.random.uniform(-bounds, bounds, 2)


    # Compute shape
    endpoints, shape_points, _ = robot.forward_kinematics(clark_coords)

    robot_cumulative = [0]
    for i in range(1, len(shape_points)):
        segment_length = np.linalg.norm(shape_points[i] - shape_points[i-1])
        robot_cumulative.append(robot_cumulative[-1] + segment_length)
    

    return {
        'clark_coords': clark_coords,
        'shape_points': shape_points,
        'endpoints': endpoints,
        'tip_position': endpoints[-1],
        'arc_length': CurveUtils.compute_arc_length(shape_points),
        'arc_length_cumulative': robot_cumulative
    }
