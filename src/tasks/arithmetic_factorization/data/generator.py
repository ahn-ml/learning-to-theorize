import numpy as np
import h5py
import itertools
from typing import List, Tuple, Set
from collections import defaultdict, Counter
from tqdm import tqdm

def int_to_digits(value: int, max_digits: int = None) -> np.ndarray:
    """Encode a nonnegative integer as a zero-padded row of digits."""
    if value < 0:
        raise ValueError(f"Cannot convert negative value {value} to digits")

    # Convert to digits
    digit_list = [int(d) for d in str(value)]

    # Pad if necessary
    if max_digits is not None and len(digit_list) < max_digits:
        digit_list = [0] * (max_digits - len(digit_list)) + digit_list
    elif max_digits is not None and len(digit_list) > max_digits:
        raise ValueError(f"Value {value} has more than {max_digits} digits")

    return np.array([digit_list], dtype=np.int32)


def parse_primitive(primitive_str: str) -> tuple:
    """
    Parse primitive string like '+2', '*3' into (op, value).

    Args:
        primitive_str: String representation of operation (e.g., '+2', '*3')

    Returns:
        (op, value): Operator character and integer value
    """
    op = primitive_str[0]
    value = int(primitive_str[1:])
    return op, value


def apply_primitive_to_int(input_num: int, primitive_str: str) -> int:
    """
    Apply primitive operation to integer.

    Args:
        input_num: Input integer
        primitive_str: Operation like '+2', '*3'

    Returns:
        Output integer
    """
    op, value = parse_primitive(primitive_str)

    if op == '+':
        return input_num + value
    elif op == '-':
        return input_num - value
    elif op == '*':
        return input_num * value
    elif op == '/':
        return input_num // value
    else:
        raise ValueError(f"Unknown operation: {op}")


def apply_program(input_num: int, program: List[str]) -> int:
    """
    Apply a sequence of primitives to an integer.

    Args:
        input_num: Input integer
        program: List of primitive strings (e.g., ['+2', '*3'])

    Returns:
        Output integer after applying all primitives in sequence
    """
    result = input_num
    for primitive in program:
        result = apply_primitive_to_int(result, primitive)
    return result


def program_to_indices(program: List[str], primitives_list: List[str]) -> Tuple[int, ...]:
    """Convert a program (list of primitive strings) to tuple of sorted indices."""
    indices = tuple(sorted(primitives_list.index(p) for p in program))
    return indices


def indices_to_program(indices: Tuple[int, ...], primitives_list: List[str]) -> List[str]:
    """Convert tuple of indices to a program (list of primitive strings)."""
    return [primitives_list[i] for i in indices]


def is_held_out_program(program: List[str],
                         primitives_list: List[str],
                         held_out_set: Set[Tuple[int, ...]]) -> bool:
    """
    Check if a program matches any held-out combination.

    Since we use combinations_with_replacement (order doesn't matter),
    we compare sorted indices.
    """
    indices = program_to_indices(program, primitives_list)
    return indices in held_out_set


def generate_all_programs(primitives_list: List[str],
                          allowed_lengths: List[int]) -> List[List[str]]:
    """
    Generate all possible programs for the allowed lengths using combinations with repetition.

    Since operations like multiplication are commutative, we treat programs as multisets
    where order doesn't matter (e.g., *2*3 is same as *3*2).

    This uses nHr = C(n+r-1, r) combinations instead of n^r permutations.

    Args:
        primitives_list: List of primitives (e.g., ['*2', '*3', '*5', '*7'])
        allowed_lengths: List of allowed program lengths (e.g., [1, 2, 4])

    Returns:
        List of all possible programs (each program is a list of primitives)
    """
    all_programs = []

    for length in allowed_lengths:
        # Generate all combinations with repetition (order doesn't matter)
        # itertools.combinations_with_replacement gives sorted combinations
        programs_of_length = list(itertools.combinations_with_replacement(primitives_list, length))
        # Convert tuples to lists
        programs_of_length = [list(prog) for prog in programs_of_length]
        all_programs.extend(programs_of_length)

    return all_programs


def generate_io_pairs_for_program(program: List[str],
                                   max_input_value: int,
                                   max_digit_length: int,
                                   min_input_value: int = 0) -> List[Tuple[int, int]]:
    """
    Generate all valid (input, output) pairs for a given program.

    Args:
        program: List of primitive strings
        max_input_value: Maximum input value
        max_digit_length: Maximum number of digits for output
        min_input_value: Minimum input value

    Returns:
        List of (input, output) tuples
    """
    max_output_value = 10 ** max_digit_length - 1
    valid_pairs = []

    for input_num in range(min_input_value, max_input_value + 1):
        output_num = apply_program(input_num, program)

        # Check if output is valid
        if output_num >= 0 and output_num <= max_output_value:
            valid_pairs.append((input_num, output_num))

    return valid_pairs


def generate_evenly_distributed_dataset(primitives_list: List[str],
                                          allowed_lengths: List[int],
                                          samples_per_program: int,
                                          max_input_value: int,
                                          max_digit_length: int,
                                          min_input_value: int = 0,
                                          test_ratio: float = 0.1,
                                          held_out_combinations: List[Tuple[int, ...]] = None,
                                          seed: int = 42) -> Tuple[Tuple[np.ndarray, List[List[str]]],
                                                                     Tuple[np.ndarray, List[List[str]]],
                                                                     Tuple[np.ndarray, List[List[str]]]]:
    """
    Generate evenly distributed dataset where each possible program gets equal representation.

    Args:
        primitives_list: List of primitives (e.g., ['*2', '*3', '*5', '*7'])
        allowed_lengths: List of allowed program lengths (e.g., [1, 2, 4])
        samples_per_program: Number of sample pairs to generate per program
        max_input_value: Maximum input value
        max_digit_length: Maximum number of digits for output
        min_input_value: Minimum input value
        test_ratio: Fraction of data for test set
        held_out_combinations: List of tuples of primitive indices to hold out for test only.
                               e.g., [(0, 1), (2, 3)] holds out programs using primitives[0]+primitives[1]
                               These are sorted tuples since order doesn't matter.
        seed: Random seed for reproducibility

    Returns:
        (train_dataset, train_programs): Training data
        (test_dataset, test_programs): Test data (non-held-out programs)
        (held_out_dataset, held_out_programs): Held-out test data (held-out programs only)
    """
    np.random.seed(seed)

    # Convert held_out_combinations to set for fast lookup
    held_out_set = set(held_out_combinations) if held_out_combinations else set()

    print("=" * 70)
    print("Evenly Distributed Dataset Generation")
    print("=" * 70)
    print(f"Primitives: {primitives_list}")
    print(f"Allowed program lengths: {allowed_lengths}")
    print(f"Samples per program: {samples_per_program}")
    print(f"Input range: [{min_input_value}, {max_input_value}]")
    print(f"Max output digits: {max_digit_length}")
    print(f"Test ratio: {test_ratio}")

    if held_out_set:
        print(f"\nHeld-out combinations ({len(held_out_set)}):")
        for combo in sorted(held_out_set):
            prog = indices_to_program(combo, primitives_list)
            print(f"  {combo} -> {prog}")

    # Step 1: Generate all possible programs
    print("\n" + "=" * 70)
    print("Step 1: Generating all possible programs")
    print("=" * 70)

    all_programs = generate_all_programs(primitives_list, allowed_lengths)

    # Separate programs into regular and held-out
    regular_programs = []
    held_out_programs = []

    for prog in all_programs:
        if is_held_out_program(prog, primitives_list, held_out_set):
            held_out_programs.append(prog)
        else:
            regular_programs.append(prog)

    # Count programs by length
    programs_by_length = defaultdict(list)
    held_out_by_length = defaultdict(list)
    for prog in regular_programs:
        programs_by_length[len(prog)].append(prog)
    for prog in held_out_programs:
        held_out_by_length[len(prog)].append(prog)

    print("\nProgram statistics:")
    total_programs = 0
    total_held_out = 0
    for length in sorted(set(list(programs_by_length.keys()) + list(held_out_by_length.keys()))):
        regular_count = len(programs_by_length.get(length, []))
        held_out_count = len(held_out_by_length.get(length, []))
        total_programs += regular_count
        total_held_out += held_out_count
        print(f"  Length {length}: {regular_count} regular + {held_out_count} held-out = {regular_count + held_out_count} programs")
    print(f"  Total: {total_programs} regular + {total_held_out} held-out = {total_programs + total_held_out} unique programs")

    # Step 2: Generate I/O pairs for each program
    print("\n" + "=" * 70)
    print("Step 2: Generating I/O pairs for each program")
    print("=" * 70)

    program_to_pairs = {}
    programs_with_insufficient_pairs = []

    for program in tqdm(all_programs, desc="Generating I/O pairs", unit="program"):
        pairs = generate_io_pairs_for_program(
            program, max_input_value, max_digit_length, min_input_value
        )

        # Need at least 2 * samples_per_program pairs (2 inputs per sample, for train+test)
        min_required = int(2 * samples_per_program / (1 - test_ratio))

        if len(pairs) < min_required:
            programs_with_insufficient_pairs.append((program, len(pairs), min_required))

        program_to_pairs[tuple(program)] = pairs

    if programs_with_insufficient_pairs:
        print(f"\nWarning: {len(programs_with_insufficient_pairs)} programs have insufficient valid I/O pairs:")
        for prog, actual, required in programs_with_insufficient_pairs[:5]:
            prog_str = ','.join(prog)
            print(f"  {prog_str}: {actual} pairs (need {required})")
        if len(programs_with_insufficient_pairs) > 5:
            print(f"  ... and {len(programs_with_insufficient_pairs) - 5} more")

    # Step 3: Sample evenly from each program
    print("\n" + "=" * 70)
    print("Step 3: Sampling evenly from each program")
    print("=" * 70)

    # Helper function to sample from programs
    def sample_from_programs(programs_to_sample, desc_prefix=""):
        samples = []
        progs_used = []

        for program in tqdm(programs_to_sample, desc=f"{desc_prefix}Sampling", unit="program"):
            prog_tuple = tuple(program)
            pairs = program_to_pairs[prog_tuple]

            # Calculate how many pairs we need (train + test)
            total_needed = int(samples_per_program / (1 - test_ratio))

            # If not enough pairs, use all available
            if len(pairs) < total_needed * 2:  # *2 because we need 2 inputs per sample
                n_samples = len(pairs) // 2
            else:
                n_samples = total_needed

            # Randomly sample pairs without replacement
            pairs_copy = list(pairs)
            np.random.shuffle(pairs_copy)
            sampled_pairs = pairs_copy[:n_samples * 2]

            # Create samples: each sample is [in0, out0, in1, out1]
            for i in range(0, len(sampled_pairs) - 1, 2):
                in0, out0 = sampled_pairs[i]
                in1, out1 = sampled_pairs[i + 1]
                samples.append([in0, out0, in1, out1])
                progs_used.append(program)

        return samples, progs_used

    # Sample from regular programs
    regular_samples, regular_programs_used = sample_from_programs(regular_programs, "Regular ")
    print(f"\nGenerated {len(regular_samples)} samples from {len(regular_programs)} regular programs")
    if len(regular_programs) > 0:
        print(f"Average samples per regular program: {len(regular_samples) / len(regular_programs):.2f}")

    # Sample from held-out programs
    held_out_samples, held_out_programs_used = sample_from_programs(held_out_programs, "Held-out ")
    print(f"Generated {len(held_out_samples)} samples from {len(held_out_programs)} held-out programs")
    if len(held_out_programs) > 0:
        print(f"Average samples per held-out program: {len(held_out_samples) / len(held_out_programs):.2f}")

    # Step 4: Split regular samples into train and test
    print("\n" + "=" * 70)
    print("Step 4: Splitting into train, test, and held-out")
    print("=" * 70)

    # Shuffle regular samples
    if len(regular_samples) > 0:
        indices = np.arange(len(regular_samples))
        np.random.shuffle(indices)

        regular_samples_array = np.array(regular_samples)
        regular_samples_shuffled = regular_samples_array[indices]
        regular_programs_shuffled = [regular_programs_used[i] for i in indices]

        # Split regular samples into train and test
        n_test = int(len(regular_samples) * test_ratio)
        n_train = len(regular_samples) - n_test

        train_dataset = regular_samples_shuffled[:n_train]
        train_programs = regular_programs_shuffled[:n_train]

        test_dataset = regular_samples_shuffled[n_train:]
        test_programs = regular_programs_shuffled[n_train:]
    else:
        train_dataset = np.array([]).reshape(0, 4)
        train_programs = []
        test_dataset = np.array([]).reshape(0, 4)
        test_programs = []

    # Held-out samples (all go to held-out test set)
    if len(held_out_samples) > 0:
        held_out_indices = np.arange(len(held_out_samples))
        np.random.shuffle(held_out_indices)

        held_out_dataset = np.array(held_out_samples)[held_out_indices]
        held_out_programs_final = [held_out_programs_used[i] for i in held_out_indices]
    else:
        held_out_dataset = np.array([]).reshape(0, 4)
        held_out_programs_final = []

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples (regular): {len(test_dataset)}")
    print(f"Test samples (held-out): {len(held_out_dataset)}")

    # Show distribution by length
    def show_distribution(programs_list, name):
        if len(programs_list) == 0:
            print(f"\n{name} distribution: Empty")
            return
        lengths = [len(p) for p in programs_list]
        counts = Counter(lengths)
        print(f"\n{name} distribution by length:")
        for length in sorted(counts.keys()):
            count = counts[length]
            pct = 100 * count / len(programs_list)
            print(f"  Length {length}: {count} samples ({pct:.1f}%)")

    show_distribution(train_programs, "Train")
    show_distribution(test_programs, "Test (regular)")
    show_distribution(held_out_programs_final, "Test (held-out)")

    # Verify no overlap between train and held-out
    if held_out_set:
        print("\n" + "=" * 70)
        print("Verifying no program overlap")
        print("=" * 70)
        train_prog_set = set(tuple(p) for p in train_programs)
        held_out_prog_set = set(tuple(p) for p in held_out_programs_final)
        overlap = train_prog_set & held_out_prog_set
        if overlap:
            print(f"  WARNING: Found {len(overlap)} overlapping programs!")
        else:
            print("  ✓ No program overlap between train and held-out test")

    return (train_dataset, train_programs), (test_dataset, test_programs), (held_out_dataset, held_out_programs_final)


def save_dataset_to_h5(dataset: np.ndarray,
                       programs: List[List[str]],
                       h5_path: str,
                       max_digit_length: int,
                       metadata: dict = None):
    """
    Save arithmetic dataset to H5 file.

    Args:
        dataset: Array of shape (N, 4) where each row is [in0, out0, in1, out1]
        programs: List of programs (each is a list of primitives)
        h5_path: Path to save H5 file
        max_digit_length: Maximum number of digits
        metadata: Optional additional metadata to save
    """
    print(f"\nSaving dataset to {h5_path}...")

    if len(dataset) == 0:
        print("  Warning: Empty dataset, skipping save.")
        return

    with h5py.File(h5_path, 'w') as f:
        # Save metadata
        f.attrs['num_samples'] = len(dataset)
        f.attrs['dataset_length'] = len(dataset)
        f.attrs['max_digit_length'] = max_digit_length
        f.attrs['format'] = 'arithmetic_dataset_even'

        # Save additional metadata if provided
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)):
                    f.attrs[key] = str(value)
                else:
                    f.attrs[key] = value

        # Save each sample
        for i, (pair_data, program) in enumerate(tqdm(zip(dataset, programs),
                                                       total=len(dataset),
                                                       desc="  Saving samples",
                                                       unit="sample")):
            sample_group = f.create_group(f'sample_{i}')

            # Extract values
            in0, out0, in1, out1 = pair_data

            # Convert to digit arrays [1, max_digits]
            in0_digits = int_to_digits(int(in0), max_digits=max_digit_length)
            out0_digits = int_to_digits(int(out0), max_digits=max_digit_length)
            in1_digits = int_to_digits(int(in1), max_digits=max_digit_length)
            out1_digits = int_to_digits(int(out1), max_digits=max_digit_length)

            # Stack grids: [in0, out0, in1] -> shape [3, 1, max_digits]
            grids = np.stack([in0_digits, out0_digits, in1_digits], axis=0)

            # Save grids and answer
            sample_group.create_dataset('grids', data=grids, dtype='int32')
            sample_group.create_dataset('answer', data=out1_digits, dtype='int32')

            # Save program as comma-separated string
            program_str = ','.join(program)
            sample_group.attrs['program'] = program_str
            sample_group.attrs['program_length'] = len(program)
            sample_group.attrs['level'] = len(program)

            # Also save raw integers for reference
            sample_group.attrs['in0'] = int(in0)
            sample_group.attrs['out0'] = int(out0)
            sample_group.attrs['in1'] = int(in1)
            sample_group.attrs['out1'] = int(out1)

    print(f"  Saved {len(dataset)} samples")
