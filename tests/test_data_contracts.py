import h5py
import pytest

from tasks.arithmetic_factorization.data.dataset import ArithmeticEpisodes


@pytest.mark.parametrize("value,expected", [("[1, 2, 'multiply']", [1, 2, "multiply"]),
                                           ("[]", [])])
def test_arithmetic_metadata_accepts_literal_lists(tmp_path, value, expected):
    path = tmp_path / "episodes.h5"
    with h5py.File(path, "w") as stream:
        stream.attrs["dataset_length"] = 1
        stream.create_group("sample_0").attrs["program"] = value
    dataset = ArithmeticEpisodes(path)
    try:
        assert dataset[0]["program"] == expected
    finally:
        dataset._file.close()


def test_arithmetic_metadata_does_not_evaluate_expressions(tmp_path):
    path = tmp_path / "episodes.h5"
    with h5py.File(path, "w") as stream:
        stream.attrs["dataset_length"] = 1
        stream.create_group("sample_0").attrs["program"] = "[sum([1, 2])]"
    dataset = ArithmeticEpisodes(path)
    try:
        with pytest.raises(ValueError):
            dataset[0]
    finally:
        dataset._file.close()
