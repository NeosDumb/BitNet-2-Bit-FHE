from __future__ import annotations
import enum
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    NDArray: TypeAlias = 'np.ndarray[Any, Any]'
    from convert import LazyTensor

logger = logging.getLogger("convert_utils")

def permute(weights: NDArray, n_head: int, n_head_kv: int) -> NDArray:
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))

@dataclass(frozen=True)
class DataType:
    name: str
    dtype: np.dtype[Any]
    valid_conversions: list[str]

    def elements_to_bytes(self, n_elements: int) -> int:
        return n_elements * self.dtype.itemsize


@dataclass(frozen=True)
class UnquantizedDataType(DataType):
    pass


DT_F16  = UnquantizedDataType('F16',  dtype = np.dtype(np.float16), valid_conversions = ['F32', 'Q8_0'])
DT_F32  = UnquantizedDataType('F32',  dtype = np.dtype(np.float32), valid_conversions = ['F16', 'Q8_0', 'I2'])
DT_I32  = UnquantizedDataType('I32',  dtype = np.dtype(np.int16),   valid_conversions = [])
DT_BF16 = UnquantizedDataType('BF16', dtype = np.dtype(np.uint16),  valid_conversions = ['F32', 'F16', 'Q8_0'])
DT_I2   = UnquantizedDataType('I2',   dtype = np.dtype(np.uint8),   valid_conversions = ['F32', 'F16', 'Q8_0'])

@dataclass(frozen=True)
class QuantizedDataType(DataType):
    block_size: int
    quantized_dtype: np.dtype[Any]
    ggml_type: Any # gguf.GGMLQuantizationType

    def quantize(self, arr: NDArray) -> NDArray:
        raise NotImplementedError(f'Quantization for {self.name} not implemented')

    def elements_to_bytes(self, n_elements: int) -> int:
        assert n_elements % self.block_size == 0, f'Invalid number of elements {n_elements} for {self.name} with block size {self.block_size}'
        return self.quantized_dtype.itemsize * (n_elements // self.block_size)


@dataclass(frozen=True)
class Q8_0QuantizedDataType(QuantizedDataType):
    # Mini Q8_0 quantization in Python!
    def quantize(self, arr: NDArray) -> NDArray:
        assert arr.size % self.block_size == 0 and arr.size != 0, f'Bad array size {arr.size}'
        assert arr.dtype == np.float32, f'Bad array type {arr.dtype}'
        n_blocks = arr.size // self.block_size
        blocks = arr.reshape((n_blocks, self.block_size))
        # Much faster implementation of block quantization contributed by @Cebtenzzre

        def quantize_blocks_q8_0(blocks: NDArray) -> Iterable[tuple[Any, Any]]:
            d = abs(blocks).max(axis = 1) / np.float32(127)
            with np.errstate(divide = 'ignore'):
                qs = (blocks / d[:, None]).round()
            qs[d == 0] = 0
            yield from zip(d, qs)
        return np.fromiter(quantize_blocks_q8_0(blocks), count = n_blocks, dtype = self.quantized_dtype)

DT_Q8_0 = Q8_0QuantizedDataType('Q8_0',
                                dtype = np.dtype(np.float32), valid_conversions = [],
                                ggml_type = 8, # gguf.GGMLQuantizationType.Q8_0
                                block_size = 32,
                                quantized_dtype = np.dtype([('d', '<f2'), ('qs', 'i1', (32,))]))

class LlamaFileType(enum.IntEnum):
    ALL_F32              = 0
    MOSTLY_F16           = 1   # except 1d tensors
    MOSTLY_Q4_0          = 2   # except 1d tensors
    MOSTLY_Q8_0          = 7   # except 1d tensors
    MOSTLY_I2_S          = 40  # except 1d tensors

    def type_for_tensor(self, name: str, tensor: LazyTensor) -> DataType:
        dt = LLAMA_FILE_TYPE_TO_DATA_TYPE.get(self)
        if dt is None:
            raise ValueError(self)
        # Convert all 1D tensors to F32.  Most of the codebase that takes in 1D tensors only handles F32 tensors, and most of the outputs tensors are F32.
        #  Also The 1d tensors aren't much of a performance/size issue.  So instead of having to have separate F32 and F16 implementations of both, just convert everything to F32 for now.
        dt = dt if len(tensor.shape) > 1 else DT_F32
        if name == "token_embd.weight" or name == "output.weight":
            dt = DT_F32
        return dt


LLAMA_FILE_TYPE_TO_DATA_TYPE: dict[LlamaFileType, DataType] = {
    LlamaFileType.ALL_F32    : DT_F32,
    LlamaFileType.MOSTLY_F16 : DT_F16,
    LlamaFileType.MOSTLY_Q4_0: DT_I2,
    LlamaFileType.MOSTLY_Q8_0: DT_Q8_0,
    LlamaFileType.MOSTLY_I2_S: DT_I2,
}
